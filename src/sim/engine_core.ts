// Port of the request path in vllm/v1/engine/: InputProcessor (tokenize,
// validate, assign_request_id) -> EngineCoreClient (InprocClient here, with
// the ZMQ/msgpack hops narrated) -> EngineCore.preprocess_add_request /
// add_request -> step() or step_with_batch_queue() -> OutputProcessor.

import { AsyncScheduler } from './async_scheduler'
import { type EventLog, NULL_LOG } from './events'
import { get_request_block_hasher } from './kv_cache_utils'
import type { EngineCoreOutputs, GrammarOutput, ModelRunnerOutput, SchedulerOutput } from './output'
import { type EngineCoreRequest, Request, type SamplingParams } from './request'
import { Scheduler, type SchedulerConfig } from './scheduler'
import { REF } from './source_refs'
import { StructuredOutputManager } from './structured_output'
import { ToyTokenizer } from './tokenizer'
import { GPUModelRunner, type SpecDecodeConfig, type TokenOracle } from './worker/model_runner'

// ------------------------------------------------------------ front end

let uuid_counter = 0
function random_uuid8(): string {
  uuid_counter += 1
  return (0x9a3f2b10 + uuid_counter * 0x1f3d).toString(16).slice(0, 8)
}

export class InputProcessor {
  tokenizer: ToyTokenizer
  log: EventLog
  constructor(tokenizer: ToyTokenizer, log: EventLog = NULL_LOG) {
    this.tokenizer = tokenizer
    this.log = log
  }

  process_inputs(request_id: string, prompt: string | number[], params: SamplingParams, arrival_time: number, priority = 0): EngineCoreRequest {
    const prompt_token_ids = typeof prompt === 'string' ? this.tokenizer.encode(prompt) : prompt.slice()
    if (params.max_tokens <= 0) throw new Error('max_tokens must be > 0')
    const req: EngineCoreRequest = {
      request_id,
      external_req_id: request_id,
      prompt_token_ids,
      sampling_params: { ...params, eos_token_id: params.eos_token_id ?? 2 },
      arrival_time,
      priority,
      client_index: 0,
    }
    this.log.emit(
      'InputProcessor',
      'process_inputs',
      `InputProcessor.process_inputs(${request_id}): ${typeof prompt === 'string' ? `"${prompt}" tokenized to ` : ''}${prompt_token_ids.length} token(s) [${prompt_token_ids.join(', ')}] -> EngineCoreRequest (msgspec Struct)`,
      { request_id, prompt_token_ids, max_tokens: params.max_tokens },
      REF.InputProcessor_process_inputs,
    )
    return req
  }

  /** request_id becomes `<external>-<8 hex>`; the original is kept in external_req_id. */
  assign_request_id(req: EngineCoreRequest): void {
    req.external_req_id = req.request_id
    req.request_id = `${req.external_req_id}-${random_uuid8()}`
    this.log.emit('InputProcessor', 'assign_request_id', `assign_request_id: "${req.external_req_id}" -> internal "${req.request_id}"`, { external_req_id: req.external_req_id, request_id: req.request_id }, REF.InputProcessor_assign_request_id)
  }
}

export interface RequestOutput {
  request_id: string
  external_req_id: string
  token_ids: number[]
  text: string
  finished: boolean
  finish_reason: string | null
}

export class OutputProcessor {
  tokenizer: ToyTokenizer
  log: EventLog
  request_states = new Map<string, RequestOutput>()
  constructor(tokenizer: ToyTokenizer, log: EventLog = NULL_LOG) {
    this.tokenizer = tokenizer
    this.log = log
  }
  add_request(req: EngineCoreRequest): void {
    this.request_states.set(req.request_id, { request_id: req.request_id, external_req_id: req.external_req_id, token_ids: [], text: '', finished: false, finish_reason: null })
  }
  process_outputs(outputs: EngineCoreOutputs): RequestOutput[] {
    const result: RequestOutput[] = []
    for (const o of outputs.outputs) {
      const st = this.request_states.get(o.request_id)
      if (!st) continue
      st.token_ids.push(...o.new_token_ids)
      st.text = this.tokenizer.decode(st.token_ids)
      if (o.finish_reason) {
        st.finished = true
        st.finish_reason = o.finish_reason
      }
      this.log.emit(
        'OutputProcessor',
        'process_outputs',
        `OutputProcessor: ${st.external_req_id} += [${o.new_token_ids.map((t) => this.tokenizer.id_to_token(t)).join(' ')}] (IncrementalDetokenizer)${o.finish_reason ? ` -> finished (${o.finish_reason})` : ''}`,
        { request_id: o.request_id, new_token_ids: o.new_token_ids, text: st.text, finished: st.finished },
        REF.OutputProcessor,
      )
      result.push({ ...st, token_ids: st.token_ids.slice() })
    }
    return result
  }
}

// ------------------------------------------------------------ engine core

export interface EngineConfig extends SchedulerConfig {
  spec: SpecDecodeConfig | null
  /** vllm_config.max_concurrent_batches: 1 = plain step(), >1 = step_with_batch_queue(). */
  batch_queue_size: number
  grammar_compile_delay_steps: number
  seed?: number
}

interface BatchQueueEntry {
  /** Future[ModelRunnerOutput] from sample_tokens (resolved eagerly here). */
  future: ModelRunnerOutput | null
  scheduler_output: SchedulerOutput
  /** Future from execute_model (None on success). */
  exec_future: null
}

export class EngineCore {
  cfg: EngineConfig
  log: EventLog
  tokenizer: ToyTokenizer
  scheduler: Scheduler
  structured_output_manager: StructuredOutputManager
  model_runner: GPUModelRunner
  batch_queue: BatchQueueEntry[] | null
  batch_queue_size: number
  async_scheduling: boolean
  private block_hasher

  constructor(cfg: EngineConfig, tokenizer: ToyTokenizer, oracle: TokenOracle, log: EventLog = NULL_LOG) {
    this.cfg = cfg
    this.log = log
    this.tokenizer = tokenizer
    this.structured_output_manager = new StructuredOutputManager(tokenizer, cfg.grammar_compile_delay_steps, log)
    this.scheduler = cfg.async_scheduling ? new AsyncScheduler(cfg, this.structured_output_manager, log) : new Scheduler(cfg, this.structured_output_manager, log)
    this.model_runner = new GPUModelRunner(
      { block_size: cfg.block_size, num_gpu_blocks: cfg.num_gpu_blocks, max_model_len: cfg.max_model_len, vocab_size: () => tokenizer.vocab_size, spec: cfg.spec, seed: cfg.seed },
      oracle,
      log,
    )
    this.model_runner.resume_token_ids = (id) => {
      const r = this.scheduler.requests.get(id)
      if (!r) throw new Error(`resume of unknown request ${id}`)
      return { prompt: r.prompt_token_ids, output: r.output_token_ids.slice(), sampling_params: r.sampling_params }
    }
    this.block_hasher = cfg.enable_prefix_caching ? get_request_block_hasher(cfg.block_size) : null
    this.async_scheduling = cfg.async_scheduling
    this.batch_queue_size = cfg.batch_queue_size
    this.batch_queue = this.batch_queue_size > 1 ? [] : null
    log.emit(
      'EngineCore',
      'init',
      `EngineCore: executor=UniProcExecutor, scheduler=${cfg.async_scheduling ? 'AsyncScheduler' : 'Scheduler'}, KV cache ${cfg.num_gpu_blocks} blocks x ${cfg.block_size} tokens, batch_queue_size=${this.batch_queue_size} -> step_fn=${this.batch_queue ? 'step_with_batch_queue' : 'step'}`,
      { async_scheduling: cfg.async_scheduling, batch_queue_size: this.batch_queue_size },
      REF.EngineCore_batch_queue,
    )
  }

  /** Runs on the ZMQ input thread in EngineCoreProc: EngineCoreRequest -> Request. */
  preprocess_add_request(req: EngineCoreRequest): Request {
    const r = Request.from_engine_core_request(req, this.block_hasher)
    this.log.emit(
      'EngineCore',
      'preprocess_add_request',
      `preprocess_add_request: EngineCoreRequest -> Request(${req.request_id}) status ${r.use_structured_output ? 'WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR' : 'WAITING'}${r.block_hashes.length ? `, ${r.block_hashes.length} full-block hash(es) precomputed [${r.block_hashes.join(', ')}]` : ''}`,
      { request_id: req.request_id, block_hashes: r.block_hashes.slice() },
      REF.EngineCore_preprocess_add_request,
    )
    if (r.use_structured_output) this.structured_output_manager.grammar_init(r)
    return r
  }

  add_request(request: Request): void {
    this.log.emit('EngineCore', 'add_request', `EngineCore.add_request(${request.request_id}) -> scheduler.add_request`, { request_id: request.request_id }, REF.EngineCore_add_request)
    this.scheduler.add_request(request)
  }

  has_requests(): boolean {
    return this.scheduler.has_requests() || (this.batch_queue !== null && this.batch_queue.length > 0)
  }

  /** EngineCore.step(): schedule -> execute_model -> bitmask -> sample_tokens -> update. */
  step(): EngineCoreOutputs | null {
    if (!this.scheduler.has_requests()) return null
    this.structured_output_manager.tick(this.scheduler.current_step + 1)
    this.log.setPhase('schedule')
    const scheduler_output = this.scheduler.schedule()
    this.log.setPhase('execute_model')
    this.log.emit('Executor', 'execute_model', `model_executor.execute_model(scheduler_output, non_block=True) -> Worker.execute_model (RPC 1 of 2)`, { total_num_scheduled_tokens: scheduler_output.total_num_scheduled_tokens }, REF.Executor_execute_model)
    const model_output_or_none = this.model_runner.execute_model(scheduler_output)
    this.log.setPhase('grammar_bitmask')
    const grammar_output = this.scheduler.get_grammar_bitmask(scheduler_output)
    this.log.setPhase('sample_tokens')
    let model_output: ModelRunnerOutput
    if (model_output_or_none === null) {
      this.log.emit('Executor', 'sample_tokens', `model_executor.sample_tokens(grammar_output) -> Worker.sample_tokens (RPC 2 of 2)`, { has_grammar: grammar_output !== null }, REF.Executor_sample_tokens)
      model_output = this.model_runner.sample_tokens(grammar_output)
    } else {
      model_output = model_output_or_none
    }
    this.log.setPhase('update_from_output')
    const outputs = this.scheduler.update_from_output(scheduler_output, model_output)
    this.log.setPhase('post_step')
    this.post_step(scheduler_output.total_num_scheduled_tokens > 0)
    return outputs
  }

  post_step(model_executed: boolean): void {
    if (this.cfg.spec && !this.async_scheduling && model_executed) {
      const drafts = this.model_runner.take_draft_token_ids()
      if (drafts) {
        this.log.emit('EngineCore', 'post_step', `post_step: take_draft_token_ids -> scheduler.update_draft_token_ids for [${drafts.req_ids.join(', ')}]`, { req_ids: drafts.req_ids }, REF.EngineCore_post_step)
        this.scheduler.update_draft_token_ids(drafts)
      }
    }
  }

  /**
   * EngineCore.step_with_batch_queue(): fill the batch queue first; only block
   * on the oldest batch when the queue is full or nothing else can be scheduled.
   */
  step_with_batch_queue(): EngineCoreOutputs | null {
    const batch_queue = this.batch_queue
    if (batch_queue === null) throw new Error('no batch queue')
    if (batch_queue.length >= this.batch_queue_size) throw new Error('batch queue overflow')
    let model_executed = false
    let deferred_scheduler_output: SchedulerOutput | null = null

    if (this.scheduler.has_requests()) {
      this.structured_output_manager.tick(this.scheduler.current_step + 1)
      this.log.setPhase('schedule')
      const scheduler_output = this.scheduler.schedule()
      this.log.setPhase('execute_model')
      this.log.emit('Executor', 'execute_model', `execute_model(scheduler_output, non_block=True): step ${scheduler_output.step} launched on the GPU`, { step: scheduler_output.step }, REF.Executor_execute_model)
      this.model_runner.execute_model(scheduler_output)
      model_executed = scheduler_output.total_num_scheduled_tokens > 0
      let future: ModelRunnerOutput | null = null
      if (!model_executed) {
        future = { req_ids: [], req_id_to_index: {}, sampled_token_ids: [] }
      } else if (!scheduler_output.pending_structured_output_tokens) {
        this.log.setPhase('grammar_bitmask')
        const grammar_output = this.scheduler.get_grammar_bitmask(scheduler_output)
        this.log.setPhase('sample_tokens')
        this.log.emit('Executor', 'sample_tokens', `sample_tokens(grammar_output, non_block=True): step ${scheduler_output.step} sampling queued`, { step: scheduler_output.step }, REF.Executor_sample_tokens)
        future = this.model_runner.sample_tokens(grammar_output)
      } else {
        deferred_scheduler_output = scheduler_output
        this.log.emit('EngineCore', 'defer_sampling', `step ${scheduler_output.step}: structured-output request has placeholders in flight -> defer sample_tokens until the previous step's tokens are known`, { step: scheduler_output.step }, REF.EngineCore_step_with_batch_queue)
      }
      if (!deferred_scheduler_output) {
        batch_queue.unshift({ future, scheduler_output, exec_future: null })
        this.log.emit(
          'EngineCore',
          'batch_queue_push',
          `batch_queue.appendleft(step ${scheduler_output.step}) -> depth ${batch_queue.length}/${this.batch_queue_size}`,
          { step: scheduler_output.step, depth: batch_queue.length, queue: batch_queue.map((b) => b.scheduler_output.step) },
          REF.EngineCore_step_with_batch_queue,
        )
        if (batch_queue.length < this.batch_queue_size && (model_executed || this.scheduler.has_requests())) {
          this.log.emit('EngineCore', 'batch_queue_no_block', `queue not full: return without blocking on step ${scheduler_output.step}'s result (schedule ahead)`, { depth: batch_queue.length }, REF.EngineCore_step_with_batch_queue)
          return null
        }
      }
    } else if (batch_queue.length === 0) {
      return null
    }

    const entry = batch_queue.pop() as BatchQueueEntry
    this.log.setPhase('update_from_output')
    this.log.emit('EngineCore', 'batch_queue_pop', `batch_queue.pop(): block on step ${entry.scheduler_output.step}'s future.result()`, { step: entry.scheduler_output.step, depth: batch_queue.length }, REF.EngineCore_step_with_batch_queue)
    if (entry.future === null) throw new Error('unexpected error')
    const engine_core_outputs = this.scheduler.update_from_output(entry.scheduler_output, entry.future)

    if (deferred_scheduler_output) {
      if (this.cfg.spec) {
        const drafts = this.model_runner.take_draft_token_ids()
        void drafts // update_draft_token_ids_in_output not modelled
      }
      this.log.setPhase('grammar_bitmask')
      const grammar_output: GrammarOutput | null = this.scheduler.get_grammar_bitmask(deferred_scheduler_output)
      this.log.setPhase('sample_tokens')
      const future = this.model_runner.sample_tokens(grammar_output)
      batch_queue.unshift({ future, scheduler_output: deferred_scheduler_output, exec_future: null })
    }
    this.log.setPhase('post_step')
    this.post_step(model_executed)
    return engine_core_outputs
  }

  step_fn(): EngineCoreOutputs | null {
    return this.batch_queue === null ? this.step() : this.step_with_batch_queue()
  }
}
