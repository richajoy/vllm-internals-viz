// Port of vllm/v1/core/sched/scheduler.py for a single full-attention KV
// group, no encoder inputs, no KV connectors, no LoRA, no DP throttling.

import { type EventLog, NULL_LOG } from './events'
import { KVCacheManager, type KVCacheBlocks } from './kv_cache_manager'
import type {
  CachedRequestData,
  DraftTokenIds,
  EngineCoreOutput,
  EngineCoreOutputs,
  GrammarOutput,
  ModelRunnerOutput,
  NewRequestData,
  SchedulerOutput,
} from './output'
import { type Request, RequestStatus, RequestStatusName } from './request'
import { type RequestQueue, type SchedulingPolicy, create_request_queue } from './request_queue'
import { REF } from './source_refs'
import type { StructuredOutputManager } from './structured_output'

export interface SchedulerConfig {
  max_num_seqs: number
  max_num_batched_tokens: number
  max_model_len: number
  enable_chunked_prefill: boolean
  long_prefill_token_threshold: number
  policy: SchedulingPolicy
  /** Admission gate: whole prompt must fit in free blocks (default True in vLLM). */
  scheduler_reserve_full_isl: boolean
  async_scheduling: boolean
  block_size: number
  num_gpu_blocks: number
  enable_prefix_caching: boolean
  num_speculative_tokens: number
  /** ngram/medusa: 0; eagle/mtp/draft_model: num_speculative_tokens. */
  num_lookahead_tokens: number
  watermark?: number
}

export const DEFAULT_SCHEDULER_CONFIG: SchedulerConfig = {
  max_num_seqs: 128,
  max_num_batched_tokens: 2048,
  max_model_len: 2048,
  enable_chunked_prefill: true,
  long_prefill_token_threshold: 0,
  policy: 'fcfs',
  scheduler_reserve_full_isl: true,
  async_scheduling: false,
  block_size: 16,
  num_gpu_blocks: 128,
  enable_prefix_caching: true,
  num_speculative_tokens: 0,
  num_lookahead_tokens: 0,
}

export class Scheduler {
  cfg: SchedulerConfig
  log: EventLog
  max_model_len: number
  max_num_running_reqs: number
  max_num_scheduled_tokens: number
  policy: SchedulingPolicy
  num_spec_tokens: number
  num_lookahead_tokens: number
  num_sampled_tokens_per_step = 1
  kv_cache_manager: KVCacheManager
  structured_output_manager: StructuredOutputManager | null
  requests = new Map<string, Request>()
  waiting: RequestQueue
  skipped_waiting: RequestQueue
  running: Request[] = []
  finished_req_ids = new Set<string>()
  reset_preempted_req_ids = new Set<string>()
  current_step = 0
  prev_step_scheduled_req_ids = new Set<string>()

  constructor(cfg: SchedulerConfig, structured_output_manager: StructuredOutputManager | null = null, log: EventLog = NULL_LOG) {
    this.cfg = cfg
    this.log = log
    this.max_model_len = cfg.max_model_len
    this.max_num_running_reqs = cfg.max_num_seqs
    this.max_num_scheduled_tokens = cfg.max_num_batched_tokens
    this.policy = cfg.policy
    this.num_spec_tokens = cfg.num_speculative_tokens
    this.num_lookahead_tokens = cfg.num_lookahead_tokens
    this.structured_output_manager = structured_output_manager
    this.kv_cache_manager = new KVCacheManager(
      {
        num_gpu_blocks: cfg.num_gpu_blocks,
        block_size: cfg.block_size,
        max_model_len: cfg.max_model_len,
        enable_caching: cfg.enable_prefix_caching,
        watermark: cfg.watermark,
      },
      log,
    )
    this.waiting = create_request_queue(cfg.policy)
    this.skipped_waiting = create_request_queue(cfg.policy)
  }

  // ---------------------------------------------------------------- add

  add_request(request: Request): void {
    this.requests.set(request.request_id, request)
    if (this._is_blocked_waiting_status(request.status)) this.skipped_waiting.add_request(request)
    else this.waiting.add_request(request)
    this.log.emit(
      'Scheduler',
      'add_request',
      `Scheduler.add_request(${request.request_id}): status ${RequestStatusName[request.status]}, ${request.num_prompt_tokens} prompt tokens -> ${this.policy === 'fcfs' ? 'waiting.append' : 'waiting.heappush'}`,
      { request_id: request.request_id, status: RequestStatusName[request.status], num_prompt_tokens: request.num_prompt_tokens },
      REF.Scheduler_queues,
    )
  }

  has_requests(): boolean {
    return this.waiting.length > 0 || this.skipped_waiting.length > 0 || this.running.length > 0
  }

  has_unfinished_requests(): boolean {
    return this.has_requests()
  }

  get_num_unfinished_requests(): number {
    return this.waiting.length + this.skipped_waiting.length + this.running.length
  }

  // ----------------------------------------------------------- schedule

  schedule(): SchedulerOutput {
    this.current_step += 1
    const cfg = this.cfg
    // NOTE(woosuk): no "decode phase" nor "prefill phase"; each request has
    // num_computed_tokens and num_tokens_with_spec and the scheduler assigns
    // tokens so the former catches up with the latter.
    const scheduled_new_reqs: Request[] = []
    const scheduled_resumed_reqs: Request[] = []
    const scheduled_running_reqs: Request[] = []
    const preempted_reqs: Request[] = []
    const req_to_new_blocks = new Map<string, KVCacheBlocks>()
    const num_scheduled_tokens: Record<string, number> = {}
    let token_budget = this.max_num_scheduled_tokens
    const scheduled_spec_decode_tokens: Record<string, number[]> = {}
    let prefill_scheduled = false

    this.log.emit(
      'Scheduler',
      'schedule_start',
      `schedule(): step ${this.current_step}, token_budget=${token_budget}, running=${this.running.length}, waiting=${this.waiting.length}${this.skipped_waiting.length ? `, skipped_waiting=${this.skipped_waiting.length}` : ''}`,
      { step: this.current_step, token_budget, running: this.running.map((r) => r.request_id), waiting: this.waiting.toArray().map((r) => r.request_id) },
      REF.Scheduler_no_phases_note,
    )

    // First, schedule the RUNNING requests.
    let req_index = 0
    while (req_index < this.running.length && token_budget > 0) {
      const request = this.running[req_index]

      if (
        request.num_output_placeholders > 0 &&
        request.num_computed_tokens + 2 - request.num_output_placeholders >= request.num_prompt_tokens + request.max_tokens
      ) {
        // Async scheduling: the previous step is guaranteed to hit max_tokens.
        this.log.emit('Scheduler', 'skip_running', `running ${request.request_id}: skip, previous in-flight step already reaches max_tokens`, { request_id: request.request_id }, REF.Scheduler_running_loop)
        req_index += 1
        continue
      }

      let num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
      const before_clamp = num_new_tokens
      if (0 < cfg.long_prefill_token_threshold && cfg.long_prefill_token_threshold < num_new_tokens) {
        num_new_tokens = cfg.long_prefill_token_threshold
      }
      num_new_tokens = Math.min(num_new_tokens, token_budget)
      num_new_tokens = Math.min(num_new_tokens, this.max_model_len - request.num_computed_tokens - this.num_sampled_tokens_per_step)

      if (num_new_tokens === 0) {
        // `continue` rather than `break`: intentionally not strict FCFS.
        req_index += 1
        continue
      }
      this.log.emit(
        'Scheduler',
        'running_num_new_tokens',
        `running ${request.request_id}: num_new_tokens = num_tokens_with_spec(${request.num_tokens_with_spec}) + placeholders(${request.num_output_placeholders}) - num_computed(${request.num_computed_tokens}) = ${before_clamp}${before_clamp !== num_new_tokens ? ` -> clamped to ${num_new_tokens}` : ''}`,
        { request_id: request.request_id, num_new_tokens, num_tokens_with_spec: request.num_tokens_with_spec, num_output_placeholders: request.num_output_placeholders, num_computed_tokens: request.num_computed_tokens, token_budget },
        REF.Scheduler_num_new_tokens_running,
      )

      let new_blocks: KVCacheBlocks | null = null
      for (;;) {
        new_blocks = this.kv_cache_manager.allocate_slots(request, num_new_tokens, {
          num_lookahead_tokens: this.num_lookahead_tokens,
        })
        if (new_blocks !== null) break

        // Preempt the lowest-priority request.
        let preempted_req: Request
        if (this.policy === 'priority') {
          preempted_req = this.running.reduce((worst, r) =>
            r.priority > worst.priority || (r.priority === worst.priority && r.arrival_time > worst.arrival_time) ? r : worst,
          )
          this.running.splice(this.running.indexOf(preempted_req), 1)
          const idx = scheduled_running_reqs.indexOf(preempted_req)
          if (idx !== -1) {
            const pid = preempted_req.request_id
            scheduled_running_reqs.splice(idx, 1)
            token_budget += num_scheduled_tokens[pid]
            delete num_scheduled_tokens[pid]
            req_to_new_blocks.delete(pid)
            delete scheduled_spec_decode_tokens[pid]
            req_index -= 1
          }
        } else {
          preempted_req = this.running.pop() as Request
        }
        this._preempt_request(preempted_req)
        preempted_reqs.push(preempted_req)
        if (preempted_req === request) break
      }
      if (new_blocks === null) break

      scheduled_running_reqs.push(request)
      prefill_scheduled ||= request.is_prefill_chunk
      req_to_new_blocks.set(request.request_id, new_blocks)
      num_scheduled_tokens[request.request_id] = num_new_tokens
      token_budget -= num_new_tokens
      req_index += 1
      this.log.emit(
        'Scheduler',
        'scheduled_running',
        `scheduled running ${request.request_id}: ${num_new_tokens} token(s), new blocks [${new_blocks.get_block_ids()[0].join(', ')}], budget left ${token_budget}`,
        { request_id: request.request_id, num_new_tokens, new_block_ids: new_blocks.get_block_ids()[0], token_budget },
        REF.Scheduler_running_loop,
      )

      if (request.spec_token_ids.length > 0) {
        const num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens - request.num_tokens - request.num_output_placeholders
        if (num_scheduled_spec_tokens > 0) {
          let spec_token_ids = request.spec_token_ids
          if (spec_token_ids.length > num_scheduled_spec_tokens) spec_token_ids = spec_token_ids.slice(0, num_scheduled_spec_tokens)
          scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
          this.log.emit(
            'Scheduler',
            'scheduled_spec_tokens',
            `spec decode ${request.request_id}: scheduling ${spec_token_ids.length} draft token(s) [${spec_token_ids.join(', ')}] for verification`,
            { request_id: request.request_id, spec_token_ids },
            REF.Scheduler_spec_tokens,
          )
        }
        request.spec_token_ids = []
      }
    }

    // Next, schedule the WAITING requests.
    if (preempted_reqs.length === 0) {
      const step_skipped_waiting = create_request_queue(this.policy)

      while ((this.waiting.length > 0 || this.skipped_waiting.length > 0) && token_budget > 0) {
        if (this.running.length >= this.max_num_running_reqs) {
          this.log.emit('Scheduler', 'max_num_seqs', `waiting: running(${this.running.length}) >= max_num_seqs(${this.max_num_running_reqs}) -> stop admitting`, { running: this.running.length }, REF.Scheduler_max_num_running)
          break
        }
        const request_queue = this._select_waiting_queue_for_scheduling()
        if (!request_queue) break
        const request = request_queue.peek_request()
        const request_id = request.request_id

        if (this._is_blocked_waiting_status(request.status) && !this._try_promote_blocked_waiting_request(request)) {
          request_queue.pop_request()
          step_skipped_waiting.prepend_request(request)
          this.log.emit('Scheduler', 'skip_waiting', `waiting ${request_id}: still ${RequestStatusName[request.status]} -> skipped_waiting`, { request_id, status: RequestStatusName[request.status] }, REF.Scheduler_waiting_loop)
          continue
        }
        if (request.num_stale_output_tokens > 0 && !request.drop_stale_output) {
          request_queue.pop_request()
          step_skipped_waiting.prepend_request(request)
          this.log.emit('Scheduler', 'skip_waiting', `waiting ${request_id}: ${request.num_stale_output_tokens} stale in-flight output token(s) -> skipped_waiting until drained`, { request_id }, REF.Scheduler_waiting_loop)
          continue
        }

        let new_computed_blocks: KVCacheBlocks
        let num_new_local_computed_tokens: number
        let num_computed_tokens: number
        if (request.num_computed_tokens === 0) {
          ;[new_computed_blocks, num_new_local_computed_tokens, request.shared_prefix_boundary] = this.kv_cache_manager.get_computed_blocks(request)
          num_computed_tokens = num_new_local_computed_tokens
        } else {
          new_computed_blocks = this.kv_cache_manager.empty_kv_cache_blocks
          num_new_local_computed_tokens = 0
          num_computed_tokens = request.num_computed_tokens
        }

        let pad_spec_decode = false
        let num_new_tokens = request.num_tokens - num_computed_tokens

        // Pad new decode requests to a uniform spec-decoding size.
        if (this.num_spec_tokens > 0 && this.num_sampled_tokens_per_step > 0 && num_new_tokens === 1 && scheduled_running_reqs.length > 0 && !prefill_scheduled) {
          num_new_tokens = 1 + this.num_spec_tokens
          if (num_new_tokens > token_budget || num_computed_tokens + num_new_tokens > this.max_model_len) break
          pad_spec_decode = true
        }

        const threshold = cfg.long_prefill_token_threshold
        const before_threshold = num_new_tokens
        if (0 < threshold && threshold < num_new_tokens) num_new_tokens = threshold

        if (!cfg.enable_chunked_prefill && num_new_tokens > token_budget) {
          this.log.emit('Scheduler', 'chunked_prefill_disabled', `waiting ${request_id}: needs ${num_new_tokens} tokens > budget ${token_budget} and chunked prefill is disabled -> stop scheduling`, { request_id, num_new_tokens, token_budget }, REF.Scheduler_chunked_prefill_break)
          break
        }
        const before_budget = num_new_tokens
        num_new_tokens = Math.min(num_new_tokens, token_budget)
        if (num_new_tokens <= 0) throw new Error('num_new_tokens must be > 0')
        this.log.emit(
          'Scheduler',
          'waiting_num_new_tokens',
          `waiting ${request_id}: num_new_tokens = num_tokens(${request.num_tokens}) - cached(${num_computed_tokens}) = ${before_threshold}${before_threshold !== before_budget ? ` -> long_prefill_token_threshold ${before_budget}` : ''}${before_budget !== num_new_tokens ? ` -> chunked to budget ${num_new_tokens}` : ''}`,
          { request_id, num_new_tokens, num_computed_tokens, token_budget, chunked: before_budget !== num_new_tokens },
          REF.Scheduler_num_new_tokens_waiting,
        )

        const new_blocks = this.kv_cache_manager.allocate_slots(request, num_new_tokens, {
          num_new_computed_tokens: num_new_local_computed_tokens,
          new_computed_blocks,
          num_lookahead_tokens: this.num_lookahead_tokens,
          full_sequence_must_fit: cfg.scheduler_reserve_full_isl,
          has_scheduled_reqs: this.running.length > 0,
        })
        if (new_blocks === null) {
          this.log.emit('Scheduler', 'waiting_no_blocks', `waiting ${request_id}: allocate_slots returned None -> stop admitting (no preemption from the waiting path)`, { request_id }, REF.Scheduler_waiting_loop)
          break
        }

        request_queue.pop_request()
        this.running.push(request)
        if (request.status === RequestStatus.WAITING) scheduled_new_reqs.push(request)
        else if (request.status === RequestStatus.PREEMPTED) scheduled_resumed_reqs.push(request)
        else throw new Error(`Invalid request status: ${request.status}`)

        req_to_new_blocks.set(request_id, this.kv_cache_manager.get_blocks(request_id))
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens
        const was = request.status
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens
        if (pad_spec_decode) scheduled_spec_decode_tokens[request_id] = Array(this.num_spec_tokens).fill(-1)
        this.log.emit(
          'Scheduler',
          was === RequestStatus.PREEMPTED ? 'resumed' : 'admitted',
          `${was === RequestStatus.PREEMPTED ? 'resumed' : 'admitted'} ${request_id}: ${RequestStatusName[was]} -> RUNNING, ${num_new_tokens} token(s) this step, blocks [${this.kv_cache_manager.get_block_ids(request_id)[0].join(', ')}], budget left ${token_budget}`,
          { request_id, num_new_tokens, num_computed_tokens, block_ids: this.kv_cache_manager.get_block_ids(request_id)[0], token_budget },
          REF.Scheduler_admit,
        )
      }
      if (step_skipped_waiting.length > 0) this.skipped_waiting.prepend_requests(step_skipped_waiting)
    } else {
      this.log.emit('Scheduler', 'waiting_skipped_after_preempt', `waiting loop skipped: ${preempted_reqs.length} request(s) preempted this step`, { preempted: preempted_reqs.map((r) => r.request_id) }, REF.Scheduler_waiting_loop)
    }

    const total_num_scheduled_tokens = Object.values(num_scheduled_tokens).reduce((a, b) => a + b, 0)
    if (total_num_scheduled_tokens > this.max_num_scheduled_tokens) throw new Error('over budget')
    if (token_budget < 0) throw new Error('negative budget')
    if (this.running.length > this.max_num_running_reqs) throw new Error('too many running')

    let num_common_prefix_blocks = [0]
    if (this.running.length > 0) num_common_prefix_blocks = this.kv_cache_manager.get_num_common_prefix_blocks(this.running[0].request_id)

    const new_reqs_data: NewRequestData[] = scheduled_new_reqs.map((req) => ({
      req_id: req.request_id,
      prompt_token_ids: req.prompt_token_ids,
      sampling_params: req.sampling_params,
      block_ids: (req_to_new_blocks.get(req.request_id) as KVCacheBlocks).get_block_ids(),
      num_computed_tokens: req.num_computed_tokens,
    }))
    const cached_reqs_data = this._make_cached_request_data(scheduled_running_reqs, scheduled_resumed_reqs, req_to_new_blocks)

    this.prev_step_scheduled_req_ids = new Set(Object.keys(num_scheduled_tokens))

    const scheduler_output: SchedulerOutput = {
      scheduled_new_reqs: new_reqs_data,
      scheduled_cached_reqs: cached_reqs_data,
      num_scheduled_tokens,
      total_num_scheduled_tokens,
      scheduled_spec_decode_tokens,
      num_common_prefix_blocks,
      preempted_req_ids: this.reset_preempted_req_ids,
      finished_req_ids: this.finished_req_ids,
      has_structured_output_requests: false,
      pending_structured_output_tokens: false,
      num_spec_tokens_to_schedule: this.num_spec_tokens,
      step: this.current_step,
    }
    this.log.emit(
      'Scheduler',
      'scheduler_output',
      `SchedulerOutput: ${total_num_scheduled_tokens} token(s) across ${Object.keys(num_scheduled_tokens).length} request(s); new=[${new_reqs_data.map((r) => r.req_id).join(', ')}] cached=[${cached_reqs_data.req_ids.join(', ')}] finished=[${[...this.finished_req_ids].join(', ')}]`,
      {
        num_scheduled_tokens: { ...num_scheduled_tokens },
        total_num_scheduled_tokens,
        scheduled_new_reqs: new_reqs_data.map((r) => r.req_id),
        scheduled_cached_reqs: cached_reqs_data.req_ids,
        resumed_req_ids: [...cached_reqs_data.resumed_req_ids],
        finished_req_ids: [...this.finished_req_ids],
        scheduled_spec_decode_tokens: { ...scheduled_spec_decode_tokens },
      },
      REF.Scheduler_build_output,
    )
    this._update_after_schedule(scheduler_output)
    return scheduler_output
  }

  protected _preempt_request(request: Request): void {
    if (request.status !== RequestStatus.RUNNING) throw new Error('Only running requests can be preempted')
    const freed = this.kv_cache_manager.get_block_ids(request.request_id)[0]
    this.kv_cache_manager.free(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    if (request.spec_token_ids.length) request.spec_token_ids = []
    request.drop_stale_output = request.drop_stale_output && request.num_stale_output_tokens > 0
    request.num_stale_output_tokens = request.num_in_flight_tokens
    request.num_output_placeholders = 0
    request.num_preemptions += 1
    this.waiting.prepend_request(request)
    this.reset_preempted_req_ids.add(request.request_id)
    this.log.emit(
      'Scheduler',
      'preempt',
      `PREEMPT ${request.request_id} (${this.policy === 'fcfs' ? 'running.pop(): last in running' : 'max(priority, arrival_time)'}): free blocks [${freed.join(', ')}], num_computed_tokens=0, status PREEMPTED, waiting.prepend (recompute preemption)`,
      { request_id: request.request_id, freed_block_ids: freed, num_preemptions: request.num_preemptions, num_stale_output_tokens: request.num_stale_output_tokens },
      REF.Scheduler_preempt_request,
    )
  }

  protected _update_after_schedule(scheduler_output: SchedulerOutput): void {
    // Advance num_computed_tokens at schedule time so the next schedule() can
    // continue a chunked prefill immediately; rejections roll it back later.
    for (const [req_id, n] of Object.entries(scheduler_output.num_scheduled_tokens)) {
      const request = this.requests.get(req_id) as Request
      request.num_computed_tokens += n
      request.num_in_flight_tokens += n
      request.is_prefill_chunk = request.num_computed_tokens < request.num_tokens + request.num_output_placeholders
      scheduler_output.has_structured_output_requests ||= request.use_structured_output && !request.is_prefill_chunk
      this.log.emit(
        'Scheduler',
        'update_after_schedule',
        `${req_id}: num_computed_tokens += ${n} -> ${request.num_computed_tokens} (optimistic, at schedule time)${request.is_prefill_chunk ? ', still a prefill chunk' : ''}`,
        { request_id: req_id, num_computed_tokens: request.num_computed_tokens, is_prefill_chunk: request.is_prefill_chunk },
        REF.Scheduler_update_after_schedule,
      )
    }
    this.finished_req_ids = new Set()
    this.reset_preempted_req_ids = new Set()
  }

  private _make_cached_request_data(running_reqs: Request[], resumed_reqs: Request[], req_to_new_blocks: Map<string, KVCacheBlocks>): CachedRequestData {
    const req_ids: string[] = []
    const new_block_ids: (number[][] | null)[] = []
    const num_computed_tokens: number[] = []
    const num_output_tokens: number[] = []
    const resumed_req_ids = new Set<string>()
    const n_running = running_reqs.length
    ;[...running_reqs, ...resumed_reqs].forEach((req, idx) => {
      req_ids.push(req.request_id)
      if (idx >= n_running) resumed_req_ids.add(req.request_id)
      const blocks = req_to_new_blocks.get(req.request_id) as KVCacheBlocks
      new_block_ids.push(blocks.is_empty ? null : blocks.get_block_ids())
      num_computed_tokens.push(req.num_computed_tokens)
      num_output_tokens.push(req.num_output_tokens + req.num_output_placeholders)
    })
    return { req_ids, resumed_req_ids, new_block_ids, num_computed_tokens, num_output_tokens }
  }

  private _is_blocked_waiting_status(status: RequestStatus): boolean {
    return (
      status === RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR ||
      status === RequestStatus.WAITING_FOR_REMOTE_KVS ||
      status === RequestStatus.WAITING_FOR_STREAMING_REQ
    )
  }

  private _select_waiting_queue_for_scheduling(): RequestQueue | null {
    if (this.policy === 'fcfs') {
      return this.skipped_waiting.length > 0 ? this.skipped_waiting : this.waiting.length > 0 ? this.waiting : null
    }
    if (this.waiting.length > 0 && this.skipped_waiting.length > 0) {
      const w = this.waiting.peek_request()
      const s = this.skipped_waiting.peek_request()
      return Request_lt(w, s) ? this.waiting : this.skipped_waiting
    }
    return this.waiting.length > 0 ? this.waiting : this.skipped_waiting.length > 0 ? this.skipped_waiting : null
  }

  private _try_promote_blocked_waiting_request(request: Request): boolean {
    if (request.status === RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR) {
      if (this.structured_output_manager?.is_grammar_ready(request)) {
        request.status = RequestStatus.WAITING
        this.log.emit('Scheduler', 'grammar_ready', `${request.request_id}: grammar compiled -> WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR -> WAITING`, { request_id: request.request_id }, REF.RequestStatus_WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR)
        return true
      }
      return false
    }
    return false
  }

  // ----------------------------------------------------- grammar bitmask

  get_grammar_bitmask(scheduler_output: SchedulerOutput): GrammarOutput | null {
    if (!scheduler_output.has_structured_output_requests || !this.structured_output_manager) return null
    const ids = Object.keys(scheduler_output.num_scheduled_tokens).filter((id) => {
      const req = this.requests.get(id)
      return req && req.use_structured_output && !req.is_prefill_chunk
    })
    if (ids.length === 0) return null
    const bitmask = this.structured_output_manager.grammar_bitmask(this.requests, ids, scheduler_output.scheduled_spec_decode_tokens)
    this.log.emit(
      'Scheduler',
      'grammar_bitmask',
      `get_grammar_bitmask: ${ids.length} structured-output request(s) [${ids.join(', ')}] -> bitmask rows computed on CPU while the GPU forward runs`,
      { request_ids: ids, rows: bitmask.length },
      REF.Scheduler_get_grammar_bitmask,
    )
    return { structured_output_request_ids: ids, grammar_bitmask: bitmask }
  }

  // ---------------------------------------------------- update_from_output

  update_from_output(scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput): EngineCoreOutputs {
    const sampled_token_ids = model_runner_output.sampled_token_ids
    const outputs: EngineCoreOutput[] = []
    const stopped_running_reqs = new Set<Request>()
    const stopped_preempted_reqs = new Set<Request>()

    for (const [req_id, num_tokens_scheduled] of Object.entries(scheduler_output.num_scheduled_tokens)) {
      const request = this.requests.get(req_id)
      let output_is_stale = false
      if (request) {
        request.num_in_flight_tokens -= num_tokens_scheduled
        if (request.num_stale_output_tokens > 0) {
          output_is_stale = true
          request.num_stale_output_tokens -= num_tokens_scheduled
          if (request.num_stale_output_tokens < 0) throw new Error('negative stale tokens')
        }
      }
      if (!request || request.is_finished()) continue
      if (output_is_stale && request.drop_stale_output) continue

      const req_index = model_runner_output.req_id_to_index[req_id]
      const generated_token_ids = req_index === undefined ? [] : (sampled_token_ids[req_index] ?? [])

      const scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens[req_id]
      if (scheduled_spec_token_ids && scheduled_spec_token_ids.length > 0 && generated_token_ids.length > 0) {
        const num_draft_tokens = scheduled_spec_token_ids.length
        const num_accepted = Math.max(generated_token_ids.length - this.num_sampled_tokens_per_step, 0)
        const num_rejected = num_draft_tokens - num_accepted
        if (!output_is_stale) {
          if (request.num_computed_tokens > 0) request.num_computed_tokens -= num_rejected
          if (request.num_output_placeholders > 0) request.num_output_placeholders -= num_rejected
        }
        this.log.emit(
          'Scheduler',
          'spec_fixup',
          `${req_id}: ${num_accepted}/${num_draft_tokens} draft token(s) accepted, ${num_rejected} rejected -> num_computed_tokens -= ${num_rejected} (now ${request.num_computed_tokens})`,
          { request_id: req_id, num_draft_tokens, num_accepted, num_rejected, num_computed_tokens: request.num_computed_tokens },
          REF.Scheduler_spec_rejection_fixup,
        )
      }

      let stopped = false
      let new_token_ids = generated_token_ids.slice()
      const status_before_stop = request.status
      if (new_token_ids.length > 0) {
        ;[new_token_ids, stopped] = this._update_request_with_output(request, new_token_ids, output_is_stale)
      }

      if (new_token_ids.length > 0 && this.structured_output_manager?.should_advance(request)) {
        if (!this.structured_output_manager.accept_tokens(request, new_token_ids)) {
          request.status = RequestStatus.FINISHED_ERROR
          stopped = true
        }
      }

      let finish_reason = null
      if (stopped) {
        finish_reason = request.get_finished_reason()
        this._free_request(request)
        if (status_before_stop === RequestStatus.RUNNING) stopped_running_reqs.add(request)
        else stopped_preempted_reqs.add(request)
      }

      if (new_token_ids.length > 0 || stopped) {
        outputs.push({ request_id: req_id, new_token_ids, finish_reason, stop_reason: request.stop_reason })
        this.log.emit(
          'Scheduler',
          'engine_core_output',
          `EngineCoreOutput(${req_id}): new_token_ids=[${new_token_ids.join(', ')}]${finish_reason ? `, finish_reason=${finish_reason}` : ''}`,
          { request_id: req_id, new_token_ids, finish_reason },
          REF.Scheduler_update_from_output,
        )
      } else {
        this.log.emit('Scheduler', 'no_output', `${req_id}: no output this step (prefill chunk, num_computed_tokens=${request.num_computed_tokens}/${request.num_tokens})`, { request_id: req_id }, REF.Scheduler_update_from_output)
      }
    }

    if (stopped_running_reqs.size) this.running = this.running.filter((r) => !stopped_running_reqs.has(r))
    if (stopped_preempted_reqs.size) {
      this.waiting.remove_requests(stopped_preempted_reqs)
      this.skipped_waiting.remove_requests(stopped_preempted_reqs)
    }
    return { outputs, finished_requests: new Set(this.finished_req_ids) }
  }

  protected _update_request_with_output(request: Request, new_token_ids: number[], _is_stale = false): [number[], boolean] {
    let stopped = false
    for (let i = 0; i < new_token_ids.length; i++) {
      request.append_output_token_ids(new_token_ids[i])
      stopped = check_stop(request, this.max_model_len)
      if (stopped) {
        new_token_ids = new_token_ids.slice(0, i + 1)
        break
      }
    }
    return [new_token_ids, stopped]
  }

  private _free_request(request: Request): void {
    if (!request.is_finished()) throw new Error('free of unfinished request')
    this.finished_req_ids.add(request.request_id)
    this.log.emit(
      'Scheduler',
      'free_request',
      `${request.request_id} finished (${RequestStatusName[request.status]}): finished_req_ids += it; kv_cache_manager.free -> blocks back to free_block_queue`,
      { request_id: request.request_id, status: RequestStatusName[request.status] },
      REF.Scheduler_free_request,
    )
    this.kv_cache_manager.free(request)
    this.requests.delete(request.request_id)
  }

  finish_requests(request_ids: Iterable<string>, status: RequestStatus): void {
    const to_remove: Request[] = []
    for (const id of request_ids) {
      const req = this.requests.get(id)
      if (!req || req.is_finished()) continue
      req.status = status
      to_remove.push(req)
      this._free_request(req)
    }
    const set = new Set(to_remove)
    this.running = this.running.filter((r) => !set.has(r))
    this.waiting.remove_requests(set)
    this.skipped_waiting.remove_requests(set)
  }

  // ---------------------------------------------------------- spec decode

  update_draft_token_ids(draft_token_ids: DraftTokenIds): void {
    draft_token_ids.req_ids.forEach((req_id, i) => {
      const request = this.requests.get(req_id)
      if (!request || request.is_finished()) return
      let spec = draft_token_ids.draft_token_ids[i]
      if (request.is_prefill_chunk) {
        if (request.spec_token_ids.length) request.spec_token_ids = []
        return
      }
      if (this.structured_output_manager?.should_advance(request)) {
        spec = this.structured_output_manager.validate_tokens(request, spec)
      }
      request.spec_token_ids = spec
      this.log.emit(
        'Scheduler',
        'update_draft_token_ids',
        `${req_id}: request.spec_token_ids = [${spec.join(', ')}] (proposed by the drafter, verified next step)`,
        { request_id: req_id, spec_token_ids: spec },
        REF.Scheduler_update_draft_token_ids,
      )
    })
  }
}

function Request_lt(a: Request, b: Request): boolean {
  return (a.constructor as typeof Request).compare(a, b) < 0
}

/** Port of vllm/v1/core/sched/utils.py::check_stop (no min_tokens, no repetition detection). */
export function check_stop(request: Request, max_model_len: number): boolean {
  const sp = request.sampling_params
  const last = request.output_token_ids[request.output_token_ids.length - 1]
  if (sp.eos_token_id !== undefined && last === sp.eos_token_id && !sp.ignore_eos) {
    request.status = RequestStatus.FINISHED_STOPPED
    return true
  }
  if (sp.stop_token_ids?.includes(last)) {
    request.status = RequestStatus.FINISHED_STOPPED
    request.stop_reason = last
    return true
  }
  if (request.num_tokens >= max_model_len || request.num_output_tokens >= request.max_tokens) {
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return true
  }
  return false
}
