// Drives a Scenario through the engine and records a Snapshot after every
// sub-phase. Snapshots are plain JSON so the UI can scrub freely.

import { EngineCore, InputProcessor, OutputProcessor, type RequestOutput } from './engine_core'
import { EventLog, type Phase, type SimEvent } from './events'
import { RequestStatusName } from './request'
import type { Scenario } from './scenario'
import { REF } from './source_refs'
import { EOS_ID, ToyTokenizer } from './tokenizer'
import type { PreparedInputs, VerificationRecord } from './worker/model_runner'
import type { NgramMatch } from './spec_decode/ngram_proposer'

export interface RequestSnapshot {
  request_id: string
  external_req_id: string
  status: string
  priority: number
  arrival_time: number
  num_prompt_tokens: number
  num_tokens: number
  num_computed_tokens: number
  num_output_tokens: number
  num_output_placeholders: number
  num_in_flight_tokens: number
  spec_token_ids: number[]
  is_prefill_chunk: boolean
  num_preemptions: number
  block_ids: number[]
  block_hashes: string[]
  all_token_ids: number[]
  max_tokens: number
  use_structured_output: boolean
}

export interface BlockSnapshot {
  block_id: number
  ref_cnt: number
  block_hash: string | null
  block_hash_num_tokens: number | null
  is_null: boolean
  /** Token id stored in each slot (GPU memory), null if never written. */
  slots: (number | null)[]
  /** External id of the request that wrote each slot. */
  writers: (string | null)[]
  /** Which request currently maps to this block (from req_to_blocks), if any. */
  owners: string[]
}

export interface Snapshot {
  index: number
  step: number
  /** The phase whose completion this snapshot captures. */
  phase: Phase
  label: string
  events_from: number
  events_to: number
  scheduler: {
    waiting: string[]
    skipped_waiting: string[]
    running: string[]
    requests: Record<string, RequestSnapshot>
    token_budget: number
    max_num_seqs: number
    last_output: {
      step: number
      num_scheduled_tokens: Record<string, number>
      total_num_scheduled_tokens: number
      scheduled_new_reqs: string[]
      scheduled_cached_reqs: string[]
      resumed_req_ids: string[]
      scheduled_spec_decode_tokens: Record<string, number[]>
      finished_req_ids: string[]
      preempted_req_ids: string[]
    } | null
  }
  kv: {
    block_size: number
    num_gpu_blocks: number
    blocks: BlockSnapshot[]
    free_queue_order: number[]
    num_free_blocks: number
    cached_hash_entries: { hash: string; block_ids: number[] }[]
    /** request_id -> block ids (req_to_blocks). */
    req_to_blocks: Record<string, number[]>
  }
  worker: {
    batch_rows: (string | null)[]
    prepared: PreparedInputs | null
    has_stashed_state: boolean
    verification: VerificationRecord[]
    ngram: Record<string, NgramMatch>
    pending_drafts: Record<string, number[]>
  }
  batch_queue: number[] | null
  outputs: Record<string, RequestOutput>
  /** internal request id -> external id, for display. */
  id_map: Record<string, string>
  vocab: string[]
  vocab_size: number
}

export interface SimulationResult {
  scenario: Scenario
  snapshots: Snapshot[]
  events: SimEvent[]
  tokenizer: ToyTokenizer
}

const PHASE_LABEL: Record<Phase, string> = {
  add_request: 'add_request',
  schedule: 'Scheduler.schedule()',
  execute_model: 'Worker.execute_model()',
  grammar_bitmask: 'get_grammar_bitmask()',
  sample_tokens: 'Worker.sample_tokens()',
  update_from_output: 'Scheduler.update_from_output()',
  post_step: 'post_step()',
  output: 'OutputProcessor',
}

export function run_simulation(scenario: Scenario): SimulationResult {
  const log = new EventLog()
  const tokenizer = new ToyTokenizer()
  // Reserve vocabulary for prompts and continuations up front so ids are stable.
  const continuations = new Map<string, number[]>()
  for (const r of scenario.requests) {
    tokenizer.encode(r.prompt)
    for (const c of r.guided_choice ?? []) tokenizer.encode(c, false)
    continuations.set(r.id, r.continuation ? [...tokenizer.encode(r.continuation, false), EOS_ID] : [EOS_ID])
  }
  const WORDS = ['the', 'model', 'runs', 'fast', 'on', 'GPUs', 'and', 'blocks', 'are', 'paged'].map((w) => tokenizer.token_to_id(w))

  const external_of = new Map<string, string>()
  const oracle = (request_id: string, context: readonly number[]): number => {
    const ext = external_of.get(request_id) ?? request_id
    const req = scenario.requests.find((r) => r.id === ext)
    const prompt_len = req ? tokenizer.encode(req.prompt).length : 0
    const cont = continuations.get(ext) ?? [EOS_ID]
    const k = context.length - prompt_len
    if (k >= 0 && k < cont.length) return cont[k]
    if (k >= cont.length) return EOS_ID
    // Position inside the prompt (spec-decode verification of a wrong draft): deterministic filler.
    return WORDS[(context.length * 7 + request_id.length) % WORDS.length]
  }

  const engine = new EngineCore(scenario.config, tokenizer, oracle, log)
  const input_processor = new InputProcessor(tokenizer, log)
  const output_processor = new OutputProcessor(tokenizer, log)
  const snapshots: Snapshot[] = []
  let events_cursor = 0
  let step = 0

  const snapshot = (phase: Phase, label = PHASE_LABEL[phase], force = false): void => {
    // Skip empty snapshots (phase transitions that emitted nothing).
    if (!force && log.events.length === events_cursor) return
    const s = engine.scheduler
    const km = s.kv_cache_manager
    const pool = km.block_pool
    const mgr = km.coordinator.single_type_managers[0]
    const req_to_blocks: Record<string, number[]> = {}
    for (const [rid, blocks] of mgr.req_to_blocks) req_to_blocks[rid] = blocks.map((b) => b.block_id)
    const owners = new Map<number, string[]>()
    for (const [rid, ids] of Object.entries(req_to_blocks)) for (const id of ids) owners.set(id, [...(owners.get(id) ?? []), external_of.get(rid) ?? rid])
    const requests: Record<string, RequestSnapshot> = {}
    for (const [rid, r] of s.requests) {
      requests[rid] = {
        request_id: rid,
        external_req_id: r.external_req_id,
        status: RequestStatusName[r.status],
        priority: r.priority,
        arrival_time: r.arrival_time,
        num_prompt_tokens: r.num_prompt_tokens,
        num_tokens: r.num_tokens,
        num_computed_tokens: r.num_computed_tokens,
        num_output_tokens: r.num_output_tokens,
        num_output_placeholders: r.num_output_placeholders,
        num_in_flight_tokens: r.num_in_flight_tokens,
        spec_token_ids: r.spec_token_ids.slice(),
        is_prefill_chunk: r.is_prefill_chunk,
        num_preemptions: r.num_preemptions,
        block_ids: req_to_blocks[rid] ?? [],
        block_hashes: r.block_hashes.slice(),
        all_token_ids: r.all_token_ids.slice(),
        max_tokens: r.max_tokens,
        use_structured_output: r.use_structured_output,
      }
    }
    const last = last_scheduler_output
    const snap: Snapshot = {
      index: snapshots.length,
      step,
      phase,
      label,
      events_from: events_cursor,
      events_to: log.events.length,
      scheduler: {
        waiting: s.waiting.toArray().map((r) => r.request_id),
        skipped_waiting: s.skipped_waiting.toArray().map((r) => r.request_id),
        running: s.running.map((r) => r.request_id),
        requests,
        token_budget: s.max_num_scheduled_tokens,
        max_num_seqs: s.max_num_running_reqs,
        last_output: last
          ? {
              step: last.step,
              num_scheduled_tokens: { ...last.num_scheduled_tokens },
              total_num_scheduled_tokens: last.total_num_scheduled_tokens,
              scheduled_new_reqs: last.scheduled_new_reqs.map((r) => r.req_id),
              scheduled_cached_reqs: last.scheduled_cached_reqs.req_ids.slice(),
              resumed_req_ids: [...last.scheduled_cached_reqs.resumed_req_ids],
              scheduled_spec_decode_tokens: structuredClone(last.scheduled_spec_decode_tokens),
              finished_req_ids: [...last.finished_req_ids],
              preempted_req_ids: [...last.preempted_req_ids],
            }
          : null,
      },
      kv: {
        block_size: scenario.config.block_size,
        num_gpu_blocks: scenario.config.num_gpu_blocks,
        blocks: pool.blocks.map((b) => ({
          block_id: b.block_id,
          ref_cnt: b.ref_cnt,
          block_hash: b.block_hash,
          block_hash_num_tokens: b.block_hash_num_tokens,
          is_null: b.is_null,
          slots: engine.model_runner.block_contents(b.block_id),
          writers: engine.model_runner.block_writers(b.block_id).map((w) => (w === null ? null : external_of.get(w) ?? w)),
          owners: owners.get(b.block_id) ?? [],
        })),
        free_queue_order: pool.free_block_queue.get_all_free_blocks().map((b) => b.block_id),
        num_free_blocks: pool.get_num_free_blocks(),
        cached_hash_entries: pool.cached_block_hash_to_block.entries().map(([hash, block_ids]) => ({ hash, block_ids })),
        req_to_blocks,
      },
      worker: {
        batch_rows: engine.model_runner.batch.slice(),
        prepared: last_prepared ? structuredClone(last_prepared) : null,
        has_stashed_state: engine.model_runner.execute_model_state !== null,
        verification: structuredClone(engine.model_runner.last_verification),
        ngram: structuredClone(engine.model_runner.last_ngram),
        pending_drafts: Object.fromEntries(engine.model_runner.pending_drafts),
      },
      batch_queue: engine.batch_queue ? engine.batch_queue.map((b) => b.scheduler_output.step) : null,
      outputs: structuredClone(Object.fromEntries(output_processor.request_states)),
      id_map: Object.fromEntries(external_of),
      vocab: Array.from({ length: tokenizer.vocab_size }, (_, i) => tokenizer.id_to_token(i)),
      vocab_size: tokenizer.vocab_size,
    }
    snapshots.push(snap)
    events_cursor = log.events.length
  }

  let last_scheduler_output: import('./output').SchedulerOutput | null = null
  let last_prepared: PreparedInputs | null = null
  // Capture the state at each phase transition (state after the phase that just ended).
  const orig_setPhase = log.setPhase.bind(log)
  log.setPhase = (phase: Phase) => {
    if (phase !== log.phase) {
      if (engine.model_runner.execute_model_state) last_prepared = engine.model_runner.execute_model_state.prepared
      snapshot(log.phase)
    }
    orig_setPhase(phase)
  }
  // Scheduler.schedule() result is captured via a light wrapper.
  const orig_schedule = engine.scheduler.schedule.bind(engine.scheduler)
  engine.scheduler.schedule = () => {
    const out = orig_schedule()
    last_scheduler_output = out
    return out
  }

  const arrivals = new Map<number, typeof scenario.requests>()
  for (const r of scenario.requests) arrivals.set(r.arrival_step, [...(arrivals.get(r.arrival_step) ?? []), r])
  const last_arrival = Math.max(-1, ...scenario.requests.map((r) => r.arrival_step))

  log.step = 0
  snapshot('add_request', 'engine ready', true)
  for (step = 0; step < scenario.max_steps; step++) {
    log.step = step
    log.setPhase('add_request')
    for (const r of arrivals.get(step) ?? []) {
      log.emit('LLMEngine', 'add_request', `LLM.generate -> LLMEngine.add_request("${r.id}")`, { request_id: r.id }, REF.LLMEngine_add_request)
      const ecr = input_processor.process_inputs(r.id, r.prompt, { max_tokens: r.max_tokens, guided_choice: r.guided_choice }, step, r.priority ?? 0)
      input_processor.assign_request_id(ecr)
      external_of.set(ecr.request_id, r.id)
      output_processor.add_request(ecr)
      log.emit(
        'EngineCoreClient',
        'send',
        `EngineCoreClient.add_request: msgspec-encode EngineCoreRequest, send_multipart(ADD) on the ZMQ ROUTER socket (InprocClient calls EngineCore directly)`,
        { request_id: ecr.request_id, bytes_hint: JSON.stringify(ecr).length },
        REF.SyncMPClient_send_input,
      )
      log.emit('EngineCoreProc', 'input_thread', `EngineCoreProc input thread: decode frames -> preprocess_add_request -> input_queue.put_nowait((ADD, request))`, { request_id: ecr.request_id }, REF.EngineCoreProc_process_input_sockets)
      const request = engine.preprocess_add_request(ecr)
      log.emit('EngineCoreProc', 'busy_loop', `run_busy_loop: _process_input_queue -> _handle_client_request(ADD)`, { request_id: ecr.request_id }, REF.EngineCoreProc_run_busy_loop)
      engine.add_request(request)
    }
    if (!engine.has_requests() && step > last_arrival) {
      snapshot('add_request', 'all requests finished', true)
      break
    }
    if ((arrivals.get(step) ?? []).length > 0) snapshot('add_request')
    const outs = engine.step_fn()
    log.setPhase('output')
    if (outs && outs.outputs.length > 0) {
      log.emit('EngineCoreProc', 'output_thread', `output_queue.put_nowait(EngineCoreOutputs) -> output thread -> msgpack -> ZMQ PUSH -> client PULL -> OutputProcessor`, { num_outputs: outs.outputs.length }, REF.EngineCoreProc_process_output_sockets)
      output_processor.process_outputs(outs)
    } else {
      log.emit('EngineCore', 'no_output', `step ${step}: no EngineCoreOutputs this step${engine.batch_queue ? ' (batch queue filling)' : ''}`, {}, REF.EngineCore_step_with_batch_queue)
    }
    snapshot('output', outs && outs.outputs.length ? 'OutputProcessor.process_outputs()' : 'no output this step')
  }
  return { scenario, snapshots, events: log.events, tokenizer }
}
