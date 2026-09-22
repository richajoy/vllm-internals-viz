// Mock of vllm/v1/worker/gpu_model_runner.py (V1 runner) at the level of
// bookkeeping: persistent batch (InputBatch), block tables, the flattened
// "super sequence" (input_ids / positions / slot_mapping / query_start_loc /
// seq_lens / logits_indices), the execute_model -> sample_tokens split with
// ExecuteModelState stashed between, the rejection sampler and the drafter.
// The "model" is an oracle that returns the next token for a context.

import { type EventLog, NULL_LOG } from '../events'
import type { DraftTokenIds, GrammarOutput, ModelRunnerOutput, SchedulerOutput } from '../output'
import type { SamplingParams } from '../request'
import { REF } from '../source_refs'
import { unpack_bitmask } from '../structured_output'
import { NgramProposer, type NgramMatch } from '../spec_decode/ngram_proposer'

export const PAD_SLOT_ID = -1

/** Next-token model. Deterministic given (request, context). */
export type TokenOracle = (request_id: string, context: readonly number[]) => number

export interface CachedRequestState {
  req_id: string
  prompt_token_ids: number[]
  sampling_params: SamplingParams
  output_token_ids: number[]
  num_computed_tokens: number
  block_ids: number[]
  /** The row this request occupies in the persistent batch (InputBatch). */
  batch_index: number
}

export interface PreparedInputs {
  req_ids: string[]
  num_scheduled_tokens: number[]
  input_ids: number[]
  positions: number[]
  slot_mapping: number[]
  query_start_loc: number[]
  seq_lens: number[]
  num_actual_tokens: number
  logits_indices: number[]
  /** Per request: [start, end) in the flattened sequence. */
  spans: [number, number][]
  /** Spec decode: for each request, the draft token ids scheduled this step. */
  draft_token_ids: number[][]
}

export interface ExecuteModelState {
  scheduler_output: SchedulerOutput
  prepared: PreparedInputs
  /** "logits": the oracle's argmax per logits_index position. */
  logits_argmax: number[]
}

export interface SpecDecodeConfig {
  method: 'ngram'
  num_speculative_tokens: number
  prompt_lookup_min: number
  prompt_lookup_max: number
  /** null = greedy verification (accept iff draft == target argmax). */
  acceptance_rate: number | null
}

export interface VerificationRecord {
  req_id: string
  draft_token_ids: number[]
  target_argmax: number[]
  accepted: boolean[]
  sampled: number[]
  /** 'greedy' compare or seeded coin flips. */
  mode: 'greedy' | 'stochastic'
}

export interface ModelRunnerConfig {
  block_size: number
  num_gpu_blocks: number
  max_model_len: number
  vocab_size: () => number
  spec: SpecDecodeConfig | null
  seed?: number
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export class GPUModelRunner {
  cfg: ModelRunnerConfig
  oracle: TokenOracle
  log: EventLog
  requests = new Map<string, CachedRequestState>()
  /** Persistent batch rows; null = hole not yet condensed. Mirrors InputBatch. */
  batch: (string | null)[] = []
  /** Paged KV memory: slot -> token id whose K/V lives there (null = never written). */
  kv_cache: (number | null)[]
  /** Which request wrote each slot (stale data keeps its old writer). */
  kv_writer: (string | null)[]
  execute_model_state: ExecuteModelState | null = null
  drafter: NgramProposer | null
  private _draft_token_ids: DraftTokenIds | null = null
  /** Worker-side draft ids per request (used under async scheduling). */
  pending_drafts = new Map<string, number[]>()
  last_verification: VerificationRecord[] = []
  last_ngram: Record<string, NgramMatch> = {}
  private rand: () => number

  constructor(cfg: ModelRunnerConfig, oracle: TokenOracle, log: EventLog = NULL_LOG) {
    this.cfg = cfg
    this.oracle = oracle
    this.log = log
    this.kv_cache = new Array<number | null>(cfg.num_gpu_blocks * cfg.block_size).fill(null)
    this.kv_writer = new Array<string | null>(cfg.num_gpu_blocks * cfg.block_size).fill(null)
    this.drafter = cfg.spec
      ? new NgramProposer({
          prompt_lookup_min: cfg.spec.prompt_lookup_min,
          prompt_lookup_max: cfg.spec.prompt_lookup_max,
          num_speculative_tokens: cfg.spec.num_speculative_tokens,
          max_model_len: cfg.max_model_len,
        })
      : null
    this.rand = mulberry32(cfg.seed ?? 1)
  }

  // ------------------------------------------------------- _update_states

  _update_states(so: SchedulerOutput): void {
    // 1. Remove finished requests from the persistent batch.
    for (const req_id of so.finished_req_ids) {
      const st = this.requests.get(req_id)
      if (!st) continue
      this.requests.delete(req_id)
      this.batch[st.batch_index] = null
      this.log.emit('GPUModelRunner', 'remove_request', `_update_states: remove finished ${req_id} from input_batch (row ${st.batch_index})`, { request_id: req_id, batch_index: st.batch_index }, REF.GPUModelRunner_update_states)
    }
    // 2. Requests that were preempted (not scheduled and not finished) leave the batch too.
    for (const [req_id, st] of [...this.requests]) {
      if (so.num_scheduled_tokens[req_id] === undefined) {
        this.requests.delete(req_id)
        this.batch[st.batch_index] = null
        this.log.emit('GPUModelRunner', 'remove_request', `_update_states: ${req_id} not scheduled (preempted) -> removed from input_batch`, { request_id: req_id }, REF.GPUModelRunner_update_states)
      }
    }
    // 3. Cached requests: update num_computed_tokens and append new block ids.
    const c = so.scheduled_cached_reqs
    c.req_ids.forEach((req_id, i) => {
      const st = this.requests.get(req_id)
      const new_blocks = c.new_block_ids[i]?.[0] ?? []
      if (!st) {
        // Resumed after preemption: rebuild from scheduler data.
        return
      }
      st.num_computed_tokens = c.num_computed_tokens[i]
      if (new_blocks.length) st.block_ids.push(...new_blocks)
      if (new_blocks.length) {
        this.log.emit('GPUModelRunner', 'block_table_append', `block_table[${req_id}] += [${new_blocks.join(', ')}] -> [${st.block_ids.join(', ')}]`, { request_id: req_id, block_ids: st.block_ids.slice() }, REF.GPUModelRunner_update_states)
      }
    })
    // 4. New requests: add CachedRequestState + block table row.
    for (const nr of so.scheduled_new_reqs) {
      const st: CachedRequestState = {
        req_id: nr.req_id,
        prompt_token_ids: nr.prompt_token_ids.slice(),
        sampling_params: nr.sampling_params,
        output_token_ids: [],
        num_computed_tokens: nr.num_computed_tokens,
        block_ids: nr.block_ids[0].slice(),
        batch_index: -1,
      }
      this.requests.set(nr.req_id, st)
      this._add_to_batch(st)
      this.log.emit('GPUModelRunner', 'add_request', `_update_states: add ${nr.req_id} to input_batch row ${st.batch_index}; block_table row = [${st.block_ids.join(', ')}]`, { request_id: nr.req_id, batch_index: st.batch_index, block_ids: st.block_ids.slice() }, REF.GPUModelRunner_update_states)
    }
    // Resumed requests (in cached data but unknown to us) need their token ids.
    c.req_ids.forEach((req_id, i) => {
      if (this.requests.has(req_id)) return
      if (!c.resumed_req_ids.has(req_id)) throw new Error(`unknown cached request ${req_id}`)
      const st: CachedRequestState = {
        req_id,
        prompt_token_ids: this.resume_token_ids?.(req_id).prompt ?? [],
        sampling_params: this.resume_token_ids?.(req_id).sampling_params ?? { max_tokens: 0 },
        output_token_ids: this.resume_token_ids?.(req_id).output ?? [],
        num_computed_tokens: c.num_computed_tokens[i],
        block_ids: (c.new_block_ids[i]?.[0] ?? []).slice(),
        batch_index: -1,
      }
      this.requests.set(req_id, st)
      this._add_to_batch(st)
      this.log.emit('GPUModelRunner', 'add_request', `_update_states: resumed ${req_id} re-added to input_batch row ${st.batch_index}; block_table row = [${st.block_ids.join(', ')}]`, { request_id: req_id, batch_index: st.batch_index, block_ids: st.block_ids.slice() }, REF.GPUModelRunner_update_states)
    })
    this._condense()
  }

  /** Scheduler-provided lookup for resumed requests' token ids (set by the engine). */
  resume_token_ids: ((req_id: string) => { prompt: number[]; output: number[]; sampling_params: SamplingParams }) | null = null

  private _add_to_batch(st: CachedRequestState): void {
    const hole = this.batch.indexOf(null)
    if (hole !== -1) {
      this.batch[hole] = st.req_id
      st.batch_index = hole
    } else {
      this.batch.push(st.req_id)
      st.batch_index = this.batch.length - 1
    }
  }

  /** InputBatch.condense: fill holes by moving the last row down. */
  private _condense(): void {
    let last = this.batch.length - 1
    while (last >= 0 && this.batch[last] === null) last -= 1
    this.batch.length = last + 1
    for (let i = 0; i < this.batch.length; i++) {
      if (this.batch[i] !== null) continue
      while (last > i && this.batch[last] === null) last -= 1
      if (last <= i) break
      const moved = this.batch[last] as string
      this.batch[i] = moved
      this.batch[last] = null
      const st = this.requests.get(moved)
      if (st) st.batch_index = i
      this.log.emit('GPUModelRunner', 'condense', `input_batch.condense: move ${moved} from row ${last} to row ${i}`, { request_id: moved, from: last, to: i }, REF.GPUModelRunner_update_states)
      this.batch.length = last
      last -= 1
    }
  }

  // ------------------------------------------------------ _prepare_inputs

  compute_slot(block_ids: readonly number[], pos: number): number {
    const bs = this.cfg.block_size
    const block_number = block_ids[Math.floor(pos / bs)]
    if (block_number === undefined) return PAD_SLOT_ID
    return block_number * bs + (pos % bs)
  }

  _prepare_inputs(so: SchedulerOutput): PreparedInputs {
    // Batch order follows the persistent batch (InputBatch), not the scheduler dict.
    const req_ids = this.batch.filter((r): r is string => r !== null && so.num_scheduled_tokens[r] !== undefined)
    const input_ids: number[] = []
    const positions: number[] = []
    const slot_mapping: number[] = []
    const query_start_loc = [0]
    const seq_lens: number[] = []
    const logits_indices: number[] = []
    const spans: [number, number][] = []
    const num_scheduled_tokens: number[] = []
    const draft_token_ids: number[][] = []
    for (const req_id of req_ids) {
      const st = this.requests.get(req_id) as CachedRequestState
      const n = so.num_scheduled_tokens[req_id]
      let drafts = so.scheduled_spec_decode_tokens[req_id] ?? []
      if (drafts.length > 0 && drafts[0] === -1) {
        // Async scheduling: the scheduler only knows the count (-1
        // placeholders); the real draft ids live worker-side.
        const real = this.pending_drafts.get(req_id) ?? []
        drafts = Array.from({ length: drafts.length }, (_, j) => real[j] ?? -1)
      }
      const all = [...st.prompt_token_ids, ...st.output_token_ids, ...drafts]
      const start = input_ids.length
      for (let q = 0; q < n; q++) {
        const pos = st.num_computed_tokens + q
        input_ids.push(all[pos] ?? PAD_SLOT_ID)
        positions.push(pos)
        slot_mapping.push(this.compute_slot(st.block_ids, pos))
      }
      const end = input_ids.length
      spans.push([start, end])
      num_scheduled_tokens.push(n)
      query_start_loc.push(end)
      seq_lens.push(st.num_computed_tokens + n)
      draft_token_ids.push(drafts)
      if (drafts.length > 0) {
        // Spec decode: logits for the bonus position AND each draft position.
        for (let j = drafts.length; j >= 0; j--) logits_indices.push(end - 1 - j)
      } else {
        logits_indices.push(end - 1) // query_start_loc[1:] - 1
      }
    }
    const prepared: PreparedInputs = {
      req_ids,
      num_scheduled_tokens,
      input_ids,
      positions,
      slot_mapping,
      query_start_loc,
      seq_lens,
      num_actual_tokens: input_ids.length,
      logits_indices,
      spans,
      draft_token_ids,
    }
    this.log.emit(
      'GPUModelRunner',
      'prepare_inputs',
      `_prepare_inputs: super-sequence of ${input_ids.length} token(s) from ${req_ids.length} request(s); query_start_loc=[${query_start_loc.join(', ')}] seq_lens=[${seq_lens.join(', ')}]`,
      { req_ids, input_ids, positions, slot_mapping, query_start_loc, seq_lens, logits_indices },
      REF.GPUModelRunner_prepare_inputs,
    )
    this.log.emit(
      'GPUModelRunner',
      'slot_mapping',
      `slot_mapping = block_table[pos // ${this.cfg.block_size}] * ${this.cfg.block_size} + pos % ${this.cfg.block_size} -> [${slot_mapping.join(', ')}]`,
      { slot_mapping, positions },
      REF.BlockTable_slot_formula,
    )
    return prepared
  }

  // -------------------------------------------------------- execute_model

  execute_model(so: SchedulerOutput): null {
    if (this.execute_model_state !== null) {
      throw new Error('State error: sample_tokens() must be called after execute_model() returns None.')
    }
    this._update_states(so)
    if (so.total_num_scheduled_tokens === 0) return null
    const prepared = this._prepare_inputs(so)
    // Forward pass: attention kernels write K/V for every scheduled token into
    // paged memory via reshape_and_cache_flash at slot_mapping.
    prepared.slot_mapping.forEach((slot, i) => {
      if (slot === PAD_SLOT_ID) return
      this.kv_cache[slot] = prepared.input_ids[i]
      const r = prepared.spans.findIndex(([a, b]) => i >= a && i < b)
      this.kv_writer[slot] = r === -1 ? null : prepared.req_ids[r]
    })
    this.log.emit(
      'GPUModelRunner',
      'forward',
      `forward: one flattened batch of ${prepared.num_actual_tokens} token(s); reshape_and_cache_flash writes K/V into slots [${prepared.slot_mapping.join(', ')}]`,
      { num_actual_tokens: prepared.num_actual_tokens, slots: prepared.slot_mapping },
      REF.GPUModelRunner_execute_model,
    )
    // "Logits" only at logits_indices: the oracle's next token for the prefix
    // ending at that position.
    const logits_argmax: number[] = []
    prepared.req_ids.forEach((req_id, r) => {
      const st = this.requests.get(req_id) as CachedRequestState
      const drafts = prepared.draft_token_ids[r]
      const [start, end] = prepared.spans[r]
      const positions_for_logits = drafts.length > 0 ? Array.from({ length: drafts.length + 1 }, (_, j) => end - 1 - drafts.length + j) : [end - 1]
      const all = [...st.prompt_token_ids, ...st.output_token_ids, ...drafts]
      for (const li of positions_for_logits) {
        if (li < start) throw new Error('logits index outside request span')
        logits_argmax.push(this.oracle(req_id, all.slice(0, prepared.positions[li] + 1)))
      }
    })
    this.log.emit(
      'GPUModelRunner',
      'compute_logits',
      `hidden_states[logits_indices=[${prepared.logits_indices.join(', ')}]] -> compute_logits: ${logits_argmax.length} row(s)`,
      { logits_indices: prepared.logits_indices, argmax: logits_argmax },
      REF.GPUModelRunner_compute_logits,
    )
    this.execute_model_state = { scheduler_output: so, prepared, logits_argmax }
    this.log.emit('GPUModelRunner', 'stash_state', 'execute_model returns None; ExecuteModelState stashed until sample_tokens()', {}, REF.GPUModelRunner_ExecuteModelState)
    return null
  }

  // -------------------------------------------------------- sample_tokens

  sample_tokens(grammar_output: GrammarOutput | null): ModelRunnerOutput {
    const state = this.execute_model_state
    if (state === null) {
      return { req_ids: [], req_id_to_index: {}, sampled_token_ids: [] }
    }
    this.execute_model_state = null
    const { prepared, scheduler_output: so } = state
    const vocab = this.cfg.vocab_size()

    // Grammar bitmask rows are ordered by structured_output_request_ids.
    const bitmask_rows = new Map<string, number[][]>()
    if (grammar_output) {
      let row = 0
      for (const id of grammar_output.structured_output_request_ids) {
        const n_rows = 1 + (so.scheduled_spec_decode_tokens[id]?.length ?? 0)
        bitmask_rows.set(id, grammar_output.grammar_bitmask.slice(row, row + n_rows))
        row += n_rows
      }
      this.log.emit('GPUModelRunner', 'apply_grammar_bitmask', `apply_grammar_bitmask: ${grammar_output.grammar_bitmask.length} row(s); disallowed logits set to -inf`, { request_ids: grammar_output.structured_output_request_ids }, REF.GPUModelRunner_apply_grammar_bitmask)
    }
    const mask = (req_id: string, row_idx: number, token: number): number => {
      const rows = bitmask_rows.get(req_id)
      if (!rows || !rows[row_idx]) return token
      const allowed = unpack_bitmask(rows[row_idx], vocab)
      if (allowed[token]) return token
      const first = allowed.findIndex(Boolean)
      return first === -1 ? token : first
    }

    const sampled_token_ids: number[][] = []
    this.last_verification = []
    let logit_row = 0
    prepared.req_ids.forEach((req_id, r) => {
      const st = this.requests.get(req_id) as CachedRequestState
      const n = prepared.num_scheduled_tokens[r]
      const drafts = prepared.draft_token_ids[r]
      const still_prefilling = st.num_computed_tokens + n < st.prompt_token_ids.length + st.output_token_ids.length
      if (drafts.length === 0) {
        const argmax = state.logits_argmax[logit_row++]
        if (still_prefilling) {
          sampled_token_ids.push([])
          return
        }
        const tok = mask(req_id, 0, argmax)
        sampled_token_ids.push([tok])
        st.output_token_ids.push(tok)
        this.log.emit('Sampler', 'sample', `Sampler: ${req_id} -> token ${tok}`, { request_id: req_id, token: tok }, REF.Sampler)
        return
      }
      // Rejection sampler: target argmax for draft positions + bonus.
      const target = state.logits_argmax.slice(logit_row, logit_row + drafts.length + 1)
      logit_row += drafts.length + 1
      const accepted: boolean[] = []
      const out: number[] = []
      const stochastic = this.cfg.spec?.acceptance_rate != null
      for (let j = 0; j < drafts.length; j++) {
        const t = mask(req_id, j, target[j])
        const ok = stochastic ? this.rand() < (this.cfg.spec?.acceptance_rate as number) : drafts[j] === t
        accepted.push(ok)
        if (!ok) {
          out.push(t) // recovered token
          break
        }
        out.push(drafts[j])
      }
      if (accepted.every(Boolean) && accepted.length === drafts.length) {
        out.push(mask(req_id, drafts.length, target[drafts.length])) // bonus token
      }
      sampled_token_ids.push(out)
      st.output_token_ids.push(...out)
      const rec: VerificationRecord = { req_id, draft_token_ids: drafts, target_argmax: target, accepted, sampled: out, mode: stochastic ? 'stochastic' : 'greedy' }
      this.last_verification.push(rec)
      this.log.emit(
        'RejectionSampler',
        'verify',
        `RejectionSampler ${req_id}: drafts [${drafts.join(', ')}] vs target [${target.join(', ')}] -> accepted ${accepted.filter(Boolean).length}/${drafts.length}, output [${out.join(', ')}]${accepted.every(Boolean) && accepted.length === drafts.length ? ' (+bonus)' : ' (+recovered)'}`,
        { ...rec },
        REF.RejectionSampler,
      )
    })

    // Draft proposal for the next step (ngram).
    this._draft_token_ids = null
    this.last_ngram = {}
    this.pending_drafts.clear()
    if (this.drafter) {
      const ids: string[] = []
      const drafts: number[][] = []
      prepared.req_ids.forEach((req_id, r) => {
        const st = this.requests.get(req_id) as CachedRequestState
        if (sampled_token_ids[r].length === 0) return
        const ctx = [...st.prompt_token_ids, ...st.output_token_ids]
        const m = this.drafter!.propose(ctx)
        this.last_ngram[req_id] = m
        if (m.drafts.length === 0) return
        ids.push(req_id)
        drafts.push(m.drafts)
        this.pending_drafts.set(req_id, m.drafts)
        this.log.emit(
          'NgramProposer',
          'propose',
          `NgramProposer ${req_id}: ${m.ngram_len}-gram suffix matched at position ${m.match_start} -> propose [${m.drafts.join(', ')}]`,
          { request_id: req_id, ...m },
          REF.NgramProposer,
        )
      })
      if (ids.length) this._draft_token_ids = { req_ids: ids, draft_token_ids: drafts }
    }

    const out: ModelRunnerOutput = {
      req_ids: prepared.req_ids,
      req_id_to_index: Object.fromEntries(prepared.req_ids.map((id, i) => [id, i])),
      sampled_token_ids,
    }
    this.log.emit(
      'GPUModelRunner',
      'model_runner_output',
      `ModelRunnerOutput: sampled_token_ids=${JSON.stringify(sampled_token_ids)} (D2H copy, back to EngineCore)`,
      { req_ids: prepared.req_ids, sampled_token_ids },
      REF.ModelRunnerOutput,
    )
    return out
  }

  take_draft_token_ids(): DraftTokenIds | null {
    const d = this._draft_token_ids
    this._draft_token_ids = null
    return d
  }

  /** Block-level view of paged memory for the UI. */
  block_contents(block_id: number): (number | null)[] {
    const bs = this.cfg.block_size
    return this.kv_cache.slice(block_id * bs, (block_id + 1) * bs)
  }

  block_writers(block_id: number): (string | null)[] {
    const bs = this.cfg.block_size
    return this.kv_writer.slice(block_id * bs, (block_id + 1) * bs)
  }
}
