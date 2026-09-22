// Differential test: replay traces produced by tools/trace_vllm.py against the
// real vLLM Scheduler (commit adc3e03517) and require identical state.

import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { AsyncScheduler } from './async_scheduler'
import { get_request_block_hasher } from './kv_cache_utils'
import type { ModelRunnerOutput, SchedulerOutput } from './output'
import { Request, RequestStatusName } from './request'
import { DEFAULT_SCHEDULER_CONFIG, Scheduler } from './scheduler'

const EOS_TOKEN_ID = 50256
const FIXTURES = join(__dirname, '..', '..', 'tests', 'fixtures')

interface FixtureRequest {
  request_id: string
  prompt_token_ids: number[]
  max_tokens: number
  arrival_step: number
  priority: number
  output_token_ids: number[]
  draft_token_ids_per_step?: number[][]
}

interface Snapshot {
  free_queue_order: number[]
  num_free_blocks: number
  blocks: Record<string, { ref_cnt: number; has_hash: boolean }>
  cached_hash_count: number
  requests: Record<string, { status: string; num_computed_tokens: number; num_tokens: number; num_prompt_tokens: number; num_output_tokens: number; num_output_placeholders: number; spec_token_ids: number[]; block_ids: number[][] }>
  running_order: string[]
  waiting_order: string[]
  skipped_waiting_order: string[]
}

interface FixtureStep {
  step: number
  arrivals: string[]
  scheduler_output: {
    scheduled_new_reqs: { req_id: string; block_ids: number[][]; num_computed_tokens: number }[]
    scheduled_cached_reqs: { req_ids: string[]; resumed_req_ids: string[]; new_block_ids: (number[][] | null)[]; num_computed_tokens: number[]; num_output_tokens: number[] }
    num_scheduled_tokens: Record<string, number>
    total_num_scheduled_tokens: number
    scheduled_spec_decode_tokens: Record<string, number[]>
    num_common_prefix_blocks: number[]
    finished_req_ids: string[]
    preempted: string[]
  } | null
  after_schedule: Snapshot
  update_for_step: number | null
  model_runner_output?: { req_ids: string[]; sampled_token_ids: number[][] }
  outputs: Record<string, { new_token_ids: number[]; finish_reason: string | null }>
  after_update: Snapshot
}

interface Fixture {
  name: string
  config: Record<string, number | boolean | string>
  requests: FixtureRequest[]
  steps: FixtureStep[]
  final_requests: Record<string, { status: string; output_token_ids: number[]; num_preemptions: number }>
}

function snapshot(s: Scheduler): Snapshot {
  const pool = s.kv_cache_manager.block_pool
  const blocks: Snapshot['blocks'] = {}
  for (const b of pool.blocks) if (!b.is_null) blocks[String(b.block_id)] = { ref_cnt: b.ref_cnt, has_hash: b.block_hash !== null }
  const requests: Snapshot['requests'] = {}
  for (const [rid, r] of s.requests) {
    requests[rid] = {
      status: RequestStatusName[r.status],
      num_computed_tokens: r.num_computed_tokens,
      num_tokens: r.num_tokens,
      num_prompt_tokens: r.num_prompt_tokens,
      num_output_tokens: r.num_output_tokens,
      num_output_placeholders: r.num_output_placeholders,
      spec_token_ids: r.spec_token_ids.slice(),
      block_ids: s.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks.has(rid) ? s.kv_cache_manager.get_block_ids(rid) : [],
    }
  }
  return {
    free_queue_order: pool.free_block_queue.get_all_free_blocks().map((b) => b.block_id),
    num_free_blocks: pool.get_num_free_blocks(),
    blocks,
    cached_hash_count: pool.cached_block_hash_to_block.size,
    requests,
    running_order: s.running.map((r) => r.request_id),
    waiting_order: s.waiting.toArray().map((r) => r.request_id),
    skipped_waiting_order: s.skipped_waiting.toArray().map((r) => r.request_id),
  }
}

function snapshotOutput(out: SchedulerOutput): NonNullable<FixtureStep['scheduler_output']> {
  return {
    scheduled_new_reqs: out.scheduled_new_reqs.map((r) => ({ req_id: r.req_id, block_ids: r.block_ids, num_computed_tokens: r.num_computed_tokens })),
    scheduled_cached_reqs: {
      req_ids: out.scheduled_cached_reqs.req_ids,
      resumed_req_ids: [...out.scheduled_cached_reqs.resumed_req_ids].sort(),
      new_block_ids: out.scheduled_cached_reqs.new_block_ids,
      num_computed_tokens: out.scheduled_cached_reqs.num_computed_tokens,
      num_output_tokens: out.scheduled_cached_reqs.num_output_tokens,
    },
    num_scheduled_tokens: out.num_scheduled_tokens,
    total_num_scheduled_tokens: out.total_num_scheduled_tokens,
    scheduled_spec_decode_tokens: out.scheduled_spec_decode_tokens,
    num_common_prefix_blocks: out.num_common_prefix_blocks,
    finished_req_ids: [...out.finished_req_ids].sort(),
    preempted: [...out.preempted_req_ids].sort(),
  }
}

/** Mirrors MockModel in tools/trace_vllm.py. */
class MockModel {
  expected: Record<string, number[]>
  cursor: Record<string, number> = {}
  constructor(reqs: FixtureRequest[]) {
    this.expected = Object.fromEntries(reqs.map((r) => [r.request_id, r.output_token_ids ?? []]))
    for (const r of reqs) this.cursor[r.request_id] = 0
  }
  peek(rid: string, offset: number): number {
    const i = this.cursor[rid] + offset
    return i < this.expected[rid].length ? this.expected[rid][i] : EOS_TOKEN_ID
  }
  next(rid: string): number {
    const t = this.peek(rid, 0)
    this.cursor[rid] += 1
    return t
  }
  sample(rid: string, drafts: number[]): number[] {
    let n = 0
    for (const d of drafts) {
      if (d !== this.peek(rid, n)) break
      n += 1
    }
    return Array.from({ length: n + 1 }, () => this.next(rid))
  }
}

function replay(fx: Fixture): void {
  const c = fx.config
  const cfg = {
    ...DEFAULT_SCHEDULER_CONFIG,
    block_size: c.block_size as number,
    num_gpu_blocks: c.num_gpu_blocks as number,
    max_num_batched_tokens: c.max_num_batched_tokens as number,
    max_num_seqs: c.max_num_seqs as number,
    max_model_len: (c.max_model_len as number) ?? (c.max_num_batched_tokens as number),
    enable_prefix_caching: c.enable_prefix_caching as boolean,
    enable_chunked_prefill: c.enable_chunked_prefill as boolean,
    long_prefill_token_threshold: c.long_prefill_token_threshold as number,
    policy: c.policy as 'fcfs' | 'priority',
    async_scheduling: c.async_scheduling as boolean,
    num_speculative_tokens: c.num_speculative_tokens as number,
    num_lookahead_tokens: 0, // ngram
  }
  const s = cfg.async_scheduling ? new AsyncScheduler(cfg) : new Scheduler(cfg)
  const model = new MockModel(fx.requests)
  const draft_queues: Record<string, number[][]> = Object.fromEntries(fx.requests.map((r) => [r.request_id, (r.draft_token_ids_per_step ?? []).slice()]))
  const by_id = Object.fromEntries(fx.requests.map((r) => [r.request_id, r]))

  const push_drafts = () => {
    const ids: string[] = []
    const drafts: number[][] = []
    for (const [rid, q] of Object.entries(draft_queues)) {
      const req = s.requests.get(rid)
      if (!req || req.is_finished() || req.is_prefill_chunk || q.length === 0) continue
      ids.push(rid)
      drafts.push(q.shift() as number[])
    }
    if (ids.length) s.update_draft_token_ids({ req_ids: ids, draft_token_ids: drafts })
  }

  const update = (out: SchedulerOutput, samples: Record<string, boolean>) => {
    const req_ids = Object.keys(out.num_scheduled_tokens)
    const sampled = req_ids.map((rid) => (samples[rid] ? model.sample(rid, out.scheduled_spec_decode_tokens[rid] ?? []) : []))
    const mro: ModelRunnerOutput = { req_ids, req_id_to_index: Object.fromEntries(req_ids.map((r, i) => [r, i])), sampled_token_ids: sampled }
    const eco = s.update_from_output(out, mro)
    const outputs: Record<string, { new_token_ids: number[]; finish_reason: string | null }> = {}
    for (const o of eco.outputs) {
      outputs[o.request_id] = { new_token_ids: o.new_token_ids, finish_reason: o.finish_reason }
      // vLLM trims new_token_ids in place (`del new_token_ids[num_new:]`) and
      // that list aliases sampled_token_ids[req_index], so the recorded model
      // output reflects the trim. Mirror it for the comparison.
      if (o.finish_reason) mro.sampled_token_ids[mro.req_id_to_index[o.request_id]] = o.new_token_ids
    }
    push_drafts()
    return { mro, outputs }
  }

  let pending: { step: number; out: SchedulerOutput; samples: Record<string, boolean> } | null = null
  for (const fs of fx.steps) {
    const where = `${fx.name} step ${fs.step}`
    for (const rid of fs.arrivals) {
      const r = by_id[rid]
      s.add_request(
        new Request({
          request_id: rid,
          prompt_token_ids: r.prompt_token_ids,
          sampling_params: { max_tokens: r.max_tokens, eos_token_id: EOS_TOKEN_ID },
          arrival_time: fs.step,
          priority: r.priority,
          block_hasher: get_request_block_hasher(cfg.block_size),
        }),
      )
    }
    if (fs.scheduler_output !== null) {
      const out = s.schedule()
      const samples = Object.fromEntries(Object.keys(out.num_scheduled_tokens).map((rid) => [rid, !s.requests.get(rid)!.is_prefill_chunk]))
      expect(snapshotOutput(out), `${where} scheduler_output`).toEqual(fs.scheduler_output)
      expect(snapshot(s), `${where} after_schedule`).toEqual(fs.after_schedule)
      if (cfg.async_scheduling) {
        if (pending) {
          expect(pending.step, `${where} update_for_step`).toBe(fs.update_for_step)
          const { mro, outputs } = update(pending.out, pending.samples)
          expect({ req_ids: mro.req_ids, sampled_token_ids: mro.sampled_token_ids }, `${where} model_runner_output`).toEqual(fs.model_runner_output)
          expect(outputs, `${where} outputs`).toEqual(fs.outputs)
        }
        pending = { step: fs.step, out, samples }
      } else {
        const { mro, outputs } = update(out, samples)
        expect({ req_ids: mro.req_ids, sampled_token_ids: mro.sampled_token_ids }, `${where} model_runner_output`).toEqual(fs.model_runner_output)
        expect(outputs, `${where} outputs`).toEqual(fs.outputs)
      }
    } else {
      // Trailing async drain step.
      expect(pending, where).not.toBeNull()
      const { mro, outputs } = update(pending!.out, pending!.samples)
      expect({ req_ids: mro.req_ids, sampled_token_ids: mro.sampled_token_ids }, `${where} model_runner_output`).toEqual(fs.model_runner_output)
      expect(outputs, `${where} outputs`).toEqual(fs.outputs)
      pending = null
    }
    expect(snapshot(s), `${where} after_update`).toEqual(fs.after_update)
  }
}

describe('differential replay vs real vLLM Scheduler traces', () => {
  const files = readdirSync(FIXTURES).filter((f) => f.endsWith('.json'))
  it('has fixtures', () => expect(files.length).toBeGreaterThan(0))
  for (const f of files) {
    it(f, () => {
      const fx = JSON.parse(readFileSync(join(FIXTURES, f), 'utf8')) as Fixture
      replay(fx)
    })
  }
})
