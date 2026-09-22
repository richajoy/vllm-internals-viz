import { describe, expect, it } from 'vitest'
import { AsyncScheduler } from './async_scheduler'
import { get_request_block_hasher } from './kv_cache_utils'
import type { ModelRunnerOutput, SchedulerOutput } from './output'
import { Request, RequestStatus, RequestStatusName } from './request'
import { DEFAULT_SCHEDULER_CONFIG, Scheduler, type SchedulerConfig } from './scheduler'

function mkScheduler(over: Partial<SchedulerConfig> = {}): Scheduler {
  const cfg: SchedulerConfig = { ...DEFAULT_SCHEDULER_CONFIG, block_size: 4, num_gpu_blocks: 12, max_num_batched_tokens: 64, max_num_seqs: 8, max_model_len: 64, ...over }
  return cfg.async_scheduling ? new AsyncScheduler(cfg) : new Scheduler(cfg)
}

function mkReq(id: string, prompt: number[], max_tokens: number, s: Scheduler, arrival = 0, priority = 0): Request {
  const r = new Request({
    request_id: id,
    prompt_token_ids: prompt,
    sampling_params: { max_tokens },
    arrival_time: arrival,
    priority,
    block_hasher: s.cfg.enable_prefix_caching ? get_request_block_hasher(s.cfg.block_size) : null,
  })
  return r
}

/** Mock model: a request that finished its prompt emits token 100+step. */
function mockOutput(out: SchedulerOutput, s: Scheduler, step: number): ModelRunnerOutput {
  const req_ids = Object.keys(out.num_scheduled_tokens)
  const sampled = req_ids.map((id) => {
    const r = s.requests.get(id)!
    const placeholders = r.num_output_placeholders
    // After _update_after_schedule num_computed_tokens is optimistic; a prefill
    // chunk that has not reached the prompt end yields nothing.
    return r.num_computed_tokens >= r.num_tokens + placeholders ? [100 + step] : []
  })
  return { req_ids, req_id_to_index: Object.fromEntries(req_ids.map((id, i) => [id, i])), sampled_token_ids: sampled }
}

function run(s: Scheduler, max_steps = 50): { steps: SchedulerOutput[]; outputs: Record<string, number[]> } {
  const steps: SchedulerOutput[] = []
  const outputs: Record<string, number[]> = {}
  for (let i = 0; i < max_steps && s.has_requests(); i++) {
    const out = s.schedule()
    steps.push(out)
    const eco = s.update_from_output(out, mockOutput(out, s, i))
    for (const o of eco.outputs) (outputs[o.request_id] ??= []).push(...o.new_token_ids)
  }
  return { steps, outputs }
}

describe('Scheduler.schedule', () => {
  it('schedules two prefills in one step and decodes them together (continuous batching)', () => {
    const s = mkScheduler({ enable_prefix_caching: false })
    s.add_request(mkReq('A', [1, 2, 3, 4, 5], 2, s))
    s.add_request(mkReq('B', [1, 6, 5, 7, 8, 9, 10], 2, s))
    const out0 = s.schedule()
    expect(out0.num_scheduled_tokens).toEqual({ A: 5, B: 7 })
    expect(out0.scheduled_new_reqs.map((r) => [r.req_id, r.block_ids[0]])).toEqual([
      ['A', [1, 2]],
      ['B', [3, 4]],
    ])
    s.update_from_output(out0, mockOutput(out0, s, 0))
    const out1 = s.schedule()
    expect(out1.num_scheduled_tokens).toEqual({ A: 1, B: 1 })
    expect(out1.scheduled_cached_reqs.req_ids).toEqual(['A', 'B'])
    // A's 6th token needs a new block? 6 tokens -> 2 blocks, already has 2. B 8 -> 2.
    expect(out1.scheduled_cached_reqs.new_block_ids).toEqual([null, null])
    s.update_from_output(out1, mockOutput(out1, s, 1))
    expect(s.running).toHaveLength(0)
    expect(s.finished_req_ids).toEqual(new Set(['A', 'B']))
    const out2 = s.schedule()
    expect(out2.finished_req_ids).toEqual(new Set(['A', 'B']))
    expect(out2.total_num_scheduled_tokens).toBe(0)
  })

  it('blog Fig 5: 18-token prompt, budget 8 -> 3 chunks, sampled token only after the last', () => {
    const s = mkScheduler({ max_num_batched_tokens: 8, enable_prefix_caching: false })
    s.add_request(mkReq('A', Array.from({ length: 18 }, (_, i) => i + 1), 2, s))
    const { steps, outputs } = run(s)
    expect(steps.slice(0, 4).map((o) => o.num_scheduled_tokens['A'])).toEqual([8, 8, 2, 1])
    expect(steps[0].scheduled_new_reqs[0].block_ids[0]).toEqual([1, 2])
    expect(steps[1].scheduled_cached_reqs.new_block_ids[0]).toEqual([[3, 4]])
    expect(steps[2].scheduled_cached_reqs.new_block_ids[0]).toEqual([[5]])
    expect(outputs['A']).toEqual([102, 103])
  })

  it('long_prefill_token_threshold caps chunks even with a large budget', () => {
    const s = mkScheduler({ long_prefill_token_threshold: 6, enable_prefix_caching: false })
    s.add_request(mkReq('A', Array.from({ length: 14 }, (_, i) => i + 1), 1, s))
    const { steps } = run(s)
    expect(steps.slice(0, 3).map((o) => o.num_scheduled_tokens['A'])).toEqual([6, 6, 2])
  })

  it('chunked prefill disabled: prompt larger than budget is never scheduled', () => {
    const s = mkScheduler({ max_num_batched_tokens: 8, enable_chunked_prefill: false, enable_prefix_caching: false })
    s.add_request(mkReq('A', Array.from({ length: 9 }, (_, i) => i + 1), 1, s))
    const out = s.schedule()
    expect(out.total_num_scheduled_tokens).toBe(0)
    expect(s.waiting.length).toBe(1)
  })

  it('FCFS preemption: last running request is preempted, goes to the front of waiting, resumes later', () => {
    // 6 usable blocks (7 - null). Three 6-token requests need 2 blocks each = 6; first decode past 8 tokens needs a 3rd.
    const s = mkScheduler({ num_gpu_blocks: 7, enable_prefix_caching: false, scheduler_reserve_full_isl: false })
    s.add_request(mkReq('A', [1, 2, 3, 4, 5, 6], 6, s))
    s.add_request(mkReq('B', [1, 2, 3, 4, 5, 7], 6, s))
    s.add_request(mkReq('C', [1, 2, 3, 4, 5, 8], 6, s))
    const out0 = s.schedule()
    expect(Object.keys(out0.num_scheduled_tokens)).toEqual(['A', 'B', 'C'])
    s.update_from_output(out0, mockOutput(out0, s, 0)) // 7 tokens each
    const out1 = s.schedule() // 8 tokens: still 2 blocks
    s.update_from_output(out1, mockOutput(out1, s, 1))
    const out2 = s.schedule() // 9th token slot: A needs a 3rd block -> none free -> preempt C (running.pop())
    s.update_from_output(out2, mockOutput(out2, s, 2))
    const out3 = s.schedule()
    expect(out3.preempted_req_ids).toEqual(new Set(['C']))
    expect(Object.keys(out3.num_scheduled_tokens)).toEqual(['A', 'B'])
    expect(s.requests.get('C')!.status).toBe(RequestStatus.PREEMPTED)
    expect(s.requests.get('C')!.num_computed_tokens).toBe(0)
    expect(s.waiting.toArray().map((r) => r.request_id)).toEqual(['C'])
    // Preemption freed C's 2 blocks; A took one, B took one.
    expect(s.kv_cache_manager.get_block_ids('A')[0]).toHaveLength(3)
    s.update_from_output(out3, mockOutput(out3, s, 3))
    // Run to completion: C must resume with num_preemptions=1 and finish with 6 outputs.
    const { outputs } = run(s)
    expect(s.requests.size).toBe(0)
    expect(outputs['C']).toHaveLength(6 - 3) // 3 tokens produced before preemption are kept
    expect(s.kv_cache_manager.block_pool.get_num_free_blocks()).toBe(6)
  })

  it('priority policy preempts the largest (priority, arrival_time) and rolls back its scheduled budget', () => {
    const s = mkScheduler({ num_gpu_blocks: 7, enable_prefix_caching: false, scheduler_reserve_full_isl: false, policy: 'priority' })
    s.add_request(mkReq('low', [1, 2, 3, 4, 5, 6], 6, s, 0, 5))
    s.add_request(mkReq('hi', [1, 2, 3, 4, 5, 7], 6, s, 1, 0))
    s.add_request(mkReq('mid', [1, 2, 3, 4, 5, 8], 6, s, 2, 2))
    const out0 = s.schedule()
    expect(out0.scheduled_new_reqs.map((r) => r.req_id)).toEqual(['hi', 'mid', 'low'])
    s.update_from_output(out0, mockOutput(out0, s, 0))
    const out1 = s.schedule()
    s.update_from_output(out1, mockOutput(out1, s, 1))
    const out2 = s.schedule()
    s.update_from_output(out2, mockOutput(out2, s, 2))
    const out3 = s.schedule()
    expect(out3.preempted_req_ids).toEqual(new Set(['low']))
    expect(Object.keys(out3.num_scheduled_tokens).sort()).toEqual(['hi', 'mid'])
  })

  it('prefix caching: second request reuses the first block and schedules only the tail', () => {
    const s = mkScheduler()
    s.add_request(mkReq('A', [1, 2, 3, 4, 5, 6], 1, s))
    const { steps } = run(s)
    expect(steps[0].num_scheduled_tokens['A']).toBe(6)
    s.add_request(mkReq('B', [1, 2, 3, 4, 9, 9], 1, s))
    const out = s.schedule()
    expect(out.scheduled_new_reqs[0].num_computed_tokens).toBe(4)
    expect(out.num_scheduled_tokens['B']).toBe(2)
    expect(out.scheduled_new_reqs[0].block_ids[0][0]).toBe(1)
    expect(s.kv_cache_manager.block_pool.blocks[1].ref_cnt).toBe(1)
  })

  it('async scheduling: placeholders let the next step be scheduled before tokens arrive', () => {
    const s = mkScheduler({ async_scheduling: true, enable_prefix_caching: false })
    s.add_request(mkReq('A', [1, 2, 3, 4, 5], 3, s))
    const out0 = s.schedule()
    const A = s.requests.get('A')!
    expect(A.num_output_placeholders).toBe(1)
    expect(A.num_computed_tokens).toBe(5)
    // Schedule step 1 BEFORE step 0's output: 1 new token (the placeholder position).
    const out1 = s.schedule()
    expect(out1.num_scheduled_tokens['A']).toBe(1)
    expect(A.num_output_placeholders).toBe(2)
    expect(A.num_computed_tokens).toBe(6)
    // Now step 0 output arrives.
    s.update_from_output(out0, { req_ids: ['A'], req_id_to_index: { A: 0 }, sampled_token_ids: [[100]] })
    expect(A.num_output_placeholders).toBe(1)
    expect(A.num_output_tokens).toBe(1)
    const out2 = s.schedule()
    expect(out2.num_scheduled_tokens['A']).toBe(1)
    s.update_from_output(out1, { req_ids: ['A'], req_id_to_index: { A: 0 }, sampled_token_ids: [[101]] })
    // max_tokens=3 reached by out2's in-flight token: the scheduler must not schedule a 4th.
    const out3 = s.schedule()
    expect(out3.num_scheduled_tokens['A']).toBeUndefined()
    s.update_from_output(out2, { req_ids: ['A'], req_id_to_index: { A: 0 }, sampled_token_ids: [[102]] })
    expect(RequestStatusName[A.status]).toBe('FINISHED_LENGTH_CAPPED')
    expect(s.has_requests()).toBe(false)
  })

  it('spec decode: rejected drafts roll back num_computed_tokens', () => {
    const s = mkScheduler({ num_speculative_tokens: 3, enable_prefix_caching: false })
    s.add_request(mkReq('A', [1, 2, 3, 4, 5], 8, s))
    const out0 = s.schedule()
    s.update_from_output(out0, { req_ids: ['A'], req_id_to_index: { A: 0 }, sampled_token_ids: [[10]] })
    s.update_draft_token_ids({ req_ids: ['A'], draft_token_ids: [[11, 12, 13]] })
    const A = s.requests.get('A')!
    expect(A.num_tokens_with_spec).toBe(9)
    const out1 = s.schedule()
    expect(out1.num_scheduled_tokens['A']).toBe(4)
    expect(out1.scheduled_spec_decode_tokens['A']).toEqual([11, 12, 13])
    expect(A.num_computed_tokens).toBe(9) // 5 prompt + 1 output + 3 drafts, optimistic
    // Accept 2 of 3 drafts + recovered token = 3 sampled -> 1 rejected.
    s.update_from_output(out1, { req_ids: ['A'], req_id_to_index: { A: 0 }, sampled_token_ids: [[11, 12, 40]] })
    expect(A.num_computed_tokens).toBe(8)
    expect(A.num_tokens).toBe(9)
    expect([...A.output_token_ids]).toEqual([10, 11, 12, 40])
  })
})
