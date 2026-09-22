import { describe, expect, it } from 'vitest'
import { PRESETS, preset } from './scenario'
import { run_simulation } from './simulation'

describe('run_simulation presets', () => {
  for (const p of PRESETS) {
    it(`${p.name} runs to completion with all requests finished`, () => {
      const res = run_simulation(preset(p.name))
      const last = res.snapshots[res.snapshots.length - 1]
      const finished = Object.values(last.outputs)
      expect(finished.length).toBe(p.requests.length)
      for (const o of finished) expect(o.finished, `${p.name}:${o.external_req_id} ${o.text}`).toBe(true)
      // No leaked blocks: everything but the null block is free again.
      expect(last.kv.num_free_blocks).toBe(p.config.num_gpu_blocks - 1)
      expect(last.scheduler.running).toEqual([])
      expect(res.events.length).toBeGreaterThan(20)
    })
  }

  it('prefix_cache: request B reuses A\'s first two blocks', () => {
    const res = run_simulation(preset('prefix_cache'))
    const ev = res.events.find((e) => e.kind === 'add_local_computed_blocks')
    expect(ev?.payload?.block_ids).toEqual([1, 2])
  })

  it('spec_decode: drafts get proposed and some accepted', () => {
    const res = run_simulation(preset('spec_decode'))
    const verify = res.events.filter((e) => e.kind === 'verify')
    expect(verify.length).toBeGreaterThan(0)
    expect(verify.some((e) => (e.payload?.accepted as boolean[]).some(Boolean))).toBe(true)
    const last = res.snapshots[res.snapshots.length - 1]
    expect(last.outputs[Object.keys(last.outputs)[0]].text).toContain('brown fox jumps over the lazy dog')
  })

  it('structured_output: answers are constrained to the choices', () => {
    const res = run_simulation(preset('structured_output'))
    const last = res.snapshots[res.snapshots.length - 1]
    for (const o of Object.values(last.outputs)) expect(['Positive', 'Negative']).toContain(o.text)
    expect(res.events.some((e) => e.kind === 'grammar_ready')).toBe(true)
    expect(res.events.some((e) => e.kind === 'apply_grammar_bitmask')).toBe(true)
  })

  it('preemption: a request is preempted and resumes', () => {
    const res = run_simulation(preset('preemption'))
    expect(res.events.some((e) => e.kind === 'preempt')).toBe(true)
    expect(res.events.some((e) => e.kind === 'resumed')).toBe(true)
  })

  it('async: placeholders appear and the batch queue holds one in-flight step', () => {
    const res = run_simulation(preset('async'))
    expect(res.events.some((e) => e.kind === 'placeholders')).toBe(true)
    expect(res.snapshots.some((s) => (s.batch_queue?.length ?? 0) === 1)).toBe(true)
  })
})
