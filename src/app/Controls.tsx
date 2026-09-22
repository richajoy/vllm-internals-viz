import type { EngineConfig } from '../sim/engine_core'
import { PRESETS, type Scenario, type ScenarioRequest } from '../sim/scenario'

interface Props {
  scenario: Scenario
  presetName: string
  onPreset: (name: string) => void
  onChange: (s: Scenario) => void
}

function Num({ label, value, min, max, onChange, hint }: { label: string; value: number; min?: number; max?: number; onChange: (v: number) => void; hint?: string }) {
  return (
    <label className="flex items-center justify-between gap-2" title={hint}>
      <span className="mono text-[11px]">{label}</span>
      <input type="number" value={value} min={min} max={max} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

function Bool({ label, value, onChange, hint }: { label: string; value: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="flex items-center justify-between gap-2" title={hint}>
      <span className="mono text-[11px]">{label}</span>
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
    </label>
  )
}

export function Controls({ scenario, presetName, onPreset, onChange }: Props) {
  const c = scenario.config
  const set = <K extends keyof EngineConfig>(k: K, v: EngineConfig[K]) => onChange({ ...scenario, config: { ...c, [k]: v } })
  const setReq = (i: number, patch: Partial<ScenarioRequest>) => {
    const requests = scenario.requests.map((r, j) => (j === i ? { ...r, ...patch } : r))
    onChange({ ...scenario, requests })
  }
  const spec = c.spec
  return (
    <div className="panel p-3 flex flex-col gap-3 text-[12px]">
      <div className="flex items-baseline justify-between">
        <div className="panel-title">Scenario</div>
        <select value={presetName} onChange={(e) => onPreset(e.target.value)}>
          {PRESETS.map((p) => (
            <option key={p.name} value={p.name}>{p.title}</option>
          ))}
          {presetName === 'custom' && <option value="custom">custom</option>}
        </select>
      </div>
      <p className="hint m-0 leading-snug">{scenario.description}</p>

      <div className="grid grid-cols-1 gap-y-1">
        <Num label="block_size" value={c.block_size} min={1} max={64} onChange={(v) => set('block_size', v)} hint="tokens per KV block (default 16)" />
        <Num label="num_gpu_blocks" value={c.num_gpu_blocks} min={2} max={64} onChange={(v) => set('num_gpu_blocks', v)} hint="block 0 is the null block" />
        <Num label="max_num_batched_tokens" value={c.max_num_batched_tokens} min={1} onChange={(v) => set('max_num_batched_tokens', v)} hint="token budget per step" />
        <Num label="max_num_seqs" value={c.max_num_seqs} min={1} onChange={(v) => set('max_num_seqs', v)} hint="max running requests" />
        <Num label="long_prefill_token_threshold" value={c.long_prefill_token_threshold} min={0} onChange={(v) => set('long_prefill_token_threshold', v)} hint="0 = off; caps one request's chunk" />
        <Num label="max_model_len" value={c.max_model_len} min={8} onChange={(v) => set('max_model_len', v)} />
        <Bool label="enable_chunked_prefill" value={c.enable_chunked_prefill} onChange={(v) => set('enable_chunked_prefill', v)} />
        <Bool label="enable_prefix_caching" value={c.enable_prefix_caching} onChange={(v) => set('enable_prefix_caching', v)} />
        <Bool label="scheduler_reserve_full_isl" value={c.scheduler_reserve_full_isl} onChange={(v) => set('scheduler_reserve_full_isl', v)} hint="admission gate: whole prompt must fit" />
        <label className="flex items-center justify-between gap-2">
          <span className="mono text-[11px]">policy</span>
          <select value={c.policy} onChange={(e) => set('policy', e.target.value as 'fcfs' | 'priority')}>
            <option value="fcfs">fcfs</option>
            <option value="priority">priority</option>
          </select>
        </label>
        <Bool label="async_scheduling" value={c.async_scheduling} onChange={(v) => onChange({ ...scenario, config: { ...c, async_scheduling: v, batch_queue_size: v ? Math.max(2, c.batch_queue_size) : 1 } })} hint="AsyncScheduler + step_with_batch_queue" />
        <Num label="batch_queue_size" value={c.batch_queue_size} min={1} max={4} onChange={(v) => set('batch_queue_size', v)} hint="max_concurrent_batches; >1 uses step_with_batch_queue" />
        <Bool
          label="speculative (ngram)"
          value={spec !== null}
          onChange={(v) =>
            onChange({
              ...scenario,
              config: {
                ...c,
                num_speculative_tokens: v ? 3 : 0,
                spec: v ? { method: 'ngram', num_speculative_tokens: 3, prompt_lookup_min: 2, prompt_lookup_max: 4, acceptance_rate: null } : null,
              },
            })
          }
        />
        {spec && (
          <>
            <Num label="num_speculative_tokens" value={spec.num_speculative_tokens} min={1} max={8} onChange={(v) => onChange({ ...scenario, config: { ...c, num_speculative_tokens: v, spec: { ...spec, num_speculative_tokens: v } } })} />
            <Num label="prompt_lookup_min" value={spec.prompt_lookup_min} min={1} onChange={(v) => set('spec', { ...spec, prompt_lookup_min: v })} />
            <Num label="prompt_lookup_max" value={spec.prompt_lookup_max} min={1} onChange={(v) => set('spec', { ...spec, prompt_lookup_max: v })} />
            <label className="flex items-center justify-between gap-2" title="empty = greedy verification (accept iff draft == target argmax)">
              <span className="mono text-[11px]">acceptance_rate</span>
              <input type="number" step={0.1} min={0} max={1} value={spec.acceptance_rate ?? ''} placeholder="greedy" onChange={(e) => set('spec', { ...spec, acceptance_rate: e.target.value === '' ? null : Number(e.target.value) })} />
            </label>
          </>
        )}
      </div>

      <div>
        <div className="flex items-baseline justify-between mb-1">
          <span className="panel-title">Requests</span>
          <button
            className="tbtn"
            onClick={() =>
              onChange({
                ...scenario,
                requests: [...scenario.requests, { id: String.fromCharCode(65 + scenario.requests.length), prompt: 'New prompt goes here', arrival_step: 0, max_tokens: 3, continuation: 'and then some' }],
              })
            }
          >
            add
          </button>
        </div>
        <div className="flex flex-col gap-2">
          {scenario.requests.map((r, i) => (
            <div key={i} className="grid gap-1 p-2 rounded" style={{ background: 'var(--free)', gridTemplateColumns: '1fr' }}>
              <div className="flex items-center gap-2">
                <input type="text" className="mono w-12" value={r.id} onChange={(e) => setReq(i, { id: e.target.value })} />
                <span className="hint text-[11px]">arrives step</span>
                <input type="number" value={r.arrival_step} min={0} onChange={(e) => setReq(i, { arrival_step: Number(e.target.value) })} style={{ width: '3.5em' }} />
                <span className="hint text-[11px]">max_tokens</span>
                <input type="number" value={r.max_tokens} min={1} onChange={(e) => setReq(i, { max_tokens: Number(e.target.value) })} style={{ width: '3.5em' }} />
                {c.policy === 'priority' && (
                  <>
                    <span className="hint text-[11px]">priority</span>
                    <input type="number" value={r.priority ?? 0} onChange={(e) => setReq(i, { priority: Number(e.target.value) })} style={{ width: '3.5em' }} />
                  </>
                )}
                <button className="tbtn ml-auto" onClick={() => onChange({ ...scenario, requests: scenario.requests.filter((_, j) => j !== i) })} aria-label={`remove request ${r.id}`}>
                  ×
                </button>
              </div>
              <input type="text" value={r.prompt} onChange={(e) => setReq(i, { prompt: e.target.value })} placeholder="prompt" />
              <input type="text" value={r.continuation ?? ''} onChange={(e) => setReq(i, { continuation: e.target.value })} placeholder="model continuation (what the mock model will say)" />
              <input type="text" value={r.guided_choice?.join(' | ') ?? ''} onChange={(e) => setReq(i, { guided_choice: e.target.value.trim() ? e.target.value.split('|').map((s) => s.trim()) : undefined })} placeholder="guided choices, e.g. Positive | Negative (optional)" />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
