import { useEffect, useMemo, useRef, useState } from 'react'
import type { Component } from '../sim/events'
import { PRESETS, preset, type Scenario } from '../sim/scenario'
import { run_simulation } from '../sim/simulation'
import { VLLM_COMMIT_SHORT } from '../sim/source_refs'
import { Controls } from './Controls'
import { Drift } from './Drift'
import { KVView } from './KVView'
import { PHASES, PHASE_TITLE, requestOrder, stepBoundary } from './model'
import { PipelineRail } from './PipelineRail'
import { SchedulerView } from './SchedulerView'
import { Timeline } from './Timeline'
import { WorkerView } from './WorkerView'

function readHash(): { preset: string; i: number } {
  const h = new URLSearchParams(location.hash.replace(/^#/, ''))
  const p = h.get('s') ?? PRESETS[0].name
  return { preset: PRESETS.some((x) => x.name === p) ? p : PRESETS[0].name, i: Number(h.get('i') ?? 0) || 0 }
}

export default function App() {
  const initial = useMemo(() => readHash(), [])
  const [presetName, setPresetName] = useState(initial.preset)
  const [scenario, setScenario] = useState<Scenario>(() => preset(initial.preset))
  const [cursor, setCursor] = useState(initial.i)
  const [selected, setSelected] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [showControls, setShowControls] = useState(true)

  const result = useMemo(() => {
    try {
      return { ok: true as const, value: run_simulation(scenario) }
    } catch (e) {
      return { ok: false as const, error: (e as Error).message }
    }
  }, [scenario])

  const snapshots = useMemo(() => (result.ok ? result.value.snapshots : []), [result])
  const events = useMemo(() => (result.ok ? result.value.events : []), [result])
  const idx = Math.min(cursor, Math.max(0, snapshots.length - 1))
  const snap = snapshots[idx]
  const order = useMemo(() => requestOrder(snapshots), [snapshots])

  const prevScenario = useRef(scenario)
  useEffect(() => {
    if (prevScenario.current === scenario) return
    prevScenario.current = scenario
    setCursor(0)
    setSelected(null)
    setPlaying(false)
  }, [scenario])

  useEffect(() => {
    if (presetName === 'custom') return
    history.replaceState(null, '', `#s=${presetName}&i=${idx}`)
  }, [presetName, idx])

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      setCursor((c) => {
        if (c >= snapshots.length - 1) {
          setPlaying(false)
          return c
        }
        return c + 1
      })
    }, 900 / speed)
    return () => clearInterval(id)
  }, [playing, speed, snapshots.length])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName?.match(/INPUT|SELECT|TEXTAREA/)) return
      if (e.key === 'ArrowRight') setCursor((c) => Math.min(c + 1, snapshots.length - 1))
      else if (e.key === 'ArrowLeft') setCursor((c) => Math.max(c - 1, 0))
      else if (e.key === 'ArrowDown') setCursor((c) => stepBoundary(snapshots, c, 1))
      else if (e.key === 'ArrowUp') setCursor((c) => stepBoundary(snapshots, c, -1))
      else if (e.key === ' ') {
        e.preventDefault()
        setPlaying((p) => !p)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [snapshots])

  const liveComponents = useMemo(() => {
    const set = new Set<Component>()
    if (!snap) return set
    for (let i = snap.events_from; i < snap.events_to; i++) set.add(events[i].component)
    return set
  }, [snap, events])

  const hotBlocks = useMemo(() => {
    const set = new Set<number>()
    if (!snap) return set
    for (let i = snap.events_from; i < snap.events_to; i++) {
      const p = events[i].payload
      if (!p) continue
      for (const k of ['block_ids', 'freed_block_ids', 'new_block_ids', 'prepended', 'appended']) {
        const v = p[k]
        if (Array.isArray(v)) for (const b of v) if (typeof b === 'number') set.add(b)
      }
      if (typeof p.block_id === 'number') set.add(p.block_id)
    }
    return set
  }, [snap, events])

  const stepsTotal = snapshots.length ? snapshots[snapshots.length - 1].step + 1 : 0
  const phasesDoneThisStep = new Set(snapshots.filter((s) => s.step === snap?.step && s.index <= idx).map((s) => s.phase))

  return (
    <div className="h-full flex flex-col" style={{ minWidth: 960 }}>
      <header className="flex items-baseline gap-4 px-4 py-2" style={{ borderBottom: '1px solid var(--rule)', background: 'var(--panel)' }}>
        <h1 className="m-0 text-[15px]" style={{ fontVariationSettings: "'CASL' 0.6, 'wght' 700" }}>vLLM engine, step by step</h1>
        <span className="hint text-xs">A faithful TypeScript replay of the V1 scheduler, KV cache manager and model runner, pinned to vllm-project/vllm@{VLLM_COMMIT_SHORT}. Every name is a real class or field.</span>
        <button className="tbtn ml-auto" onClick={() => setShowControls((v) => !v)}>{showControls ? 'hide scenario' : 'edit scenario'}</button>
      </header>

      {!result.ok && (
        <div className="m-4 p-3 rounded" style={{ background: 'var(--danger-soft)', color: 'var(--danger)' }}>
          The simulator hit an invariant violation with this configuration: {result.error}. Adjust the knobs (usually num_gpu_blocks too small for max_model_len, or a prompt longer than max_model_len).
        </div>
      )}

      {snap && (
        <main className="flex-1 min-h-0 grid gap-3 p-3" style={{ gridTemplateColumns: showControls ? '270px 210px minmax(480px, 1fr) 330px' : '210px minmax(480px, 1fr) 330px' }}>
          {showControls && (
            <div className="overflow-y-auto scroll-thin min-h-0 flex flex-col gap-3">
              <Controls
                scenario={scenario}
                presetName={presetName}
                onPreset={(n) => {
                  setPresetName(n)
                  setScenario(preset(n))
                }}
                onChange={(s) => {
                  setPresetName('custom')
                  setScenario(s)
                }}
              />
              <Drift />
            </div>
          )}
          <div className="overflow-y-auto scroll-thin min-h-0 flex flex-col gap-3">
            <PipelineRail snap={snap} liveComponents={liveComponents} />
          </div>
          <div className="overflow-y-auto scroll-thin min-h-0 flex flex-col gap-3 [&>*]:shrink-0">
            <SchedulerView snap={snap} order={order} policy={scenario.config.policy} />
            <KVView snap={snap} order={order} hot={hotBlocks} />
            <WorkerView snap={snap} order={order} />
            {snap.batch_queue !== null && (
              <div className="panel p-3">
                <div className="flex items-baseline gap-2">
                  <span className="panel-title">batch_queue</span>
                  <span className="hint text-xs">in-flight steps under async scheduling; appendleft on schedule, pop when a result is needed</span>
                </div>
                <div className="flex gap-1 mt-1">
                  {snap.batch_queue.length === 0 && <span className="hint text-xs">empty</span>}
                  {snap.batch_queue.map((s) => (
                    <span key={s} className="chip mono">step {s} in flight</span>
                  ))}
                </div>
              </div>
            )}
            <div className="panel p-3">
              <div className="panel-title mb-1">Outputs</div>
              {Object.values(snap.outputs).length === 0 && <div className="hint text-xs">nothing generated yet</div>}
              {Object.values(snap.outputs).map((o) => (
                <div key={o.request_id} className="flex gap-2 items-baseline text-[12px]">
                  <span className="mono w-8 shrink-0">{o.external_req_id}</span>
                  <span>{o.text || <span className="hint">…</span>}</span>
                  {o.finished && <span className="hint text-[11px]">finished ({o.finish_reason})</span>}
                </div>
              ))}
            </div>
          </div>
          <div className="min-h-0">
            <Timeline events={events} from={snap.events_from} to={snap.events_to} selected={selected} onSelect={setSelected} />
          </div>
        </main>
      )}

      {snap && (
        <footer className="flex items-center gap-3 px-4 py-2" style={{ borderTop: '1px solid var(--rule)', background: 'var(--panel)' }}>
          <button className="tbtn" onClick={() => setCursor(0)} title="first">⇤</button>
          <button className="tbtn" onClick={() => setCursor((c) => stepBoundary(snapshots, c, -1))} title="previous step (↑)">⇠ step</button>
          <button className="tbtn" onClick={() => setCursor((c) => Math.max(0, c - 1))} title="previous phase (←)">←</button>
          <button className="tbtn" data-active={playing} onClick={() => setPlaying((p) => !p)} title="play/pause (space)">{playing ? 'pause' : 'play'}</button>
          <button className="tbtn" onClick={() => setCursor((c) => Math.min(snapshots.length - 1, c + 1))} title="next phase (→)">→</button>
          <button className="tbtn" onClick={() => setCursor((c) => stepBoundary(snapshots, c, 1))} title="next step (↓)">step ⇢</button>
          <label className="flex items-center gap-1 hint text-xs">
            speed
            <input type="range" min={0.5} max={4} step={0.5} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} style={{ width: 70 }} />
          </label>
          <input type="range" min={0} max={Math.max(0, snapshots.length - 1)} value={idx} onChange={(e) => setCursor(Number(e.target.value))} className="flex-1" aria-label="position" />
          <span className="mono text-xs whitespace-nowrap" title="← → phase · ↑ ↓ step · space play">step {snap.step} / {stepsTotal - 1}</span>
          <div className="flex gap-1">
            {PHASES.map((p) => (
              <span key={p} className="chip phase-chip mono text-[10px]" data-active={snap.phase === p} data-done={snap.phase !== p && phasesDoneThisStep.has(p)}>
                {PHASE_TITLE[p]}
              </span>
            ))}
          </div>
        </footer>
      )}
    </div>
  )
}
