import { useEffect, useRef } from 'react'
import type { SimEvent } from '../sim/events'
import { githubUrl, VLLM_COMMIT_SHORT } from '../sim/source_refs'

interface Props {
  events: SimEvent[]
  /** Events belonging to the current snapshot are [from, to). */
  from: number
  to: number
  selected: number | null
  onSelect: (seq: number | null) => void
}

export function Timeline({ events, from, to, selected, onSelect }: Props) {
  const listRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-seq="${from}"]`)
    el?.scrollIntoView({ block: 'start' })
  }, [from])
  const sel = selected !== null ? events[selected] : null
  return (
    <div className="panel flex flex-col min-h-0" style={{ height: '100%' }}>
      <div className="p-3 pb-2 flex items-baseline justify-between">
        <div className="panel-title">Event log</div>
        <div className="hint text-xs">{events.length} events · vLLM @ {VLLM_COMMIT_SHORT}</div>
      </div>
      <div ref={listRef} className="overflow-y-auto scroll-thin flex-1 min-h-0" style={{ borderTop: '1px solid var(--rule)', borderBottom: '1px solid var(--rule)' }}>
        {events.map((e) => {
          const current = e.seq >= from && e.seq < to
          return (
            <button
              key={e.seq}
              data-seq={e.seq}
              className="event-row w-full text-left px-3 py-[3px] border-0 bg-transparent cursor-pointer"
              data-selected={selected === e.seq}
              style={{ opacity: current ? 1 : 0.5, font: 'inherit', color: 'inherit' }}
              onClick={() => onSelect(selected === e.seq ? null : e.seq)}
            >
              <div className="flex gap-2 items-baseline">
                <span className="hint text-[10px] mono w-8 shrink-0">s{e.step}</span>
                <span className="mono text-[11px] shrink-0" style={{ color: 'var(--accent)' }}>{e.component}</span>
                <span className="text-[12px] leading-snug break-words">{e.message}</span>
              </div>
            </button>
          )
        })}
      </div>
      <div className="p-3 text-[12px] leading-snug" style={{ minHeight: 96 }}>
        {!sel && <div className="hint">Select an event to see its payload and where it lives in vLLM.</div>}
        {sel && (
          <div className="flex flex-col gap-1">
            <div>
              <span className="mono">{sel.component}.{sel.kind}</span>
              <span className="hint"> · step {sel.step} · {sel.phase}</span>
            </div>
            {sel.ref && (
              <div className="mono text-[11px]">
                <a href={githubUrl(sel.ref)} target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>
                  {sel.ref.file}:{sel.ref.line}
                </a>
                {sel.ref.blogEraName && <span className="hint"> · the Aug-2025 blog called this “{sel.ref.blogEraName}”</span>}
              </div>
            )}
            {sel.payload && (
              <pre className="mono text-[11px] whitespace-pre-wrap break-all m-0 p-2 rounded" style={{ background: 'var(--free)', maxHeight: 160, overflow: 'auto' }}>
                {JSON.stringify(sel.payload, null, 1).replace(/\n\s+(?=[\d"\]}-])/g, ' ')}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
