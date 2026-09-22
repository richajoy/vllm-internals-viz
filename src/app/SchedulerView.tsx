import type { RequestSnapshot, Snapshot } from '../sim/simulation'
import { requestColor } from './model'

function Bar({ r }: { r: RequestSnapshot }) {
  // Token strip: prompt | outputs | placeholders | spec drafts, with the
  // computed boundary marked.
  const total = Math.max(r.num_tokens + r.num_output_placeholders + r.spec_token_ids.length, 1)
  const pct = (n: number) => `${(100 * n) / total}%`
  return (
    <div className="relative h-[10px] rounded-sm overflow-hidden" style={{ background: 'var(--free)' }} title={`num_computed_tokens=${r.num_computed_tokens} / num_tokens=${r.num_tokens} (+${r.num_output_placeholders} placeholders, +${r.spec_token_ids.length} spec)`}>
      <div className="absolute inset-y-0 left-0" style={{ width: pct(r.num_prompt_tokens), background: 'var(--rule-strong)' }} />
      <div className="absolute inset-y-0" style={{ left: pct(r.num_prompt_tokens), width: pct(r.num_output_tokens), background: 'var(--muted)' }} />
      <div className="absolute inset-y-0" style={{ left: pct(r.num_tokens), width: pct(r.num_output_placeholders), background: 'repeating-linear-gradient(90deg,var(--accent) 0 2px,transparent 2px 4px)' }} />
      <div className="absolute inset-y-0" style={{ left: pct(r.num_tokens + r.num_output_placeholders), width: pct(r.spec_token_ids.length), background: 'var(--req-2)', opacity: 0.6 }} />
      <div className="absolute inset-y-0 w-[2px]" style={{ left: `calc(${pct(Math.min(r.num_computed_tokens, total))} - 1px)`, background: 'var(--accent)' }} />
    </div>
  )
}

function Row({ r, ext, color, scheduled }: { r: RequestSnapshot; ext: string; color: string; scheduled?: number }) {
  return (
    <div className="grid gap-x-2 items-center" style={{ gridTemplateColumns: '3.5rem 1fr' }}>
      <span className="chip mono justify-center" style={{ borderColor: color, color }}>{ext}</span>
      <div className="flex flex-col gap-[3px]">
        <Bar r={r} />
        <div className="flex gap-x-3 flex-wrap text-[11px] mono hint leading-tight">
          <span>computed {r.num_computed_tokens}/{r.num_tokens}</span>
          {r.num_output_placeholders > 0 && <span style={{ color: 'var(--accent)' }}>placeholders {r.num_output_placeholders}</span>}
          {r.spec_token_ids.length > 0 && <span>spec [{r.spec_token_ids.join(',')}]</span>}
          {scheduled !== undefined && <span style={{ color: 'var(--ink)' }}>+{scheduled} this step</span>}
          {r.is_prefill_chunk && <span>prefill chunk</span>}
          {r.num_preemptions > 0 && <span style={{ color: 'var(--danger)' }}>preempted ×{r.num_preemptions}</span>}
          {r.priority !== 0 && <span>priority {r.priority}</span>}
          {r.use_structured_output && <span>grammar</span>}
        </div>
      </div>
    </div>
  )
}

export function SchedulerView({ snap, order, policy }: { snap: Snapshot; order: string[]; policy: 'fcfs' | 'priority' }) {
  const { scheduler } = snap
  const last = scheduler.last_output
  const q = (title: string, hint: string, ids: string[], extra?: (id: string) => number | undefined) => (
    <section>
      <div className="flex items-baseline gap-2 mb-1">
        <span className="mono">{title}</span>
        <span className="hint text-xs">{hint}</span>
        <span className="hint text-xs ml-auto">{ids.length}</span>
      </div>
      <div className="flex flex-col gap-1">
        {ids.length === 0 && <div className="hint text-xs">empty</div>}
        {ids.map((id) => {
          const r = scheduler.requests[id]
          if (!r) return null
          const ext = snap.id_map[id] ?? id
          return <Row key={id} r={r} ext={ext} color={requestColor(ext, order)} scheduled={extra?.(id)} />
        })}
      </div>
    </section>
  )
  const used = last ? last.total_num_scheduled_tokens : 0
  return (
    <div className="panel p-3 flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <div className="panel-title">Scheduler</div>
        <div className="hint text-xs">policy {policy} · max_num_seqs {scheduler.max_num_seqs}</div>
      </div>
      <div>
        <div className="flex items-baseline gap-2 mb-1">
          <span className="mono">token_budget</span>
          <span className="hint text-xs">max_num_batched_tokens per step; running requests are scheduled first, then waiting</span>
          <span className="mono ml-auto text-xs">{used}/{scheduler.token_budget}</span>
        </div>
        <div className="h-[8px] rounded-sm overflow-hidden" style={{ background: 'var(--free)' }}>
          <div className="h-full" style={{ width: `${Math.min(100, (100 * used) / scheduler.token_budget)}%`, background: 'var(--accent)' }} />
        </div>
      </div>
      {q('running', 'decode + in-progress prefills; preemption pops from the end', scheduler.running, (id) => last?.num_scheduled_tokens[id])}
      {q('waiting', policy === 'priority' ? 'heap ordered by (priority, arrival_time)' : 'FCFS deque; preempted requests go to the front', scheduler.waiting)}
      {scheduler.skipped_waiting.length > 0 && q('skipped_waiting', 'blocked on grammar compile / remote KVs', scheduler.skipped_waiting)}
      {last && (
        <div className="text-[11px] mono hint leading-snug">
          SchedulerOutput step {last.step}: new [{last.scheduled_new_reqs.map((i) => snap.id_map[i] ?? i).join(', ')}] cached [{last.scheduled_cached_reqs.map((i) => snap.id_map[i] ?? i).join(', ')}]
          {last.resumed_req_ids.length > 0 && <> resumed [{last.resumed_req_ids.map((i) => snap.id_map[i] ?? i).join(', ')}]</>}
          {last.preempted_req_ids.length > 0 && <span style={{ color: 'var(--danger)' }}> preempted [{last.preempted_req_ids.map((i) => snap.id_map[i] ?? i).join(', ')}]</span>}
          {last.finished_req_ids.length > 0 && <> finished [{last.finished_req_ids.map((i) => snap.id_map[i] ?? i).join(', ')}]</>}
        </div>
      )}
      <div className="text-[10px] hint flex gap-3 flex-wrap">
        <span><i className="inline-block w-3 h-2 align-middle" style={{ background: 'var(--rule-strong)' }} /> prompt</span>
        <span><i className="inline-block w-3 h-2 align-middle" style={{ background: 'var(--muted)' }} /> output</span>
        <span><i className="inline-block w-3 h-2 align-middle" style={{ background: 'repeating-linear-gradient(90deg,var(--accent) 0 2px,transparent 2px 4px)' }} /> placeholder</span>
        <span><i className="inline-block w-3 h-2 align-middle" style={{ background: 'var(--req-2)', opacity: 0.6 }} /> draft</span>
        <span><i className="inline-block w-[2px] h-2 align-middle" style={{ background: 'var(--accent)' }} /> num_computed_tokens</span>
      </div>
    </div>
  )
}
