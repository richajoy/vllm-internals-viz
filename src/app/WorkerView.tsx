import type { Snapshot } from '../sim/simulation'
import { requestColor } from './model'

export function WorkerView({ snap, order }: { snap: Snapshot; order: string[] }) {
  const { worker } = snap
  const p = worker.prepared
  const live = snap.phase === 'execute_model' || snap.phase === 'sample_tokens' || snap.phase === 'grammar_bitmask'
  const reqOf = (flatIdx: number): string | null => {
    if (!p) return null
    const r = p.spans.findIndex(([s, e]) => flatIdx >= s && flatIdx < e)
    return r === -1 ? null : p.req_ids[r]
  }
  const rows: { name: string; hint: string; values: (number | string)[] }[] = p
    ? [
        { name: 'input_ids', hint: 'all requests flattened into one super-sequence', values: p.input_ids.map((t) => snap.vocab[t] ?? t) },
        { name: 'positions', hint: 'num_computed_tokens + query offset, per request', values: p.positions },
        { name: 'slot_mapping', hint: `block_table[pos // ${snap.kv.block_size}] * ${snap.kv.block_size} + pos % ${snap.kv.block_size}`, values: p.slot_mapping },
      ]
    : []

  return (
    <div className="panel p-3 flex flex-col gap-3" style={{ opacity: live || p ? 1 : 0.7 }}>
      <div className="flex items-baseline justify-between">
        <div className="panel-title">Worker · GPUModelRunner</div>
        <div className="hint text-xs">
          input_batch rows [{worker.batch_rows.map((r) => (r ? snap.id_map[r] ?? r : '·')).join(' ')}]{worker.has_stashed_state ? ' · ExecuteModelState stashed' : ''}
        </div>
      </div>
      {!p && <div className="hint text-xs">No forward pass yet. execute_model builds the flattened batch; sample_tokens consumes the stashed state.</div>}
      {p && (
        <>
          <div className="overflow-x-auto scroll-thin">
            <table className="mono text-[11px] border-separate" style={{ borderSpacing: '2px 3px' }}>
              <tbody>
                <tr>
                  <td className="hint pr-2 whitespace-nowrap">request</td>
                  {p.input_ids.map((_, i) => {
                    const rid = reqOf(i)
                    const ext = rid ? snap.id_map[rid] ?? rid : ''
                    const isLogit = p.logits_indices.includes(i)
                    return (
                      <td key={i} className="text-center px-1 rounded-sm" style={{ background: rid ? requestColor(ext, order) : undefined, color: 'var(--on-accent)', outline: isLogit ? '2px solid var(--ink)' : undefined }} title={isLogit ? 'logits_indices: logits computed here' : ''}>
                        {ext}
                      </td>
                    )
                  })}
                </tr>
                {rows.map((row) => (
                  <tr key={row.name}>
                    <td className="hint pr-2 whitespace-nowrap" title={row.hint}>{row.name}</td>
                    {row.values.map((v, i) => (
                      <td key={i} className="text-center px-1 rounded-sm" style={{ background: 'var(--free)' }}>{v}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex gap-x-4 gap-y-1 flex-wrap text-[11px] mono hint">
            <span>query_start_loc [{p.query_start_loc.join(', ')}]</span>
            <span>seq_lens [{p.seq_lens.join(', ')}]</span>
            <span>num_actual_tokens {p.num_actual_tokens}</span>
            <span>logits_indices [{p.logits_indices.join(', ')}]</span>
          </div>
          <div className="hint text-[11px]">Outlined cells are where logits are gathered: the last query token per request, plus every draft position under speculative decoding. Attention metadata keeps each request inside its own span; there is no padding.</div>
        </>
      )}
      {worker.verification.length > 0 && (
        <section>
          <div className="flex items-baseline gap-2 mb-1">
            <span className="mono">RejectionSampler</span>
            <span className="hint text-xs">{worker.verification[0].mode === 'greedy' ? 'greedy: accept while draft == target argmax, then one bonus or recovered token' : 'stochastic acceptance'}</span>
          </div>
          {worker.verification.map((v) => (
            <div key={v.req_id} className="flex items-center gap-1 flex-wrap mono text-[11px]">
              <span className="chip" style={{ color: requestColor(snap.id_map[v.req_id] ?? v.req_id, order), borderColor: requestColor(snap.id_map[v.req_id] ?? v.req_id, order) }}>{snap.id_map[v.req_id] ?? v.req_id}</span>
              {v.draft_token_ids.map((d, i) => (
                <span key={i} className="chip" style={{ borderColor: v.accepted[i] ? 'var(--ok)' : v.accepted[i] === false ? 'var(--danger)' : 'var(--rule)', color: v.accepted[i] ? 'var(--ok)' : v.accepted[i] === false ? 'var(--danger)' : 'var(--faint)' }} title={`draft ${d} vs target ${v.target_argmax[i]}`}>
                  {snap.vocab[d] ?? d}
                  <span className="hint">/{snap.vocab[v.target_argmax[i]] ?? v.target_argmax[i]}</span>
                </span>
              ))}
              <span className="hint">→ sampled [{v.sampled.map((t) => snap.vocab[t] ?? t).join(' ')}]</span>
            </div>
          ))}
        </section>
      )}
      {Object.keys(worker.ngram).length > 0 && (
        <section>
          <div className="flex items-baseline gap-2 mb-1">
            <span className="mono">NgramProposer</span>
            <span className="hint text-xs">longest n-gram in [min,max] matching the suffix; earliest occurrence wins; k tokens after it become the draft</span>
          </div>
          {Object.entries(worker.ngram).map(([rid, m]) => (
            <div key={rid} className="mono text-[11px]">
              {snap.id_map[rid] ?? rid}: {m.ngram_len > 0 ? `${m.ngram_len}-gram at ${m.match_start} → draft [${m.drafts.map((t) => snap.vocab[t] ?? t).join(' ')}]` : 'no match, no draft'}
            </div>
          ))}
        </section>
      )}
    </div>
  )
}
