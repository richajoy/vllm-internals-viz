import { LayoutGroup, motion } from 'motion/react'
import type { BlockSnapshot, Snapshot } from '../sim/simulation'
import { requestColor } from './model'

interface Props {
  snap: Snapshot
  order: string[]
  hot: Set<number>
}

function Block({ b, color, hot, showHash }: { b: BlockSnapshot; color?: string; hot: boolean; showHash: boolean }) {
  const free = b.ref_cnt === 0 && !b.is_null
  return (
    <motion.div
      layoutId={`blk-${b.block_id}`}
      layout
      transition={{ type: 'spring', stiffness: 420, damping: 34 }}
      className="block-card mono"
      data-free={free}
      data-null={b.is_null}
      data-hot={hot}
      style={color && !free ? { borderColor: color, borderWidth: 1.5 } : undefined}
      title={`block_id=${b.block_id} ref_cnt=${b.ref_cnt}${b.block_hash ? ` hash=${b.block_hash}` : ''}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span style={{ fontVariationSettings: "'MONO' 1, 'wght' 650" }}>{b.is_null ? 'null' : b.block_id}</span>
        {!b.is_null && <span className="text-[10px] hint">ref {b.ref_cnt}</span>}
      </div>
      {showHash && (
        <div className="text-[10px]" style={{ color: b.block_hash ? 'var(--ink)' : 'var(--faint)' }}>
          {b.block_hash ? `#${b.block_hash.split(':')[0].slice(0, 6)}` : 'no hash'}
        </div>
      )}
    </motion.div>
  )
}

export function KVView({ snap, order, hot }: Props) {
  const { kv } = snap
  const byId = new Map(kv.blocks.map((b) => [b.block_id, b]))
  const showHash = snap.kv.cached_hash_entries.length > 0 || kv.blocks.some((b) => b.block_hash)
  const reqRows = Object.entries(kv.req_to_blocks)
  const inUse = kv.num_gpu_blocks - 1 - kv.num_free_blocks

  return (
    <LayoutGroup id="kv">
      <div className="panel p-3 flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <div className="panel-title">KV cache manager</div>
          <div className="hint text-xs">
            {inUse}/{kv.num_gpu_blocks - 1} blocks in use · block_size {kv.block_size} · {kv.cached_hash_entries.length} cached hash{kv.cached_hash_entries.length === 1 ? '' : 'es'}
          </div>
        </div>

        <section>
          <div className="flex items-baseline gap-2 mb-1">
            <span className="mono">free_block_queue</span>
            <span className="hint text-xs">doubly linked list, head = next evicted; fake head/tail sentinels never popped</span>
          </div>
          <div className="flex items-center gap-1 flex-wrap">
            <span className="block-card mono" data-null="true" style={{ minWidth: 0 }}>head</span>
            {kv.free_queue_order.map((id) => (
              <div key={id} className="flex items-center gap-1">
                <span className="hint text-[10px]">⇄</span>
                <Block b={byId.get(id) as BlockSnapshot} hot={hot.has(id)} showHash={showHash} />
              </div>
            ))}
            <span className="hint text-[10px]">⇄</span>
            <span className="block-card mono" data-null="true" style={{ minWidth: 0 }}>tail</span>
          </div>
        </section>

        <section>
          <div className="flex items-baseline gap-2 mb-1">
            <span className="mono">req_to_blocks</span>
            <span className="hint text-xs">per-request block table (SingleTypeKVCacheManager); ref_cnt &gt; 1 means a shared prefix</span>
          </div>
          {reqRows.length === 0 && <div className="hint text-xs">no request holds blocks</div>}
          <div className="flex flex-col gap-1">
            {reqRows.map(([rid, ids]) => {
              const ext = snap.id_map[rid] ?? rid
              const color = requestColor(ext, order)
              return (
                <div key={rid} className="flex items-center gap-1 flex-wrap">
                  <span className="chip mono" style={{ borderColor: color, color }}>{ext}</span>
                  {ids.map((id, i) => (
                    <div key={`${rid}-${i}`} className="flex items-center gap-1">
                      <Block b={byId.get(id) as BlockSnapshot} color={color} hot={hot.has(id)} showHash={showHash} />
                    </div>
                  ))}
                </div>
              )
            })}
          </div>
        </section>

        {showHash && (
          <section>
            <div className="flex items-baseline gap-2 mb-1">
              <span className="mono">cached_block_hash_to_block</span>
              <span className="hint text-xs">BlockHashToBlockMap; entries stay until the block is re-allocated (lazy eviction)</span>
            </div>
            <div className="flex gap-1 flex-wrap">
              {kv.cached_hash_entries.length === 0 && <span className="hint text-xs">empty</span>}
              {kv.cached_hash_entries.map((e) => (
                <span key={e.hash} className="chip mono">
                  #{e.hash.split(':')[0].slice(0, 6)} → {e.block_ids.join(',')}
                </span>
              ))}
            </div>
          </section>
        )}

        <section>
          <div className="flex items-baseline gap-2 mb-1">
            <span className="mono">paged KV memory (GPU)</span>
            <span className="hint text-xs">slot = block_id × {kv.block_size} + pos % {kv.block_size}; reshape_and_cache_flash writes here</span>
          </div>
          <div className="flex gap-1 flex-wrap">
            {kv.blocks.map((b) => {
              return (
                <div key={b.block_id} className="flex flex-col items-center gap-[2px]" title={`block ${b.block_id}${b.owners.length ? ` owned by ${b.owners.join(', ')}` : ' (free)'}`}>
                  <div className="flex gap-[2px] p-[2px] rounded" style={{ outline: hot.has(b.block_id) ? '2px solid var(--accent)' : undefined, background: b.is_null ? 'transparent' : undefined }}>
                    {b.slots.map((t, i) => {
                      const writer = b.writers[i]
                      const stale = t !== null && (writer === null || !b.owners.includes(writer))
                      return (
                        <div
                          key={i}
                          className="slot"
                          title={t === null ? 'never written' : `token ${t} (${snap.vocab[t] ?? '?'}) written by ${writer}${stale ? ' (stale)' : ''}`}
                          style={{
                            background: t === null ? 'var(--panel)' : writer ? requestColor(writer, order) : 'var(--rule-strong)',
                            opacity: stale ? 0.3 : 1,
                            borderStyle: b.is_null ? 'dashed' : 'solid',
                          }}
                        />
                      )
                    })}
                  </div>
                  <span className="text-[10px] mono hint">{b.is_null ? 'null' : b.block_id}</span>
                </div>
              )
            })}
          </div>
          <div className="hint text-[11px] mt-1">Slots keep the colour of the request that wrote them; faded slots are stale K/V left by a previous owner. Freeing never zeroes memory, reuse overwrites it.</div>
        </section>
      </div>
    </LayoutGroup>
  )
}
