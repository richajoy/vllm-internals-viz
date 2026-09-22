// Port of vllm/v1/core/block_pool.py (BlockHashToBlockMap, BlockPool) without
// KV events, metrics collectors, partial-block caching or copy-on-write.

import { type EventLog, NULL_LOG } from './events'
import {
  type BlockHash,
  type BlockHashWithGroupId,
  FreeKVCacheBlockQueue,
  KVCacheBlock,
  make_block_hash_with_group_id,
} from './kv_cache_utils'
import type { Request } from './request'
import { REF } from './source_refs'

/**
 * hash -> block, or hash -> {block_id: block} only on collision; vLLM does
 * this to avoid allocating an inner dict per entry.
 */
export class BlockHashToBlockMap {
  private _cache = new Map<BlockHashWithGroupId, KVCacheBlock | Map<number, KVCacheBlock>>()

  get_one_block(key: BlockHashWithGroupId): KVCacheBlock | null {
    const blocks = this._cache.get(key)
    if (blocks === undefined) return null
    if (blocks instanceof KVCacheBlock) return blocks
    return blocks.values().next().value ?? null
  }

  contain(key: BlockHashWithGroupId, block_id: number): boolean {
    const blocks = this._cache.get(key)
    if (blocks === undefined) return false
    if (blocks instanceof KVCacheBlock) return blocks.block_id === block_id
    return blocks.has(block_id)
  }

  insert(key: BlockHashWithGroupId, block: KVCacheBlock): void {
    const blocks = this._cache.get(key)
    if (blocks === undefined) {
      this._cache.set(key, block)
    } else if (blocks instanceof KVCacheBlock) {
      this._cache.set(
        key,
        new Map([
          [blocks.block_id, blocks],
          [block.block_id, block],
        ]),
      )
    } else {
      blocks.set(block.block_id, block)
    }
  }

  pop(key: BlockHashWithGroupId, block_id: number): KVCacheBlock | null {
    const blocks = this._cache.get(key)
    if (blocks === undefined) return null
    this._cache.delete(key)
    if (blocks instanceof KVCacheBlock) {
      if (blocks.block_id === block_id) return blocks
      this._cache.set(key, blocks)
      return null
    }
    const block = blocks.get(block_id) ?? null
    blocks.delete(block_id)
    if (blocks.size > 0) this._cache.set(key, blocks)
    return block
  }

  get size(): number {
    return this._cache.size
  }

  keys(): BlockHashWithGroupId[] {
    return [...this._cache.keys()]
  }

  entries(): Array<[BlockHashWithGroupId, number[]]> {
    return [...this._cache.entries()].map(([k, v]) => [
      k,
      v instanceof KVCacheBlock ? [v.block_id] : [...v.keys()],
    ])
  }
}

export class BlockPool {
  num_gpu_blocks: number
  enable_caching: boolean
  hash_block_size: number
  blocks: KVCacheBlock[]
  free_block_queue: FreeKVCacheBlockQueue
  cached_block_hash_to_block = new BlockHashToBlockMap()
  cached_block_hashes_by_block = new Map<number, Set<BlockHashWithGroupId>>()
  null_block: KVCacheBlock
  log: EventLog

  constructor(num_gpu_blocks: number, enable_caching: boolean, hash_block_size: number, log: EventLog = NULL_LOG) {
    if (!(Number.isInteger(num_gpu_blocks) && num_gpu_blocks > 0)) {
      throw new Error('num_gpu_blocks must be a positive integer')
    }
    this.num_gpu_blocks = num_gpu_blocks
    this.enable_caching = enable_caching
    this.hash_block_size = hash_block_size
    this.log = log
    this.blocks = Array.from({ length: num_gpu_blocks }, (_, i) => new KVCacheBlock(i))
    this.free_block_queue = new FreeKVCacheBlockQueue(this.blocks)
    // Block 0 is reserved as the null placeholder; its ref_cnt is not maintained.
    this.null_block = this.free_block_queue.popleft()
    this.null_block.is_null = true
    log.emit('BlockPool', 'init', `BlockPool: ${num_gpu_blocks} blocks, block 0 reserved as null_block`, {
      num_gpu_blocks,
      enable_caching,
    }, REF.BlockPool_null_block)
  }

  get_cached_block(block_hash: BlockHash, kv_cache_group_ids: number[]): KVCacheBlock[] | null {
    const cached: KVCacheBlock[] = []
    for (const group_id of kv_cache_group_ids) {
      const block = this.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(block_hash, group_id))
      if (!block) return null
      cached.push(block)
    }
    return cached
  }

  cache_full_blocks(
    request: Request,
    blocks: KVCacheBlock[],
    num_cached_blocks: number,
    num_full_blocks: number,
    block_size: number,
    kv_cache_group_id: number,
  ): void {
    if (num_cached_blocks >= num_full_blocks) return
    const new_full_blocks = blocks.slice(num_cached_blocks, num_full_blocks)
    // hash_block_size == block_size in the unitary case, so no resolve step.
    const new_block_hashes = request.block_hashes.slice(num_cached_blocks)
    new_full_blocks.forEach((blk, i) => {
      if (blk.is_null) return
      const block_hash = new_block_hashes[i]
      if (block_hash === undefined) throw new Error('missing block hash for full block')
      const num_hash_tokens = (num_cached_blocks + i + 1) * block_size
      const key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
      if (blk.block_hash !== null) {
        this._remove_cached_block_hashes(blk)
      }
      this._insert_block_hash(key, blk, num_hash_tokens)
      this.log.emit(
        'BlockPool',
        'cache_block',
        `cache_full_blocks: block ${blk.block_id} <- hash ${block_hash} (tokens ${num_hash_tokens - block_size}..${num_hash_tokens - 1} of ${request.request_id})`,
        { block_id: blk.block_id, block_hash, request_id: request.request_id, num_hash_tokens },
        REF.BlockPool_cache_full_blocks,
      )
    })
  }

  private _insert_block_hash(key: BlockHashWithGroupId, block: KVCacheBlock, num_tokens: number | null): void {
    if (block.block_hash === key) return
    if (this.cached_block_hash_to_block.contain(key, block.block_id)) return
    if (block.block_hash === null) {
      block.set_block_hash(key, num_tokens)
    } else {
      let set = this.cached_block_hashes_by_block.get(block.block_id)
      if (!set) {
        set = new Set()
        this.cached_block_hashes_by_block.set(block.block_id, set)
      }
      set.add(key)
    }
    this.cached_block_hash_to_block.insert(key, block)
  }

  private _remove_cached_block_hashes(block: KVCacheBlock): BlockHashWithGroupId[] {
    const block_hashes: BlockHashWithGroupId[] = []
    if (block.block_hash !== null) block_hashes.push(block.block_hash)
    const secondary = this.cached_block_hashes_by_block.get(block.block_id)
    if (secondary) {
      block_hashes.push(...secondary)
      this.cached_block_hashes_by_block.delete(block.block_id)
    }
    if (block_hashes.length === 0) return []
    const removed: BlockHashWithGroupId[] = []
    for (const h of block_hashes) {
      if (this.cached_block_hash_to_block.pop(h, block.block_id) !== null) removed.push(h)
    }
    block.reset_hash()
    return removed
  }

  get_new_blocks(num_blocks: number): KVCacheBlock[] {
    if (num_blocks > this.get_num_free_blocks()) {
      throw new Error(`Cannot get ${num_blocks} free blocks from the pool`)
    }
    const ret = this.free_block_queue.popleft_n(num_blocks)
    for (const block of ret) {
      if (this.enable_caching) this._maybe_evict_cached_block(block)
      if (block.ref_cnt !== 0) throw new Error(`block ${block.block_id} popped with ref_cnt ${block.ref_cnt}`)
      block.ref_cnt += 1
    }
    if (ret.length > 0) {
      this.log.emit(
        'BlockPool',
        'get_new_blocks',
        `get_new_blocks(${num_blocks}): popleft_n from free_block_queue -> [${ret.map((b) => b.block_id).join(', ')}]`,
        { block_ids: ret.map((b) => b.block_id), num_free_after: this.get_num_free_blocks() },
        REF.BlockPool_get_new_blocks,
      )
    }
    return ret
  }

  /** Lazy eviction: only when a hashed block is re-allocated from the free list. */
  private _maybe_evict_cached_block(block: KVCacheBlock): boolean {
    const had_hash = block.block_hash
    const evicted = this._remove_cached_block_hashes(block)
    if (evicted.length === 0) return false
    this.log.emit(
      'BlockPool',
      'evict',
      `lazy eviction: block ${block.block_id} re-allocated, hash ${had_hash} removed from cached_block_hash_to_block`,
      { block_id: block.block_id, block_hash: had_hash },
      REF.BlockPool_maybe_evict_cached_block,
    )
    return true
  }

  /** Prefix-cache hit: ref_cnt++ and unlink from the free list if it was there. */
  touch(blocks: readonly KVCacheBlock[]): void {
    for (const block of blocks) {
      const was_free = block.ref_cnt === 0 && !block.is_null
      if (was_free) this.free_block_queue.remove(block)
      block.ref_cnt += 1
      this.log.emit(
        'BlockPool',
        'touch',
        `touch: block ${block.block_id} ref_cnt -> ${block.ref_cnt}${was_free ? ' (O(1) remove from free_block_queue)' : ''}`,
        { block_id: block.block_id, ref_cnt: block.ref_cnt, removed_from_free_queue: was_free },
        REF.BlockPool_touch,
      )
    }
  }

  /**
   * Free in eviction-priority order (caller passes tail-first). Hashless
   * blocks go to the HEAD (evicted first), hashed blocks to the TAIL (LRU).
   */
  free_blocks(ordered_blocks: Iterable<KVCacheBlock>): void {
    const blocks_with_hash: KVCacheBlock[] = []
    const blocks_without_hash: KVCacheBlock[] = []
    for (const block of ordered_blocks) {
      block.ref_cnt -= 1
      if (block.ref_cnt === 0 && !block.is_null) {
        if (block.block_hash === null && this.enable_caching) blocks_without_hash.push(block)
        else blocks_with_hash.push(block)
      }
    }
    this.free_block_queue.prepend_n(blocks_without_hash)
    this.free_block_queue.append_n(blocks_with_hash)
    this.log.emit(
      'BlockPool',
      'free_blocks',
      `free_blocks: prepend_n(hashless=[${blocks_without_hash.map((b) => b.block_id).join(', ')}]) append_n(hashed=[${blocks_with_hash.map((b) => b.block_id).join(', ')}])`,
      {
        prepended: blocks_without_hash.map((b) => b.block_id),
        appended: blocks_with_hash.map((b) => b.block_id),
        num_free_after: this.get_num_free_blocks(),
      },
      REF.BlockPool_free_blocks,
    )
  }

  get_num_free_blocks(): number {
    return this.free_block_queue.num_free_blocks
  }

  get_usage(): number {
    const total = this.num_gpu_blocks - 1
    if (!total) return 0
    return 1 - this.get_num_free_blocks() / total
  }
}
