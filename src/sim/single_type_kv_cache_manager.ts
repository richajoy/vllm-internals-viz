// Port of vllm/v1/core/single_type_kv_cache_manager.py restricted to
// FullAttentionManager (no sliding window, no CoW partial hits, no encoder).

import type { BlockPool } from './block_pool'
import { type EventLog, NULL_LOG } from './events'
import { type BlockHash, type KVCacheBlock, cdiv } from './kv_cache_utils'
import type { Request } from './request'
import { REF } from './source_refs'

export class FullAttentionManager {
  block_size: number
  block_pool: BlockPool
  enable_caching: boolean
  kv_cache_group_id: number
  /** request_id -> blocks in allocation order (the "block table" on the CPU side). */
  req_to_blocks = new Map<string, KVCacheBlock[]>()
  /** request_id -> number of blocks already cached; only tracked for running requests. */
  num_cached_block = new Map<string, number>()
  log: EventLog

  constructor(block_size: number, block_pool: BlockPool, enable_caching: boolean, kv_cache_group_id = 0, log: EventLog = NULL_LOG) {
    this.block_size = block_size
    this.block_pool = block_pool
    this.enable_caching = enable_caching
    this.kv_cache_group_id = kv_cache_group_id
    this.log = log
  }

  private blocks_for(request_id: string): KVCacheBlock[] {
    let blocks = this.req_to_blocks.get(request_id)
    if (!blocks) {
      blocks = []
      this.req_to_blocks.set(request_id, blocks)
    }
    return blocks
  }

  static _get_num_evictable_blocks(blocks: readonly KVCacheBlock[]): number {
    return blocks.filter((b) => b.ref_cnt === 0 && !b.is_null).length
  }

  get_num_blocks_to_allocate(
    request_id: string,
    num_tokens: number,
    new_computed_blocks: readonly KVCacheBlock[],
  ): number {
    const num_required_blocks = cdiv(num_tokens, this.block_size)
    const num_req_blocks = this.req_to_blocks.get(request_id)?.length ?? 0
    if (this.num_cached_block.has(request_id)) {
      // Running request: no new prefix-cache hits possible. Spec-decode may
      // have allocated for drafts later rejected, hence the max(.., 0).
      if (new_computed_blocks.length !== 0) throw new Error('running request with new computed blocks')
      return Math.max(num_required_blocks - num_req_blocks, 0)
    }
    const num_local_computed_blocks = new_computed_blocks.length + num_req_blocks
    const num_new_blocks = Math.max(num_required_blocks - num_local_computed_blocks, 0)
    // Cache-hit blocks sitting in the free queue (ref_cnt == 0) leave it when
    // touched, so they count against free capacity too.
    const num_evictable_blocks = FullAttentionManager._get_num_evictable_blocks(new_computed_blocks)
    return num_new_blocks + num_evictable_blocks
  }

  add_local_computed_blocks(request_id: string, new_computed_blocks: readonly KVCacheBlock[]): void {
    const req_blocks = this.blocks_for(request_id)
    if (req_blocks.length !== 0) throw new Error('add_local_computed_blocks on a request that already has blocks')
    if (this.enable_caching) {
      this.block_pool.touch(new_computed_blocks)
    } else if (new_computed_blocks.length > 0) {
      throw new Error('Computed blocks should be empty when prefix caching is disabled')
    }
    req_blocks.push(...new_computed_blocks)
    this.num_cached_block.set(request_id, req_blocks.length)
    if (new_computed_blocks.length > 0) {
      this.log.emit(
        'SingleTypeKVCacheManager',
        'add_local_computed_blocks',
        `prefix-cache hit: req_to_blocks[${request_id}] = [${new_computed_blocks.map((b) => b.block_id).join(', ')}] (touched)`,
        { request_id, block_ids: new_computed_blocks.map((b) => b.block_id) },
        REF.SingleTypeKVCacheManager_add_local_computed_blocks,
      )
    }
  }

  allocate_new_blocks(request_id: string, num_tokens: number): KVCacheBlock[] {
    const req_blocks = this.blocks_for(request_id)
    const num_required_blocks = cdiv(num_tokens, this.block_size)
    const num_new_blocks = num_required_blocks - req_blocks.length
    if (num_new_blocks <= 0) return []
    const new_blocks = this.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.push(...new_blocks)
    this.log.emit(
      'SingleTypeKVCacheManager',
      'allocate_new_blocks',
      `req_to_blocks[${request_id}] += [${new_blocks.map((b) => b.block_id).join(', ')}] (need ${num_required_blocks} blocks for ${num_tokens} tokens)`,
      { request_id, block_ids: new_blocks.map((b) => b.block_id), num_required_blocks, num_tokens },
      REF.SingleTypeKVCacheManager_allocate_new_blocks,
    )
    return new_blocks
  }

  cache_blocks(request: Request, num_tokens: number): void {
    const num_cached_blocks = this.num_cached_block.get(request.request_id) ?? 0
    const num_full_blocks = Math.floor(num_tokens / this.block_size)
    if (num_cached_blocks >= num_full_blocks) return
    this.block_pool.cache_full_blocks(
      request,
      this.blocks_for(request.request_id),
      num_cached_blocks,
      num_full_blocks,
      this.block_size,
      this.kv_cache_group_id,
    )
    this.num_cached_block.set(request.request_id, num_full_blocks)
  }

  pop_blocks_for_free(request_id: string): KVCacheBlock[] {
    const req_blocks = this.req_to_blocks.get(request_id) ?? []
    this.req_to_blocks.delete(request_id)
    this.num_cached_block.delete(request_id)
    return req_blocks
  }

  /** Free in reverse order so tail blocks are evicted first. */
  free(request_id: string): void {
    const blocks = this.pop_blocks_for_free(request_id)
    this.log.emit(
      'SingleTypeKVCacheManager',
      'free',
      `free(${request_id}): return [${blocks.map((b) => b.block_id).join(', ')}] reversed to block pool`,
      { request_id, block_ids: blocks.map((b) => b.block_id) },
      REF.SingleTypeKVCacheManager_free,
    )
    this.block_pool.free_blocks(blocks.slice().reverse())
  }

  get_num_common_prefix_blocks(running_request_id: string): number {
    const blocks = this.req_to_blocks.get(running_request_id) ?? []
    let n = 0
    for (const block of blocks) {
      if (block.ref_cnt === this.req_to_blocks.size) n += 1
      else break
    }
    return n
  }

  /** Longest run of cached full blocks from the start (chained hashes). */
  find_longest_cache_hit(
    block_hashes: readonly BlockHash[],
    max_length: number,
    kv_cache_group_ids: number[],
  ): [KVCacheBlock[][], number] {
    const computed_blocks: KVCacheBlock[][] = kv_cache_group_ids.map(() => [])
    const limit = Math.floor(max_length / this.block_size)
    for (let i = 0; i < Math.min(limit, block_hashes.length); i++) {
      const cached = this.block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids)
      if (!cached) break
      cached.forEach((b, g) => computed_blocks[g].push(b))
    }
    const hit_length = computed_blocks[0].length * this.block_size
    this.log.emit(
      'SingleTypeKVCacheManager',
      'find_longest_cache_hit',
      `find_longest_cache_hit: ${computed_blocks[0].length} block(s) hit (${hit_length} tokens, max ${max_length}) -> [${computed_blocks[0].map((b) => b.block_id).join(', ')}]`,
      { block_ids: computed_blocks[0].map((b) => b.block_id), hit_length, max_length, hashes_probed: Math.min(limit, block_hashes.length) },
      REF.FullAttentionManager_find_longest_cache_hit,
    )
    return [computed_blocks, hit_length]
  }
}
