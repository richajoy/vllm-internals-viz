// Port of vllm/v1/core/kv_cache_coordinator.py: UnitaryKVCacheCoordinator
// (one full-attention group) and KVCacheCoordinatorNoPrefixCache.

import { BlockPool } from './block_pool'
import { type EventLog, NULL_LOG } from './events'
import type { BlockHash, KVCacheBlock } from './kv_cache_utils'
import type { Request } from './request'
import { FullAttentionManager } from './single_type_kv_cache_manager'

export class UnitaryKVCacheCoordinator {
  block_pool: BlockPool
  single_type_managers: [FullAttentionManager]
  block_size: number
  enable_caching: boolean

  constructor(args: { num_gpu_blocks: number; block_size: number; enable_caching: boolean; log?: EventLog }) {
    const log = args.log ?? NULL_LOG
    this.block_size = args.block_size
    this.enable_caching = args.enable_caching
    // hash_block_size == block_size for the unitary coordinator.
    this.block_pool = new BlockPool(args.num_gpu_blocks, args.enable_caching, args.block_size, log)
    this.single_type_managers = [new FullAttentionManager(args.block_size, this.block_pool, args.enable_caching, 0, log)]
  }

  get_num_blocks_to_allocate(request_id: string, num_tokens: number, new_computed_blocks: readonly KVCacheBlock[][]): number {
    let n = 0
    this.single_type_managers.forEach((m, i) => {
      n += m.get_num_blocks_to_allocate(request_id, num_tokens, new_computed_blocks[i] ?? [])
    })
    return n
  }

  allocate_new_computed_blocks(request_id: string, new_computed_blocks: readonly KVCacheBlock[][]): void {
    if (this.single_type_managers.some((m) => m.num_cached_block.has(request_id))) {
      if (new_computed_blocks.some((b) => b.length !== 0)) throw new Error('running request with computed blocks')
      return
    }
    // Two-phase: touch every group's hits first, then external allocation.
    this.single_type_managers.forEach((m, i) => m.add_local_computed_blocks(request_id, new_computed_blocks[i] ?? []))
  }

  allocate_new_blocks(request_id: string, num_tokens: number): KVCacheBlock[][] {
    return this.single_type_managers.map((m) => m.allocate_new_blocks(request_id, num_tokens))
  }

  cache_blocks(request: Request, num_computed_tokens: number): void {
    for (const m of this.single_type_managers) m.cache_blocks(request, num_computed_tokens)
  }

  free(request_id: string): void {
    for (const m of this.single_type_managers) m.free(request_id)
  }

  pop_blocks_for_free(request_id: string): KVCacheBlock[] {
    return this.single_type_managers.flatMap((m) => m.pop_blocks_for_free(request_id))
  }

  get_blocks(request_id: string): KVCacheBlock[][] {
    return this.single_type_managers.map((m) => m.req_to_blocks.get(request_id) ?? [])
  }

  get_num_common_prefix_blocks(running_request_id: string): number[] {
    return this.single_type_managers.map((m) => m.get_num_common_prefix_blocks(running_request_id))
  }

  /** Returns (hit blocks per group, hit length, num_uncached=0 for a single group). */
  find_longest_cache_hit(block_hashes: readonly BlockHash[], max_cache_hit_length: number): [KVCacheBlock[][], number, number] {
    if (!this.enable_caching) return [this.single_type_managers.map(() => []), 0, 0]
    const [hit_blocks, hit_length] = this.single_type_managers[0].find_longest_cache_hit(block_hashes, max_cache_hit_length, [0])
    return [hit_blocks, hit_length, 0]
  }
}
