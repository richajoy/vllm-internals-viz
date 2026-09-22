// Port of vllm/v1/core/kv_cache_manager.py (KVCacheBlocks, KVCacheManager)
// for the unitary full-attention case, without KV connectors.

import { type EventLog, NULL_LOG } from './events'
import { UnitaryKVCacheCoordinator } from './kv_cache_coordinator'
import type { KVCacheBlock } from './kv_cache_utils'
import { type Request, RequestStatus } from './request'
import { REF } from './source_refs'

/** blocks[group][idx]; tuple of per-group block lists in vLLM. */
export class KVCacheBlocks {
  blocks: readonly (readonly KVCacheBlock[])[]

  constructor(blocks: readonly (readonly KVCacheBlock[])[]) {
    this.blocks = blocks
  }

  add(other: KVCacheBlocks): KVCacheBlocks {
    return new KVCacheBlocks(this.blocks.map((b, i) => [...b, ...(other.blocks[i] ?? [])]))
  }

  get_block_ids(): number[][] {
    return this.blocks.map((group) => group.map((b) => b.block_id))
  }

  get is_empty(): boolean {
    return this.blocks.every((g) => g.length === 0)
  }
}

export interface KVCacheManagerConfig {
  num_gpu_blocks: number
  block_size: number
  max_model_len: number
  enable_caching: boolean
  /** Fraction of blocks kept free for WAITING/PREEMPTED admissions (default 0). */
  watermark?: number
}

export class KVCacheManager {
  block_size: number
  max_model_len: number
  enable_caching: boolean
  coordinator: UnitaryKVCacheCoordinator
  block_pool: import('./block_pool').BlockPool
  watermark_blocks: number
  empty_kv_cache_blocks: KVCacheBlocks
  log: EventLog

  constructor(cfg: KVCacheManagerConfig, log: EventLog = NULL_LOG) {
    this.block_size = cfg.block_size
    this.max_model_len = cfg.max_model_len
    this.enable_caching = cfg.enable_caching
    this.log = log
    this.coordinator = new UnitaryKVCacheCoordinator({
      num_gpu_blocks: cfg.num_gpu_blocks,
      block_size: cfg.block_size,
      enable_caching: cfg.enable_caching,
      log,
    })
    this.block_pool = this.coordinator.block_pool
    this.watermark_blocks = Math.floor((cfg.watermark ?? 0) * cfg.num_gpu_blocks)
    this.empty_kv_cache_blocks = new KVCacheBlocks([[]])
  }

  get usage(): number {
    return this.block_pool.get_usage()
  }

  prefix_cache_lookup_enabled(): boolean {
    return this.enable_caching
  }

  /**
   * Returns [computed blocks, num computed tokens, shared_prefix_boundary].
   * All-hit prompts must still recompute the last token to get logits.
   */
  get_computed_blocks(request: Request): [KVCacheBlocks, number, number] {
    if (!this.prefix_cache_lookup_enabled()) return [this.empty_kv_cache_blocks, 0, 0]
    const max_cache_hit_length = request.num_tokens - 1
    this.log.emit(
      'KVCacheManager',
      'get_computed_blocks',
      `get_computed_blocks(${request.request_id}): ${request.block_hashes.length} block hash(es), max hit ${max_cache_hit_length} tokens (last token must be recomputed for logits)`,
      { request_id: request.request_id, num_block_hashes: request.block_hashes.length, max_cache_hit_length },
      REF.KVCacheManager_recompute_last_token,
    )
    const [computed_blocks, num_new_computed_tokens] = this.coordinator.find_longest_cache_hit(
      request.block_hashes,
      max_cache_hit_length,
    )
    return [new KVCacheBlocks(computed_blocks), num_new_computed_tokens, 0]
  }

  /**
   * Blocks layout:
   *   | comp | new_comp | new | lookahead |
   * Returns the newly allocated blocks, or null when the pool cannot satisfy
   * the request (the scheduler then preempts or stops scheduling).
   */
  allocate_slots(
    request: Request,
    num_new_tokens: number,
    opts: {
      num_new_computed_tokens?: number
      new_computed_blocks?: KVCacheBlocks | null
      num_lookahead_tokens?: number
      has_scheduled_reqs?: boolean
      /** Admission gate: the whole prompt must fit, not just this chunk. */
      full_sequence_must_fit?: boolean
    } = {},
  ): KVCacheBlocks | null {
    if (num_new_tokens === 0) throw new Error('num_new_tokens must be greater than 0')
    const num_new_computed_tokens = opts.num_new_computed_tokens ?? 0
    const num_lookahead_tokens = opts.num_lookahead_tokens ?? 0
    const has_scheduled_reqs = opts.has_scheduled_reqs ?? true
    const new_computed_block_list = opts.new_computed_blocks?.blocks ?? this.empty_kv_cache_blocks.blocks

    const num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
    const total_computed_tokens = Math.min(num_local_computed_tokens, this.max_model_len)

    let watermark_blocks = 0
    if (has_scheduled_reqs && (request.status === RequestStatus.WAITING || request.status === RequestStatus.PREEMPTED)) {
      watermark_blocks = this.watermark_blocks
    }

    if (opts.full_sequence_must_fit) {
      const full_num_tokens = Math.min(request.num_tokens, this.max_model_len)
      const full_blocks = this.coordinator.get_num_blocks_to_allocate(
        request.request_id,
        full_num_tokens,
        new_computed_block_list as KVCacheBlock[][],
      )
      if (full_blocks + watermark_blocks > this.block_pool.get_num_free_blocks()) {
        this.log.emit(
          'KVCacheManager',
          'allocate_slots_fail',
          `allocate_slots(${request.request_id}): full sequence needs ${full_blocks} block(s) (scheduler_reserve_full_isl) > ${this.block_pool.get_num_free_blocks()} free -> None`,
          { request_id: request.request_id, required_blocks: full_blocks, available_blocks: this.block_pool.get_num_free_blocks(), gate: 'full_sequence_must_fit' },
          REF.KVCacheManager_allocate_slots,
        )
        return null
      }
    }

    const num_tokens_main_model = total_computed_tokens + num_new_tokens
    const num_tokens_need_slot = Math.min(num_tokens_main_model + num_lookahead_tokens, this.max_model_len)

    const num_blocks_to_allocate = this.coordinator.get_num_blocks_to_allocate(
      request.request_id,
      num_tokens_need_slot,
      new_computed_block_list as KVCacheBlock[][],
    )

    const available_blocks = this.block_pool.get_num_free_blocks()
    const required_blocks = num_blocks_to_allocate + watermark_blocks
    this.log.emit(
      'KVCacheManager',
      'allocate_slots',
      `allocate_slots(${request.request_id}, new=${num_new_tokens}${num_new_computed_tokens ? `, cached=${num_new_computed_tokens}` : ''}${num_lookahead_tokens ? `, lookahead=${num_lookahead_tokens}` : ''}): need slots for ${num_tokens_need_slot} tokens -> ${num_blocks_to_allocate} new block(s), ${available_blocks} free`,
      {
        request_id: request.request_id,
        num_new_tokens,
        num_new_computed_tokens,
        num_lookahead_tokens,
        num_tokens_need_slot,
        num_blocks_to_allocate,
        watermark_blocks,
        available_blocks,
      },
      REF.KVCacheManager_allocate_slots,
    )
    if (required_blocks > available_blocks) {
      this.log.emit(
        'KVCacheManager',
        'allocate_slots_fail',
        `allocate_slots(${request.request_id}): ${required_blocks} required > ${available_blocks} free -> None`,
        { request_id: request.request_id, required_blocks, available_blocks },
        REF.KVCacheManager_free_check,
      )
      return null
    }

    if (new_computed_block_list !== this.empty_kv_cache_blocks.blocks) {
      this.coordinator.allocate_new_computed_blocks(request.request_id, new_computed_block_list as KVCacheBlock[][])
    }

    const new_blocks = this.coordinator.allocate_new_blocks(request.request_id, num_tokens_need_slot)

    if (!this.enable_caching) return new KVCacheBlocks(new_blocks)

    // Only "finalized" tokens are cached: cap at request.num_tokens so
    // unverified draft tokens never get a hash.
    const num_tokens_to_cache = Math.min(total_computed_tokens + num_new_tokens, request.num_tokens)
    this.coordinator.cache_blocks(request, num_tokens_to_cache)
    return new KVCacheBlocks(new_blocks)
  }

  free(request: Request): void {
    this.coordinator.free(request.request_id)
  }

  cache_blocks(request: Request, num_computed_tokens: number): void {
    if (this.enable_caching) this.coordinator.cache_blocks(request, num_computed_tokens)
  }

  get_blocks(request_id: string): KVCacheBlocks {
    return new KVCacheBlocks(this.coordinator.get_blocks(request_id))
  }

  get_block_ids(request_id: string): number[][] {
    return this.get_blocks(request_id).get_block_ids()
  }

  get_num_common_prefix_blocks(running_request_id: string): number[] {
    return this.coordinator.get_num_common_prefix_blocks(running_request_id)
  }
}
