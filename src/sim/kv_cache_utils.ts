// Port of the parts of vllm/v1/core/kv_cache_utils.py the simulator needs:
// KVCacheBlock, FreeKVCacheBlockQueue, block hashing and the request block
// hasher. Hashes are short hex strings instead of bytes so they stay legible
// in the UI; chaining semantics are identical.

import type { Request } from './request'

/** Chained prefix hash at hash_block_size granularity (bytes in vLLM). */
export type BlockHash = string
/** BlockHash + 4-byte big-endian group id in vLLM; `hash:group` here. */
export type BlockHashWithGroupId = string

export function make_block_hash_with_group_id(h: BlockHash, group_id: number): BlockHashWithGroupId {
  return `${h}:${group_id}`
}

export function get_block_hash(key: BlockHashWithGroupId): BlockHash {
  return key.slice(0, key.lastIndexOf(':'))
}

export class KVCacheBlock {
  block_id: number
  ref_cnt = 0
  private _block_hash: BlockHashWithGroupId | null = null
  private _block_hash_num_tokens: number | null = null
  prev_free_block: KVCacheBlock | null = null
  next_free_block: KVCacheBlock | null = null
  is_null = false

  constructor(block_id: number) {
    this.block_id = block_id
  }

  get block_hash(): BlockHashWithGroupId | null {
    return this._block_hash
  }

  get block_hash_num_tokens(): number | null {
    return this._block_hash_num_tokens
  }

  set_block_hash(block_hash: BlockHashWithGroupId, num_tokens: number | null = null): void {
    if (this._block_hash !== null || this._block_hash_num_tokens !== null) {
      throw new Error('The block already has a hash. This should not happen.')
    }
    this._block_hash = block_hash
    this._block_hash_num_tokens = num_tokens
  }

  reset_hash(): void {
    this._block_hash = null
    this._block_hash_num_tokens = null
  }
}

/**
 * Doubly linked list of free blocks with fake head/tail sentinels so a block
 * can be removed from the middle in O(1) when a prefix-cache hit touches it.
 */
export class FreeKVCacheBlockQueue {
  num_free_blocks: number
  fake_free_list_head: KVCacheBlock
  fake_free_list_tail: KVCacheBlock

  constructor(blocks: KVCacheBlock[]) {
    this.num_free_blocks = blocks.length
    for (let i = 0; i < this.num_free_blocks; i++) {
      if (i > 0) blocks[i].prev_free_block = blocks[i - 1]
      if (i < this.num_free_blocks - 1) blocks[i].next_free_block = blocks[i + 1]
    }
    this.fake_free_list_head = new KVCacheBlock(-1)
    this.fake_free_list_tail = new KVCacheBlock(-1)
    if (this.num_free_blocks > 0) {
      this.fake_free_list_head.next_free_block = blocks[0]
      blocks[0].prev_free_block = this.fake_free_list_head
      this.fake_free_list_tail.prev_free_block = blocks[blocks.length - 1]
      blocks[blocks.length - 1].next_free_block = this.fake_free_list_tail
    } else {
      this.fake_free_list_head.next_free_block = this.fake_free_list_tail
      this.fake_free_list_tail.prev_free_block = this.fake_free_list_head
    }
  }

  popleft(): KVCacheBlock {
    const first = this.fake_free_list_head.next_free_block
    if (first === null || first === this.fake_free_list_tail) {
      throw new Error('No free blocks available')
    }
    if (first.next_free_block === null) {
      throw new Error("Invalid block found in popleft() which doesn't have a valid next_free_block")
    }
    this.fake_free_list_head.next_free_block = first.next_free_block
    first.next_free_block.prev_free_block = this.fake_free_list_head
    first.prev_free_block = first.next_free_block = null
    this.num_free_blocks -= 1
    return first
  }

  popleft_n(n: number): KVCacheBlock[] {
    if (n === 0) return []
    if (this.num_free_blocks < n) throw new Error(`popleft_n(${n}) with ${this.num_free_blocks} free`)
    this.num_free_blocks -= n
    let curr = this.fake_free_list_head.next_free_block
    const ret: KVCacheBlock[] = []
    for (let i = 0; i < n; i++) {
      if (curr === null) throw new Error('free list corrupted')
      ret.push(curr)
      const last = curr
      curr = curr.next_free_block
      last.prev_free_block = null
      last.next_free_block = null
    }
    if (curr !== null) {
      this.fake_free_list_head.next_free_block = curr
      curr.prev_free_block = this.fake_free_list_head
    }
    return ret
  }

  remove(block: KVCacheBlock): void {
    if (block.prev_free_block === null || block.next_free_block === null) {
      throw new Error(`remove() called on an invalid block: ${block.block_id}`)
    }
    block.prev_free_block.next_free_block = block.next_free_block
    block.next_free_block.prev_free_block = block.prev_free_block
    block.prev_free_block = block.next_free_block = null
    this.num_free_blocks -= 1
  }

  append(block: KVCacheBlock): void {
    const last = this.fake_free_list_tail.prev_free_block
    if (last === null) throw new Error('prev_free_block of fake_free_list_tail should always exist')
    last.next_free_block = block
    block.prev_free_block = last
    block.next_free_block = this.fake_free_list_tail
    this.fake_free_list_tail.prev_free_block = block
    this.num_free_blocks += 1
  }

  prepend_n(blocks: KVCacheBlock[]): void {
    if (blocks.length === 0) return
    const first = this.fake_free_list_head.next_free_block
    if (first === null) throw new Error('next_free_block of fake_free_list_head should always exist')
    let prev: KVCacheBlock = this.fake_free_list_head
    for (const block of blocks) {
      block.prev_free_block = prev
      prev.next_free_block = block
      prev = block
    }
    prev.next_free_block = first
    first.prev_free_block = prev
    this.num_free_blocks += blocks.length
  }

  append_n(blocks: KVCacheBlock[]): void {
    if (blocks.length === 0) return
    let last = this.fake_free_list_tail.prev_free_block
    if (last === null) throw new Error('prev_free_block of fake_free_list_tail should always exist')
    for (const block of blocks) {
      block.prev_free_block = last
      last.next_free_block = block
      last = block
    }
    last.next_free_block = this.fake_free_list_tail
    this.fake_free_list_tail.prev_free_block = last
    this.num_free_blocks += blocks.length
  }

  get_all_free_blocks(): KVCacheBlock[] {
    const ret: KVCacheBlock[] = []
    let curr = this.fake_free_list_head.next_free_block
    if (curr === null) throw new Error('next_free_block of fake_free_list_head should always exist')
    while (curr.next_free_block !== null) {
      ret.push(curr)
      curr = curr.next_free_block
    }
    return ret
  }
}

// ---------------------------------------------------------------------------
// Hashing. vLLM: hash_fn((parent_block_hash, tuple(token_ids), extra_keys)).
// We use FNV-1a over a canonical string and render 8 hex chars.

export type HashFn = (payload: string) => BlockHash

export const fnv1a: HashFn = (payload) => {
  let h = 0x811c9dc5
  for (let i = 0; i < payload.length; i++) {
    h ^= payload.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h.toString(16).padStart(8, '0')
}

/** vLLM seeds the chain with NONE_HASH (random unless PYTHONHASHSEED is set). */
export const NONE_HASH: BlockHash = fnv1a('NONE_HASH')

export function hash_block_tokens(
  hash_fn: HashFn,
  parent_block_hash: BlockHash | null,
  curr_block_token_ids: readonly number[],
  extra_keys: readonly unknown[] | null = null,
): BlockHash {
  const parent = parent_block_hash ?? NONE_HASH
  const extra = extra_keys && extra_keys.length ? JSON.stringify(extra_keys) : ''
  return hash_fn(`${parent}|${curr_block_token_ids.join(',')}|${extra}`)
}

export interface BlockHashDebug {
  parent: BlockHash
  token_ids: number[]
  hash: BlockHash
}

/**
 * Returns a function computing the not-yet-computed full-block hashes of a
 * request, chained over the full prefix at hash_block_size granularity.
 */
export function get_request_block_hasher(
  hash_block_size: number,
  caching_hash_fn: HashFn = fnv1a,
): (request: Request) => BlockHash[] {
  return function request_block_hasher(request: Request): BlockHash[] {
    let start_token_idx = request.block_hashes.length * hash_block_size
    const num_tokens = request.num_tokens
    if (start_token_idx + hash_block_size > num_tokens) return []

    let prev_block_hash_value: BlockHash | null =
      request.block_hashes.length > 0 ? request.block_hashes[request.block_hashes.length - 1] : null
    const new_block_hashes: BlockHash[] = []
    for (;;) {
      const end_token_idx = start_token_idx + hash_block_size
      if (end_token_idx > num_tokens) break
      const block_tokens = request.all_token_ids.slice(start_token_idx, end_token_idx)
      const block_hash = hash_block_tokens(caching_hash_fn, prev_block_hash_value, block_tokens, null)
      new_block_hashes.push(block_hash)
      start_token_idx += hash_block_size
      prev_block_hash_value = block_hash
    }
    return new_block_hashes
  }
}

export function cdiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b)
}
