import { describe, expect, it } from 'vitest'
import { BlockPool } from './block_pool'
import { FreeKVCacheBlockQueue, KVCacheBlock, get_request_block_hasher, NONE_HASH, hash_block_tokens, fnv1a } from './kv_cache_utils'
import { KVCacheManager } from './kv_cache_manager'
import { Request, RequestStatus } from './request'

const ids = (blocks: { block_id: number }[]) => blocks.map((b) => b.block_id)

function mkReq(request_id: string, prompt: number[], block_size: number, max_tokens = 4, caching = true): Request {
  return new Request({
    request_id,
    prompt_token_ids: prompt,
    sampling_params: { max_tokens },
    arrival_time: 0,
    block_hasher: caching ? get_request_block_hasher(block_size) : null,
  })
}

describe('FreeKVCacheBlockQueue', () => {
  it('links blocks in id order with sentinels and supports O(1) middle removal', () => {
    const blocks = [0, 1, 2, 3].map((i) => new KVCacheBlock(i))
    const q = new FreeKVCacheBlockQueue(blocks)
    expect(ids(q.get_all_free_blocks())).toEqual([0, 1, 2, 3])
    expect(q.fake_free_list_head.block_id).toBe(-1)
    q.remove(blocks[2])
    expect(ids(q.get_all_free_blocks())).toEqual([0, 1, 3])
    expect(q.num_free_blocks).toBe(3)
    expect(blocks[2].prev_free_block).toBeNull()
    expect(() => q.remove(blocks[2])).toThrow()
  })

  it('popleft_n pops from head; append_n goes to tail; prepend_n goes to head', () => {
    const blocks = [0, 1, 2, 3, 4].map((i) => new KVCacheBlock(i))
    const q = new FreeKVCacheBlockQueue(blocks)
    expect(ids(q.popleft_n(2))).toEqual([0, 1])
    q.append_n([blocks[0]])
    q.prepend_n([blocks[1]])
    expect(ids(q.get_all_free_blocks())).toEqual([1, 2, 3, 4, 0])
    expect(ids(q.popleft_n(5))).toEqual([1, 2, 3, 4, 0])
    expect(q.num_free_blocks).toBe(0)
    expect(() => q.popleft()).toThrow()
  })
})

describe('block hashing', () => {
  it('chains from NONE_HASH through parents', () => {
    const h1 = hash_block_tokens(fnv1a, null, [1, 2, 3, 4])
    const h1b = hash_block_tokens(fnv1a, NONE_HASH, [1, 2, 3, 4])
    const h2 = hash_block_tokens(fnv1a, h1, [5, 6, 7, 8])
    const h2other = hash_block_tokens(fnv1a, 'xxxxxxxx', [5, 6, 7, 8])
    expect(h1).toBe(h1b)
    expect(h2).not.toBe(h2other)
  })

  it('request hasher only hashes full blocks and extends incrementally', () => {
    const req = mkReq('A', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2], 4)
    expect(req.block_hashes).toHaveLength(2)
    req.append_output_token_ids(11)
    expect(req.block_hashes).toHaveLength(3)
    expect(req.block_hashes[2]).toBe(hash_block_tokens(fnv1a, req.block_hashes[1], [9, 10, 2, 11]))
  })
})

describe('BlockPool', () => {
  it('reserves block 0 as null_block and pops from head', () => {
    const pool = new BlockPool(6, false, 4)
    expect(pool.null_block.block_id).toBe(0)
    expect(pool.null_block.is_null).toBe(true)
    expect(pool.get_num_free_blocks()).toBe(5)
    expect(ids(pool.get_new_blocks(2))).toEqual([1, 2])
    expect(pool.blocks[1].ref_cnt).toBe(1)
  })

  it('free_blocks: hashless blocks prepended (evicted first), hashed appended to tail', () => {
    const pool = new BlockPool(8, true, 4)
    const [b1, b2, b3] = pool.get_new_blocks(3)
    b1.set_block_hash('h1:0', 4)
    pool.cached_block_hash_to_block.insert('h1:0', b1)
    // Caller frees tail-first: [b3, b2, b1]; b2 and b3 are hashless.
    pool.free_blocks([b3, b2, b1])
    expect(ids(pool.free_block_queue.get_all_free_blocks())).toEqual([3, 2, 4, 5, 6, 7, 1])
  })

  it('lazy eviction: hash removed only when the block is re-allocated', () => {
    const pool = new BlockPool(4, true, 4)
    const [b1] = pool.get_new_blocks(1)
    b1.set_block_hash('h1:0', 4)
    pool.cached_block_hash_to_block.insert('h1:0', b1)
    pool.free_blocks([b1])
    expect(pool.cached_block_hash_to_block.size).toBe(1)
    expect(pool.get_cached_block('h1', [0])?.[0]).toBe(b1)
    // Touch pulls it back out of the free queue without losing the hash.
    pool.touch([b1])
    expect(b1.ref_cnt).toBe(1)
    expect(pool.get_num_free_blocks()).toBe(2)
    pool.free_blocks([b1])
    // Allocate everything: b2, b3 come first (never hashed), then b1 is evicted.
    expect(ids(pool.get_new_blocks(3))).toEqual([2, 3, 1])
    expect(b1.block_hash).toBeNull()
    expect(pool.cached_block_hash_to_block.size).toBe(0)
  })
})

describe('KVCacheManager', () => {
  it('Fig 3: 10 tokens, block_size 4 -> blocks 1,2,3; free queue continues at 4', () => {
    const mgr = new KVCacheManager({ num_gpu_blocks: 11, block_size: 4, max_model_len: 64, enable_caching: false })
    const req = mkReq('r0', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4, 4, false)
    req.status = RequestStatus.WAITING
    const [computed, n] = mgr.get_computed_blocks(req)
    expect(n).toBe(0)
    const blocks = mgr.allocate_slots(req, 10, { new_computed_blocks: computed, num_new_computed_tokens: 0 })
    expect(blocks?.get_block_ids()).toEqual([[1, 2, 3]])
    expect(mgr.block_pool.blocks[1].ref_cnt).toBe(1)
    expect(mgr.block_pool.blocks[1].block_hash).toBeNull()
    expect(ids(mgr.block_pool.free_block_queue.get_all_free_blocks())).toEqual([4, 5, 6, 7, 8, 9, 10])
  })

  it('Fig 6-8: second request reuses blocks 1,2 and gets block 6 as its first fresh block', () => {
    const mgr = new KVCacheManager({ num_gpu_blocks: 16, block_size: 4, max_model_len: 64, enable_caching: true })
    const prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    const a = mkReq('A', [...prefix, 9, 10, 2], 4)
    const [ca, na] = mgr.get_computed_blocks(a)
    expect(na).toBe(0)
    // First request: allocate for 11 prompt tokens, then grow with decode
    // tokens until it holds 5 blocks.
    const got = mgr.allocate_slots(a, 11, { new_computed_blocks: ca })
    expect(got?.get_block_ids()).toEqual([[1, 2, 3]])
    expect(mgr.block_pool.cached_block_hash_to_block.size).toBe(2) // blocks 1,2 hashed, 3 incomplete
    expect(mgr.block_pool.blocks[3].block_hash).toBeNull()
    a.status = RequestStatus.RUNNING
    a.num_computed_tokens = 11
    // Decode enough tokens to occupy blocks 4 and 5.
    for (let t = 0; t < 9; t++) {
      a.append_output_token_ids(100 + t)
      const nb = mgr.allocate_slots(a, 1)
      expect(nb).not.toBeNull()
      a.num_computed_tokens += 1
    }
    expect(mgr.get_block_ids('A')).toEqual([[1, 2, 3, 4, 5]])
    mgr.free(a)
    expect(mgr.block_pool.blocks[1].ref_cnt).toBe(0)
    // Tail-first free: blocks 5,4,3,2,1 are all hashed (20 tokens = 5 full blocks) -> appended in that order.
    expect(ids(mgr.block_pool.free_block_queue.get_all_free_blocks())).toEqual([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 5, 4, 3, 2, 1])

    const b = mkReq('B', [...prefix, 12, 10, 2], 4)
    const [cb, nb] = mgr.get_computed_blocks(b)
    expect(nb).toBe(8)
    expect(cb.get_block_ids()).toEqual([[1, 2]])
    const fresh = mgr.allocate_slots(b, 3, { new_computed_blocks: cb, num_new_computed_tokens: 8 })
    expect(fresh?.get_block_ids()).toEqual([[6]])
    expect(mgr.get_block_ids('B')).toEqual([[1, 2, 6]])
    expect(mgr.block_pool.blocks[1].ref_cnt).toBe(1)
    expect(ids(mgr.block_pool.free_block_queue.get_all_free_blocks())).toEqual([7, 8, 9, 10, 11, 12, 13, 14, 15, 5, 4, 3])
  })

  it('all-hit prompt recomputes the last token (max hit = num_tokens - 1)', () => {
    const mgr = new KVCacheManager({ num_gpu_blocks: 8, block_size: 4, max_model_len: 64, enable_caching: true })
    const a = mkReq('A', [1, 2, 3, 4, 5, 6, 7, 8], 4)
    mgr.allocate_slots(a, 8, { new_computed_blocks: mgr.get_computed_blocks(a)[0] })
    const b = mkReq('B', [1, 2, 3, 4, 5, 6, 7, 8], 4)
    const [, n] = mgr.get_computed_blocks(b)
    expect(n).toBe(4)
  })

  it('returns null when the pool is exhausted (preemption trigger)', () => {
    const mgr = new KVCacheManager({ num_gpu_blocks: 3, block_size: 4, max_model_len: 64, enable_caching: false })
    const a = mkReq('A', [1, 2, 3, 4, 5, 6, 7, 8], 4, 4, false)
    expect(mgr.allocate_slots(a, 8)).not.toBeNull()
    const b = mkReq('B', [1, 2, 3, 4], 4, 4, false)
    expect(mgr.allocate_slots(b, 4)).toBeNull()
  })
})
