import { useState } from 'react'
import { githubUrl, REF } from '../sim/source_refs'
import type { SourceRef } from '../sim/events'

interface Item {
  then: string
  now: string
  ref: SourceRef
}

const ITEMS: Item[] = [
  { then: '`Processor` turns raw inputs into `EngineCoreRequest`', now: 'Renamed `InputProcessor`; tokenization is moving into a `Renderer`. `assign_request_id` rewrites the id to `<external>-<8 hex>`.', ref: REF.InputProcessor },
  { then: 'Engine core wraps `EngineCoreRequest` in `Request` when adding it', now: '`preprocess_add_request` does the conversion on the ZMQ input thread so it overlaps the forward pass; `add_request` now takes a `Request`.', ref: REF.EngineCore_preprocess_add_request },
  { then: '`step()` = schedule, forward pass, postprocess', now: '`execute_model` is launched non-blocking, `get_grammar_bitmask` runs on the CPU meanwhile, then `sample_tokens` is a second RPC with `ExecuteModelState` stashed between.', ref: REF.EngineCore_step },
  { then: 'Async scheduling is experimental and skipped', now: 'It is the default: `AsyncScheduler` + `step_with_batch_queue`, with `num_output_placeholders` standing in for tokens not yet sampled.', ref: REF.VllmConfig_async_scheduling_default },
  { then: 'One `model_runner` with an `InputBatch`', now: 'Two runners; the V2 runner under `vllm/v1/worker/gpu/` is the default for most dense generate models.', ref: REF.VllmConfig_use_v2_model_runner },
  { then: '`slot_mapping` built on the CPU in prepare-inputs', now: 'Same formula, but a Triton kernel (`_compute_slot_mapping_kernel`) with context-parallel and kernel-block-size branches.', ref: REF.BlockTable_slot_formula },
  { then: '`WAITING_FOR_FSM` status', now: '`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`; plus new `WAITING_FOR_STREAMING_REQ`, `FINISHED_ERROR`, `FINISHED_REPETITION`.', ref: REF.RequestStatus_WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR },
  { then: 'Scheduler has `waiting` and `running`', now: 'Plus a second `skipped_waiting` queue for requests blocked on grammar compile or remote KVs.', ref: REF.Scheduler_queues },
  { then: '`num_new_tokens = num_tokens_with_spec - num_computed_tokens`', now: '`+ num_output_placeholders` is added for async scheduling before the clamps.', ref: REF.Scheduler_num_new_tokens_running },
  { then: 'Chunked prefill is enabled via `long_prefill_token_threshold`', now: '`enable_chunked_prefill` defaults to True; the threshold (default 0) only caps a single request\'s chunk.', ref: REF.SchedulerConfig_enable_chunked_prefill },
  { then: 'Freed blocks are appended to the tail of `free_block_queue`', now: 'Hashless blocks are prepended to the head (evicted first); only hashed blocks go to the tail in LRU order.', ref: REF.BlockPool_free_blocks },
  { then: '`BlockHash` holds the hash and its token ids; `cached_block_hash_to_block` is a dict', now: '`BlockHash` is raw bytes; the map is a `BlockHashToBlockMap` that only allocates an inner dict on collision.', ref: REF.BlockHashToBlockMap },
  { then: '`save_new_computed_blocks` then allocate', now: '`coordinator.allocate_new_computed_blocks` touches every group\'s hits first (two-phase), then `allocate_new_blocks`; `allocate_slots` also has a watermark and a full-sequence admission gate.', ref: REF.SingleTypeKVCacheManager_add_local_computed_blocks },
  { then: 'Grammar bitmask prepared inside scheduling', now: '`Scheduler.get_grammar_bitmask` is called between `execute_model` and `sample_tokens`; the mask is applied worker-side.', ref: REF.Scheduler_get_grammar_bitmask },
  { then: 'V1 does not support the draft-model method', now: '`draft_model`, `eagle3`, `dflash`, `dspark`, `ngram_gpu` and ~30 MTP variants exist; `EagleProposer` is a subclass of `SpecDecodeBaseProposer`.', ref: REF.GPUModelRunner_propose_draft_token_ids },
  { then: '`gpu_memory_utilization` 0.9', now: '0.92, and `max_num_seqs` / `max_num_batched_tokens` defaults depend on GPU memory.', ref: REF.CacheConfig_gpu_memory_utilization },
]

function md(s: string) {
  return s.split(/(`[^`]+`)/).map((part, i) => (part.startsWith('`') ? <code key={i} className="mono">{part.slice(1, -1)}</code> : <span key={i}>{part}</span>))
}

export function Drift() {
  const [open, setOpen] = useState(false)
  return (
    <div className="panel p-3">
      <button className="tbtn w-full text-left" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? 'Hide' : 'Show'} what changed in vLLM since Aug 2025 ({ITEMS.length})
      </button>
      {open && (
        <ol className="m-0 mt-2 pl-4 flex flex-col gap-2 text-[12px] leading-snug" style={{ overflowWrap: 'anywhere' }}>
          {ITEMS.map((it, i) => (
            <li key={i}>
              <div className="hint">{md(it.then)}</div>
              <div>{md(it.now)}</div>
              <a className="mono text-[11px]" style={{ color: 'var(--accent)' }} href={githubUrl(it.ref)} target="_blank" rel="noreferrer">
                {it.ref.file}:{it.ref.line}
              </a>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
