// Source references pinned to vLLM commit adc3e03517 (main, 2026-08-06).
// `blogEraName` records what Aleksa Gordic's Aug-2025 post (commit 42172ad)
// called the thing when the name differs today.

import type { SourceRef } from './events'

export const VLLM_COMMIT = 'adc3e03517d2e7333a3bb2083bb4d394a2986876'
export const VLLM_COMMIT_SHORT = 'adc3e03517'

export function githubUrl(ref: SourceRef): string {
  return `https://github.com/vllm-project/vllm/blob/${VLLM_COMMIT}/${ref.file}#L${ref.line}`
}

const r = (file: string, line: number, blogEraName?: string): SourceRef => ({
  file,
  line,
  blogEraName,
})

export const REF = {
  // Engine front end
  LLMEngine: r('vllm/v1/engine/llm_engine.py', 48),
  LLMEngine_add_request: r('vllm/v1/engine/llm_engine.py', 218),
  LLMEngine_step: r('vllm/v1/engine/llm_engine.py', 298),
  InputProcessor: r('vllm/v1/engine/input_processor.py', 38, 'Processor'),
  InputProcessor_process_inputs: r('vllm/v1/engine/input_processor.py', 251),
  InputProcessor_assign_request_id: r('vllm/v1/engine/input_processor.py', 231),
  EngineCoreRequest: r('vllm/v1/engine/__init__.py', 97),
  OutputProcessor: r('vllm/v1/engine/output_processor.py', 429),
  RequestState: r('vllm/v1/engine/output_processor.py', 129),
  IncrementalDetokenizer: r('vllm/v1/engine/detokenizer.py', 31, 'Detokenizer'),

  // Engine core client / proc
  EngineCoreClient_make_client: r('vllm/v1/engine/core_client.py', 89),
  InprocClient: r('vllm/v1/engine/core_client.py', 306),
  SyncMPClient: r('vllm/v1/engine/core_client.py', 805),
  SyncMPClient_send_input: r('vllm/v1/engine/core_client.py', 887),
  EngineCoreProc: r('vllm/v1/engine/core.py', 1004),
  EngineCoreProc_process_input_sockets: r('vllm/v1/engine/core.py', 1639),
  EngineCoreProc_run_busy_loop: r('vllm/v1/engine/core.py', 1372),
  EngineCoreProc_process_output_sockets: r('vllm/v1/engine/core.py', 1737),
  EngineCore_preprocess_add_request: r('vllm/v1/engine/core.py', 965),
  EngineCore_add_request: r('vllm/v1/engine/core.py', 435),
  EngineCore_step: r('vllm/v1/engine/core.py', 580),
  EngineCore_step_with_batch_queue: r('vllm/v1/engine/core.py', 621),
  EngineCore_batch_queue: r('vllm/v1/engine/core.py', 206),
  EngineCore_post_step: r('vllm/v1/engine/core.py', 612),
  EngineCore_initialize_kv_caches: r('vllm/v1/engine/core.py', 250),

  // Request
  Request: r('vllm/v1/request.py', 59),
  Request_from_engine_core_request: r('vllm/v1/request.py', 224),
  Request_update_block_hashes: r('vllm/v1/request.py', 265),
  Request_num_tokens_with_spec: r('vllm/v1/request.py', 279),
  RequestStatus: r('vllm/v1/request.py', 351),
  RequestStatus_WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR: r(
    'vllm/v1/request.py',
    353,
    'WAITING_FOR_FSM',
  ),

  // Scheduler
  Scheduler: r('vllm/v1/core/sched/scheduler.py', 79),
  Scheduler_queues: r('vllm/v1/core/sched/scheduler.py', 188),
  Scheduler_schedule: r('vllm/v1/core/sched/scheduler.py', 440),
  Scheduler_no_phases_note: r('vllm/v1/core/sched/scheduler.py', 442),
  Scheduler_running_loop: r('vllm/v1/core/sched/scheduler.py', 484),
  Scheduler_num_new_tokens_running: r('vllm/v1/core/sched/scheduler.py', 517),
  Scheduler_preempt_loop: r('vllm/v1/core/sched/scheduler.py', 577),
  Scheduler_spec_tokens: r('vllm/v1/core/sched/scheduler.py', 641),
  Scheduler_waiting_loop: r('vllm/v1/core/sched/scheduler.py', 684),
  Scheduler_max_num_running: r('vllm/v1/core/sched/scheduler.py', 691),
  Scheduler_num_new_tokens_waiting: r('vllm/v1/core/sched/scheduler.py', 876),
  Scheduler_chunked_prefill_break: r('vllm/v1/core/sched/scheduler.py', 905),
  Scheduler_admit: r('vllm/v1/core/sched/scheduler.py', 1056),
  Scheduler_build_output: r('vllm/v1/core/sched/scheduler.py', 1209),
  Scheduler_preempt_request: r('vllm/v1/core/sched/scheduler.py', 1275),
  Scheduler_update_after_schedule: r('vllm/v1/core/sched/scheduler.py', 1318),
  Scheduler_make_cached_request_data: r('vllm/v1/core/sched/scheduler.py', 1411),
  Scheduler_get_grammar_bitmask: r('vllm/v1/core/sched/scheduler.py', 1647),
  Scheduler_update_from_output: r('vllm/v1/core/sched/scheduler.py', 1671),
  Scheduler_spec_rejection_fixup: r('vllm/v1/core/sched/scheduler.py', 1767),
  Scheduler_update_request_with_output: r('vllm/v1/core/sched/scheduler.py', 2106),
  Scheduler_check_stop: r('vllm/v1/core/sched/utils.py', 94),
  Scheduler_free_request: r('vllm/v1/core/sched/scheduler.py', 2312),
  Scheduler_update_draft_token_ids: r('vllm/v1/core/sched/scheduler.py', 2158),
  AsyncScheduler: r('vllm/v1/core/sched/async_scheduler.py', 12),
  AsyncScheduler_update_after_schedule: r('vllm/v1/core/sched/async_scheduler.py', 19),
  AsyncScheduler_update_request_with_output: r('vllm/v1/core/sched/async_scheduler.py', 51),
  SchedulerOutput: r('vllm/v1/core/sched/output.py', 192),
  NewRequestData: r('vllm/v1/core/sched/output.py', 35),
  CachedRequestData: r('vllm/v1/core/sched/output.py', 116),
  FCFSRequestQueue: r('vllm/v1/core/sched/request_queue.py', 75),
  PriorityRequestQueue: r('vllm/v1/core/sched/request_queue.py', 131),

  // KV cache manager
  KVCacheManager: r('vllm/v1/core/kv_cache_manager.py', 117),
  KVCacheBlocks: r('vllm/v1/core/kv_cache_manager.py', 32),
  KVCacheManager_get_computed_blocks: r('vllm/v1/core/kv_cache_manager.py', 229),
  KVCacheManager_recompute_last_token: r('vllm/v1/core/kv_cache_manager.py', 253),
  KVCacheManager_allocate_slots: r('vllm/v1/core/kv_cache_manager.py', 344),
  KVCacheManager_watermark: r('vllm/v1/core/kv_cache_manager.py', 463),
  KVCacheManager_free_check: r('vllm/v1/core/kv_cache_manager.py', 521),
  KVCacheManager_cache_cap: r('vllm/v1/core/kv_cache_manager.py', 554),
  KVCacheManager_free: r('vllm/v1/core/kv_cache_manager.py', 567),
  UnitaryKVCacheCoordinator: r('vllm/v1/core/kv_cache_coordinator.py', 435),
  KVCacheCoordinator_allocate_new_computed_blocks: r('vllm/v1/core/kv_cache_coordinator.py', 192),
  SingleTypeKVCacheManager_req_to_blocks: r('vllm/v1/core/single_type_kv_cache_manager.py', 97),
  SingleTypeKVCacheManager_get_num_blocks_to_allocate: r('vllm/v1/core/single_type_kv_cache_manager.py', 144),
  SingleTypeKVCacheManager_add_local_computed_blocks: r('vllm/v1/core/single_type_kv_cache_manager.py', 232, 'save_new_computed_blocks'),
  SingleTypeKVCacheManager_allocate_new_blocks: r('vllm/v1/core/single_type_kv_cache_manager.py', 330),
  SingleTypeKVCacheManager_cache_blocks: r('vllm/v1/core/single_type_kv_cache_manager.py', 427),
  SingleTypeKVCacheManager_free: r('vllm/v1/core/single_type_kv_cache_manager.py', 519),
  FullAttentionManager_find_longest_cache_hit: r('vllm/v1/core/single_type_kv_cache_manager.py', 682),

  // Block pool
  BlockPool: r('vllm/v1/core/block_pool.py', 143),
  BlockHashToBlockMap: r('vllm/v1/core/block_pool.py', 33, 'cached_block_hash_to_block (dict)'),
  BlockPool_null_block: r('vllm/v1/core/block_pool.py', 190),
  BlockPool_cache_full_blocks: r('vllm/v1/core/block_pool.py', 225),
  BlockPool_get_new_blocks: r('vllm/v1/core/block_pool.py', 647),
  BlockPool_maybe_evict_cached_block: r('vllm/v1/core/block_pool.py', 679),
  BlockPool_touch: r('vllm/v1/core/block_pool.py', 702),
  BlockPool_free_blocks: r('vllm/v1/core/block_pool.py', 719),
  KVCacheBlock: r('vllm/v1/core/kv_cache_utils.py', 117),
  FreeKVCacheBlockQueue: r('vllm/v1/core/kv_cache_utils.py', 184),
  FreeKVCacheBlockQueue_popleft_n: r('vllm/v1/core/kv_cache_utils.py', 273),
  FreeKVCacheBlockQueue_remove: r('vllm/v1/core/kv_cache_utils.py', 306),
  FreeKVCacheBlockQueue_prepend_n: r('vllm/v1/core/kv_cache_utils.py', 349),
  FreeKVCacheBlockQueue_append_n: r('vllm/v1/core/kv_cache_utils.py', 370),
  hash_block_tokens: r('vllm/v1/core/kv_cache_utils.py', 576),
  NONE_HASH: r('vllm/v1/core/kv_cache_utils.py', 87),
  get_request_block_hasher: r('vllm/v1/core/kv_cache_utils.py', 671),
  BlockHash: r('vllm/v1/core/kv_cache_utils.py', 41, 'BlockHash(hash_value, token_ids)'),

  // Executor / worker
  Executor_execute_model: r('vllm/v1/executor/abstract.py', 221),
  Executor_sample_tokens: r('vllm/v1/executor/abstract.py', 241),
  UniProcExecutor: r('vllm/v1/executor/uniproc_executor.py', 45),
  MultiprocExecutor: r('vllm/v1/executor/multiproc_executor.py', 108, 'MultiProcExecutor'),
  MultiprocExecutor_rpc_broadcast_mq: r('vllm/v1/executor/multiproc_executor.py', 156),
  WorkerProc_worker_busy_loop: r('vllm/v1/executor/multiproc_executor.py', 1008),
  Worker: r('vllm/v1/worker/gpu_worker.py', 128),
  Worker_execute_model: r('vllm/v1/worker/gpu_worker.py', 1024),
  Worker_sample_tokens: r('vllm/v1/worker/gpu_worker.py', 1017),
  GPUModelRunner_execute_model: r('vllm/v1/worker/gpu_model_runner.py', 4174),
  GPUModelRunner_update_states: r('vllm/v1/worker/gpu_model_runner.py', 1201),
  GPUModelRunner_prepare_inputs: r('vllm/v1/worker/gpu_model_runner.py', 1969),
  GPUModelRunner_logits_indices: r('vllm/v1/worker/gpu_model_runner.py', 2248),
  GPUModelRunner_compute_logits: r('vllm/v1/worker/gpu_model_runner.py', 4481),
  GPUModelRunner_ExecuteModelState: r('vllm/v1/worker/gpu_model_runner.py', 4513),
  GPUModelRunner_sample_tokens: r('vllm/v1/worker/gpu_model_runner.py', 4550),
  GPUModelRunner_apply_grammar_bitmask: r('vllm/v1/worker/gpu_model_runner.py', 4581),
  GPUModelRunner_propose_draft_token_ids: r('vllm/v1/worker/gpu_model_runner.py', 4999),
  GPUModelRunnerV2: r('vllm/v1/worker/gpu/model_runner.py', 136),
  BlockTable_compute_slot_mapping: r('vllm/v1/worker/block_table.py', 182),
  BlockTable_slot_formula: r('vllm/v1/worker/block_table.py', 430),
  ModelRunnerOutput: r('vllm/v1/outputs.py', 261),
  Sampler: r('vllm/v1/sample/sampler.py', 20),
  RejectionSampler: r('vllm/v1/sample/rejection_sampler.py', 38),
  NgramProposer: r('vllm/v1/spec_decode/ngram_proposer.py', 12),
  StructuredOutputManager_grammar_bitmask: r('vllm/v1/structured_output/__init__.py', 212),

  // Config defaults
  CacheConfig_block_size: r('vllm/config/cache.py', 47),
  CacheConfig_gpu_memory_utilization: r('vllm/config/cache.py', 68),
  SchedulerConfig_enable_chunked_prefill: r('vllm/config/scheduler.py', 74),
  SchedulerConfig_long_prefill_token_threshold: r('vllm/config/scheduler.py', 70),
  SchedulerConfig_async_scheduling: r('vllm/config/scheduler.py', 148),
  VllmConfig_async_scheduling_default: r('vllm/config/vllm.py', 1113),
  VllmConfig_use_v2_model_runner: r('vllm/config/vllm.py', 596),
} as const satisfies Record<string, SourceRef>

export type RefKey = keyof typeof REF
