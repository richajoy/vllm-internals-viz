import type { Component, Phase } from '../sim/events'
import type { Snapshot } from '../sim/simulation'

interface Node {
  name: string
  file: string
  phases: Phase[]
  components: Component[]
  boundary?: string
}

const NODES: Node[] = [
  { name: 'LLM.generate / AsyncLLM', file: 'entrypoints/llm.py', phases: ['add_request'], components: ['LLMEngine'] },
  { name: 'InputProcessor', file: 'v1/engine/input_processor.py', phases: ['add_request'], components: ['InputProcessor'], boundary: 'tokenize · validate · assign_request_id → EngineCoreRequest' },
  { name: 'EngineCoreClient', file: 'v1/engine/core_client.py', phases: ['add_request'], components: ['EngineCoreClient'], boundary: 'process boundary: msgpack over ZMQ ROUTER→DEALER (InprocClient skips it)' },
  { name: 'EngineCoreProc input thread', file: 'v1/engine/core.py', phases: ['add_request'], components: ['EngineCoreProc'], boundary: 'thread boundary: EngineCoreRequest → Request, then input_queue' },
  { name: 'EngineCore busy loop', file: 'v1/engine/core.py', phases: ['add_request', 'post_step'], components: ['EngineCore'] },
  { name: 'Scheduler.schedule', file: 'v1/core/sched/scheduler.py', phases: ['schedule'], components: ['Scheduler'] },
  { name: 'KVCacheManager · BlockPool', file: 'v1/core/kv_cache_manager.py', phases: ['schedule', 'update_from_output'], components: ['KVCacheManager', 'BlockPool', 'SingleTypeKVCacheManager', 'KVCacheCoordinator', 'FreeKVCacheBlockQueue'] },
  { name: 'Executor.execute_model', file: 'v1/executor/', phases: ['execute_model'], components: ['Executor'], boundary: 'RPC 1: shm MessageQueue broadcast to workers (UniProc: direct call)' },
  { name: 'GPUModelRunner forward', file: 'v1/worker/gpu_model_runner.py', phases: ['execute_model'], components: ['GPUModelRunner', 'Worker'] },
  { name: 'get_grammar_bitmask (CPU, overlapped)', file: 'v1/core/sched/scheduler.py', phases: ['grammar_bitmask'], components: ['StructuredOutputManager'] },
  { name: 'Executor.sample_tokens', file: 'v1/worker/gpu_model_runner.py', phases: ['sample_tokens'], components: ['Sampler', 'RejectionSampler', 'NgramProposer'], boundary: 'RPC 2: ExecuteModelState → ModelRunnerOutput' },
  { name: 'Scheduler.update_from_output', file: 'v1/core/sched/scheduler.py', phases: ['update_from_output'], components: ['Scheduler'] },
  { name: 'output thread → OutputProcessor', file: 'v1/engine/output_processor.py', phases: ['output'], components: ['OutputProcessor', 'EngineCoreProc'], boundary: 'ZMQ PUSH→PULL, IncrementalDetokenizer → RequestOutput' },
]

export function PipelineRail({ snap, liveComponents }: { snap: Snapshot; liveComponents: Set<Component> }) {
  return (
    <div className="panel p-3 flex flex-col gap-1">
      <div className="panel-title mb-1">Request path</div>
      {NODES.map((n, i) => {
        const inPhase = !snap.idle && n.phases.includes(snap.phase)
        const live = inPhase && n.components.some((c) => liveComponents.has(c))
        return (
          <div key={n.name}>
            <div className="rail-node" data-live={live} style={{ opacity: inPhase ? 1 : 0.6 }}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[12px] leading-tight min-w-0" style={{ fontVariationSettings: live ? "'wght' 650" : "'wght' 450", overflowWrap: 'anywhere' }} title={n.file}>
                  {n.name.split(/(?<=[._])/).map((part, k) => (k === 0 ? part : [<wbr key={k} />, part]))}
                </span>
                
              </div>
            </div>
            {i < NODES.length - 1 && (
              <div className="pl-3 text-[10px] hint leading-tight py-[2px]" style={{ borderLeft: '1px solid var(--rule-strong)', marginLeft: 10 }}>
                {n.boundary ?? ''}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
