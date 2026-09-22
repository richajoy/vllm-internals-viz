// Every state mutation in the simulator emits an Event. The UI animates from
// this stream and the timeline/inspector renders it; it is the single source
// of truth for "what happened and where in vLLM it happens".

export type Component =
  | 'LLMEngine'
  | 'InputProcessor'
  | 'EngineCoreClient'
  | 'EngineCoreProc'
  | 'EngineCore'
  | 'Scheduler'
  | 'KVCacheManager'
  | 'KVCacheCoordinator'
  | 'SingleTypeKVCacheManager'
  | 'BlockPool'
  | 'FreeKVCacheBlockQueue'
  | 'Executor'
  | 'Worker'
  | 'GPUModelRunner'
  | 'Sampler'
  | 'RejectionSampler'
  | 'NgramProposer'
  | 'StructuredOutputManager'
  | 'OutputProcessor'

export type Phase =
  | 'add_request'
  | 'schedule'
  | 'execute_model'
  | 'grammar_bitmask'
  | 'sample_tokens'
  | 'update_from_output'
  | 'post_step'
  | 'output'

export interface SourceRef {
  /** Path inside the vLLM repo. */
  file: string
  line: number
  /** Name the 2025 blog used, if it differs from current source. */
  blogEraName?: string
}

export interface SimEvent {
  seq: number
  step: number
  phase: Phase
  component: Component
  kind: string
  /** Human-readable one-liner shown in the timeline. */
  message: string
  payload?: Record<string, unknown>
  ref?: SourceRef
}

export class EventLog {
  events: SimEvent[] = []
  step = 0
  phase: Phase = 'add_request'
  private seq = 0

  emit(
    component: Component,
    kind: string,
    message: string,
    payload?: Record<string, unknown>,
    ref?: SourceRef,
  ): SimEvent {
    const ev: SimEvent = {
      seq: this.seq++,
      step: this.step,
      phase: this.phase,
      component,
      kind,
      message,
      payload,
      ref,
    }
    this.events.push(ev)
    return ev
  }

  setPhase(phase: Phase): void {
    this.phase = phase
  }
}

/** Shared no-op log for unit tests that do not care about events. */
export const NULL_LOG = new EventLog()
