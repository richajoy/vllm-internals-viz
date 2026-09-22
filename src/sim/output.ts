// Port of vllm/v1/core/sched/output.py (SchedulerOutput and friends) and the
// worker-side ModelRunnerOutput from vllm/v1/outputs.py.

import type { FinishReason, SamplingParams } from './request'

export interface NewRequestData {
  req_id: string
  prompt_token_ids: number[]
  sampling_params: SamplingParams
  block_ids: number[][]
  num_computed_tokens: number
}

export interface CachedRequestData {
  req_ids: string[]
  resumed_req_ids: Set<string>
  new_block_ids: (number[][] | null)[]
  num_computed_tokens: number[]
  num_output_tokens: number[]
}

export interface SchedulerOutput {
  scheduled_new_reqs: NewRequestData[]
  scheduled_cached_reqs: CachedRequestData
  num_scheduled_tokens: Record<string, number>
  total_num_scheduled_tokens: number
  scheduled_spec_decode_tokens: Record<string, number[]>
  num_common_prefix_blocks: number[]
  preempted_req_ids: Set<string>
  finished_req_ids: Set<string>
  has_structured_output_requests: boolean
  pending_structured_output_tokens: boolean
  num_spec_tokens_to_schedule: number
  /** Simulator-only: which step produced this output. */
  step: number
}

export interface GrammarOutput {
  structured_output_request_ids: string[]
  /** One packed row per request; bit i set = token i allowed. */
  grammar_bitmask: number[][]
}

export interface DraftTokenIds {
  req_ids: string[]
  draft_token_ids: number[][]
}

export interface ModelRunnerOutput {
  req_ids: string[]
  req_id_to_index: Record<string, number>
  /** Per request: [] while still prefilling, else sampled ids (spec: accepted + bonus/recovered). */
  sampled_token_ids: number[][]
}

export const EMPTY_MODEL_RUNNER_OUTPUT: ModelRunnerOutput = {
  req_ids: [],
  req_id_to_index: {},
  sampled_token_ids: [],
}

export interface EngineCoreOutput {
  request_id: string
  new_token_ids: number[]
  finish_reason: FinishReason | null
  stop_reason: number | string | null
}

export interface EngineCoreOutputs {
  outputs: EngineCoreOutput[]
  finished_requests: Set<string>
}
