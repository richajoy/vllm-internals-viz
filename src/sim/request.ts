// Port of vllm/v1/request.py (Request, RequestStatus) reduced to the fields
// the scheduler and KV cache manager touch.

import type { BlockHash } from './kv_cache_utils'

export const RequestStatus = {
  WAITING: 1,
  WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR: 2,
  WAITING_FOR_REMOTE_KVS: 3,
  WAITING_FOR_STREAMING_REQ: 4,
  RUNNING: 5,
  PREEMPTED: 6,
  // Anything after PREEMPTED is a finished status.
  FINISHED_STOPPED: 7,
  FINISHED_LENGTH_CAPPED: 8,
  FINISHED_ABORTED: 9,
  FINISHED_IGNORED: 10,
  FINISHED_ERROR: 11,
  FINISHED_REPETITION: 12,
} as const
export type RequestStatus = (typeof RequestStatus)[keyof typeof RequestStatus]

export const RequestStatusName: Record<RequestStatus, string> = Object.fromEntries(
  Object.entries(RequestStatus).map(([k, v]) => [v, k]),
) as Record<RequestStatus, string>

export type FinishReason = 'stop' | 'length' | 'abort' | 'error' | 'repetition'

const FINISHED_REASON_MAP: Partial<Record<RequestStatus, FinishReason>> = {
  [RequestStatus.FINISHED_STOPPED]: 'stop',
  [RequestStatus.FINISHED_LENGTH_CAPPED]: 'length',
  [RequestStatus.FINISHED_ABORTED]: 'abort',
  [RequestStatus.FINISHED_IGNORED]: 'length',
  [RequestStatus.FINISHED_ERROR]: 'error',
  [RequestStatus.WAITING_FOR_STREAMING_REQ]: 'stop',
  [RequestStatus.FINISHED_REPETITION]: 'repetition',
}

export function isFinishedStatus(status: RequestStatus): boolean {
  return status > RequestStatus.PREEMPTED
}

export interface SamplingParams {
  max_tokens: number
  ignore_eos?: boolean
  stop_token_ids?: number[]
  /** Structured output: allowed completions (toy choice grammar). */
  guided_choice?: string[]
}

/** What InputProcessor produces; msgspec Struct in vLLM. */
export interface EngineCoreRequest {
  request_id: string
  external_req_id: string
  prompt_token_ids: number[]
  sampling_params: SamplingParams
  arrival_time: number
  priority: number
  client_index: number
}

export type BlockHasher = (request: Request) => BlockHash[]

export class Request {
  request_id: string
  external_req_id: string
  client_index: number
  priority: number
  sampling_params: SamplingParams
  arrival_time: number
  status: RequestStatus
  max_tokens: number
  prompt_token_ids: number[]
  num_prompt_tokens: number
  private _output_token_ids: number[] = []
  private _all_token_ids: number[]

  // Async scheduling
  num_output_placeholders = 0
  num_stale_output_tokens = 0
  drop_stale_output = false
  num_in_flight_tokens = 0
  next_decode_eligible_step = 0
  last_sched_seq = 0

  spec_token_ids: number[] = []
  num_computed_tokens = 0
  is_prefill_chunk = false
  shared_prefix_boundary = 0
  num_preemptions = 0
  block_hashes: BlockHash[] = []
  stop_reason: number | string | null = null
  private _block_hasher: BlockHasher | null

  constructor(args: {
    request_id: string
    external_req_id?: string
    prompt_token_ids: number[]
    sampling_params: SamplingParams
    arrival_time: number
    priority?: number
    client_index?: number
    block_hasher?: BlockHasher | null
  }) {
    this.request_id = args.request_id
    this.external_req_id = args.external_req_id ?? args.request_id
    this.client_index = args.client_index ?? 0
    this.priority = args.priority ?? 0
    this.sampling_params = args.sampling_params
    this.arrival_time = args.arrival_time
    this.max_tokens = args.sampling_params.max_tokens
    this.status = this.use_structured_output
      ? RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
      : RequestStatus.WAITING
    this.prompt_token_ids = args.prompt_token_ids
    this.num_prompt_tokens = args.prompt_token_ids.length
    this._all_token_ids = args.prompt_token_ids.slice()
    this._block_hasher = args.block_hasher ?? null
    this.update_block_hashes()
  }

  static from_engine_core_request(req: EngineCoreRequest, block_hasher: BlockHasher | null): Request {
    return new Request({
      request_id: req.request_id,
      external_req_id: req.external_req_id,
      prompt_token_ids: req.prompt_token_ids,
      sampling_params: req.sampling_params,
      arrival_time: req.arrival_time,
      priority: req.priority,
      client_index: req.client_index,
      block_hasher,
    })
  }

  append_output_token_ids(token_ids: number | number[]): void {
    if (typeof token_ids === 'number') {
      this._output_token_ids.push(token_ids)
      this._all_token_ids.push(token_ids)
    } else {
      this._output_token_ids.push(...token_ids)
      this._all_token_ids.push(...token_ids)
    }
    this.update_block_hashes()
  }

  update_block_hashes(): void {
    if (this._block_hasher !== null) {
      this.block_hashes.push(...this._block_hasher(this))
    }
  }

  get use_structured_output(): boolean {
    return this.sampling_params.guided_choice !== undefined
  }

  get output_token_ids(): readonly number[] {
    return this._output_token_ids
  }

  get all_token_ids(): readonly number[] {
    return this._all_token_ids
  }

  get num_tokens(): number {
    return this._all_token_ids.length
  }

  get num_tokens_with_spec(): number {
    return this._all_token_ids.length + this.spec_token_ids.length
  }

  get num_output_tokens(): number {
    return this._output_token_ids.length
  }

  is_finished(): boolean {
    return isFinishedStatus(this.status)
  }

  get_finished_reason(): FinishReason | null {
    return FINISHED_REASON_MAP[this.status] ?? null
  }

  /** Python `__lt__`: priority, then arrival_time, then request_id. */
  static compare(a: Request, b: Request): number {
    if (a.priority !== b.priority) return a.priority - b.priority
    if (a.arrival_time !== b.arrival_time) return a.arrival_time - b.arrival_time
    return a.request_id < b.request_id ? -1 : a.request_id > b.request_id ? 1 : 0
  }
}
