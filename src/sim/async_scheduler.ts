// Port of vllm/v1/core/sched/async_scheduler.py. With async scheduling the
// engine schedules step N+1 before step N's tokens exist; placeholders stand
// in for them and are settled in update_from_output.

import type { SchedulerOutput } from './output'
import { type Request, RequestStatus } from './request'
import { Scheduler } from './scheduler'
import { REF } from './source_refs'

export class AsyncScheduler extends Scheduler {
  private _spec_token_placeholders: number[] = []

  protected override _update_after_schedule(scheduler_output: SchedulerOutput): void {
    super._update_after_schedule(scheduler_output)
    const spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    this._spec_token_placeholders = Array(scheduler_output.num_spec_tokens_to_schedule).fill(-1)
    for (const req_id of Object.keys(scheduler_output.num_scheduled_tokens)) {
      const request = this.requests.get(req_id) as Request
      if (request.is_prefill_chunk) continue
      scheduler_output.pending_structured_output_tokens ||= request.use_structured_output && request.num_output_placeholders > 0
      const cur_num_spec_tokens = spec_decode_tokens[req_id]?.length ?? 0
      request.num_output_placeholders += this.num_sampled_tokens_per_step + cur_num_spec_tokens
      // Real draft ids are filled in worker-side; the scheduler only needs the count.
      request.spec_token_ids = this._spec_token_placeholders
      this.log.emit(
        'Scheduler',
        'placeholders',
        `async: ${req_id} num_output_placeholders += ${this.num_sampled_tokens_per_step + cur_num_spec_tokens} -> ${request.num_output_placeholders}${this._spec_token_placeholders.length ? `; spec_token_ids = ${this._spec_token_placeholders.length} x -1 placeholder` : ''}`,
        { request_id: req_id, num_output_placeholders: request.num_output_placeholders, spec_placeholders: this._spec_token_placeholders.length },
        REF.AsyncScheduler_update_after_schedule,
      )
    }
  }

  protected override _update_request_with_output(request: Request, new_token_ids: number[], is_stale = false): [number[], boolean] {
    const status_before_update = request.status
    const [ids, stopped] = super._update_request_with_output(request, new_token_ids)
    if (!is_stale) {
      request.num_output_placeholders -= ids.length
      if (request.num_output_placeholders < 0) throw new Error('placeholders underflow')
    }
    // Cache the now-real tokens; skip preempted requests.
    if (status_before_update === RequestStatus.RUNNING) {
      this.kv_cache_manager.cache_blocks(request, request.num_computed_tokens - request.num_output_placeholders)
    }
    this.log.emit(
      'Scheduler',
      'settle_placeholders',
      `async: ${request.request_id} received ${ids.length} real token(s), num_output_placeholders -> ${request.num_output_placeholders}`,
      { request_id: request.request_id, num_output_placeholders: request.num_output_placeholders },
      REF.AsyncScheduler_update_request_with_output,
    )
    return [ids, stopped]
  }
}
