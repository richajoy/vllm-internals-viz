// Port of vllm/v1/core/sched/request_queue.py.

import { Request } from './request'

export type SchedulingPolicy = 'fcfs' | 'priority'

export interface RequestQueue {
  add_request(request: Request): void
  pop_request(): Request
  peek_request(): Request
  prepend_request(request: Request): void
  prepend_requests(requests: RequestQueue): void
  remove_requests(requests: Iterable<Request>): void
  readonly length: number
  /** Requests in scheduling order (a copy). */
  toArray(): Request[]
}

export class FCFSRequestQueue implements RequestQueue {
  private q: Request[] = []

  add_request(request: Request): void {
    this.q.push(request)
  }
  pop_request(): Request {
    const r = this.q.shift()
    if (!r) throw new Error('pop from empty queue')
    return r
  }
  peek_request(): Request {
    if (this.q.length === 0) throw new Error('peek from empty queue')
    return this.q[0]
  }
  prepend_request(request: Request): void {
    this.q.unshift(request)
  }
  /** deque.extendleft: elements end up in reverse order of `requests`. */
  prepend_requests(requests: RequestQueue): void {
    for (const r of requests.toArray()) this.q.unshift(r)
  }
  remove_requests(requests: Iterable<Request>): void {
    const set = new Set(requests)
    this.q = this.q.filter((r) => !set.has(r))
  }
  get length(): number {
    return this.q.length
  }
  toArray(): Request[] {
    return this.q.slice()
  }
}

/** heapq ordered by Request.__lt__ (priority, arrival_time, request_id). */
export class PriorityRequestQueue implements RequestQueue {
  private heap: Request[] = []

  add_request(request: Request): void {
    this.heap.push(request)
    this.heap.sort(Request.compare)
  }
  pop_request(): Request {
    const r = this.heap.shift()
    if (!r) throw new Error('pop from empty heap')
    return r
  }
  peek_request(): Request {
    if (this.heap.length === 0) throw new Error('peek from empty heap')
    return this.heap[0]
  }
  prepend_request(request: Request): void {
    this.add_request(request)
  }
  prepend_requests(requests: RequestQueue): void {
    for (const r of requests.toArray()) this.add_request(r)
  }
  remove_requests(requests: Iterable<Request>): void {
    const set = new Set(requests)
    this.heap = this.heap.filter((r) => !set.has(r))
  }
  get length(): number {
    return this.heap.length
  }
  toArray(): Request[] {
    return this.heap.slice()
  }
}

export function create_request_queue(policy: SchedulingPolicy): RequestQueue {
  return policy === 'priority' ? new PriorityRequestQueue() : new FCFSRequestQueue()
}
