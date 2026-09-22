#!/usr/bin/env python
"""Differential-test tracer: drive the real vLLM V1 scheduler on a scenario.

Usage (from the vLLM checkout, with its .venv):

    cd $VLLM_ROOT && PYTHONHASHSEED=0 PYTHONPATH=. \\
        .venv/bin/python <repo>/tools/trace_vllm.py \\
        <scenario.json> <output.json>

Pinned against vllm-project/vllm @ adc3e03517. Uses the scheduler test
helpers in tests/v1/core/utils.py so the fixture reflects exactly what the
upstream unit tests exercise. No GPU, no compiled extensions.

Scenario schema (tools/scenarios/*.json):
  name: str
  config: block_size, num_gpu_blocks (block 0 is the null block, so usable =
          num_gpu_blocks-1), max_num_batched_tokens, max_num_seqs,
          enable_prefix_caching, enable_chunked_prefill,
          long_prefill_token_threshold, policy ("fcfs"|"priority"),
          async_scheduling, num_speculative_tokens (0 = off), max_model_len
  requests[]: request_id, prompt_token_ids, max_tokens, arrival_step,
          priority, output_token_ids (what the mock model samples, in order;
          EOS once exhausted), optional draft_token_ids_per_step [[int]]
          (pushed via scheduler.update_draft_token_ids after each update,
          for requests past prefill, one list per update)
  max_steps: int

Fixture layout per step:
  scheduler_output   -> SchedulerOutput fields (pre-schedule num_computed_tokens)
  after_schedule     -> state right after schedule(); request.num_computed_tokens
                        is ALREADY advanced by this step's scheduled tokens
                        (Scheduler._update_after_schedule)
  update_for_step    -> which step's SchedulerOutput update_from_output consumed
                        (== step for sync; step-1 for async, mirroring
                        EngineCore.step_with_batch_queue with queue size 2)
  model_runner_output, outputs -> mock samples and EngineCoreOutputs
  after_update       -> state right after update_from_output()
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# --- Guard: this must run under the vLLM .venv with PYTHONPATH=<vllm root>.
try:
    from tests.v1.core.utils import EOS_TOKEN_ID, create_scheduler
except ModuleNotFoundError as exc:  # pragma: no cover
    sys.stderr.write(
        f"import failed ({exc}). Run with PYTHONPATH=<vllm root> using "
        "<vllm root>/.venv/bin/python\n"
    )
    raise

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request

VLLM_COMMIT = "adc3e03517"


# ---------------------------------------------------------------------------
# Scheduler / request construction
# ---------------------------------------------------------------------------


def build_scheduler(cfg: dict[str, Any]) -> Scheduler:
    num_spec = int(cfg.get("num_speculative_tokens", 0))
    scheduler = create_scheduler(
        max_num_seqs=cfg["max_num_seqs"],
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        enable_chunked_prefill=cfg.get("enable_chunked_prefill", True),
        enable_prefix_caching=cfg.get("enable_prefix_caching", False),
        long_prefill_token_threshold=cfg.get("long_prefill_token_threshold", 0),
        num_blocks=cfg["num_gpu_blocks"],
        block_size=cfg["block_size"],
        max_model_len=cfg.get("max_model_len"),
        num_speculative_tokens=num_spec if num_spec > 0 else None,
        async_scheduling=cfg.get("async_scheduling", False),
        use_v2_model_runner=False,
    )
    # create_scheduler has no policy knob; the scheduler reads policy only
    # through these three attributes, all set in Scheduler.__init__.
    policy = SchedulingPolicy(cfg.get("policy", "fcfs"))
    if policy != scheduler.policy:
        assert not scheduler.requests, "policy must be set before add_request"
        scheduler.policy = policy
        scheduler.waiting = create_request_queue(policy)
        scheduler.skipped_waiting = create_request_queue(policy)
    return scheduler


_none_hash_ready = False


def build_request(spec: dict[str, Any], block_size: int, arrival_time: float) -> Request:
    global _none_hash_ready
    if not _none_hash_ready:
        init_none_hash(sha256)
        _none_hash_ready = True
    sampling_params = SamplingParams(
        max_tokens=spec["max_tokens"],
        ignore_eos=spec.get("ignore_eos", False),
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=spec["request_id"],
        prompt_token_ids=list(spec["prompt_token_ids"]),
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=arrival_time,
        priority=int(spec.get("priority", 0)),
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def snapshot_scheduler_output(out: SchedulerOutput) -> dict[str, Any]:
    cached = out.scheduled_cached_reqs
    return {
        "scheduled_new_reqs": [
            {
                "req_id": r.req_id,
                "block_ids": [list(g) for g in r.block_ids],
                "num_computed_tokens": r.num_computed_tokens,
            }
            for r in out.scheduled_new_reqs
        ],
        "scheduled_cached_reqs": {
            "req_ids": list(cached.req_ids),
            "resumed_req_ids": sorted(cached.resumed_req_ids),
            "new_block_ids": [
                None if b is None else [list(g) for g in b] for b in cached.new_block_ids
            ],
            "num_computed_tokens": list(cached.num_computed_tokens),
            "num_output_tokens": list(cached.num_output_tokens),
        },
        "num_scheduled_tokens": dict(out.num_scheduled_tokens),
        "total_num_scheduled_tokens": out.total_num_scheduled_tokens,
        "scheduled_spec_decode_tokens": {
            k: list(v) for k, v in out.scheduled_spec_decode_tokens.items()
        },
        "num_common_prefix_blocks": list(out.num_common_prefix_blocks),
        "finished_req_ids": sorted(out.finished_req_ids),
        "preempted": sorted(out.preempted_req_ids or ()),
    }


def snapshot_state(scheduler: Scheduler) -> dict[str, Any]:
    pool = scheduler.kv_cache_manager.block_pool
    blocks = {
        str(b.block_id): {"ref_cnt": b.ref_cnt, "has_hash": b.block_hash is not None}
        for b in pool.blocks
        if not b.is_null
    }
    free_order = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]
    requests = {
        rid: {
            "status": req.status.name,
            "num_computed_tokens": req.num_computed_tokens,
            "num_tokens": req.num_tokens,
            "num_prompt_tokens": req.num_prompt_tokens,
            "num_output_tokens": req.num_output_tokens,
            "num_output_placeholders": req.num_output_placeholders,
            "spec_token_ids": list(req.spec_token_ids),
            "block_ids": [
                list(g) for g in scheduler.kv_cache_manager.get_block_ids(rid)
            ]
            if rid in scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks
            else [],
        }
        for rid, req in scheduler.requests.items()
    }
    return {
        "free_queue_order": free_order,
        "num_free_blocks": pool.get_num_free_blocks(),
        "blocks": blocks,
        "cached_hash_count": len(pool.cached_block_hash_to_block._cache),
        "requests": requests,
        "running_order": [r.request_id for r in scheduler.running],
        "waiting_order": [r.request_id for r in scheduler.waiting],
        "skipped_waiting_order": [r.request_id for r in scheduler.skipped_waiting],
    }


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockModel:
    """Emits the scenario's output tokens in order, honouring draft acceptance.

    For a request whose step scheduled draft tokens, the drafts are accepted
    while they match the scenario's upcoming output tokens; the sampled list is
    the accepted prefix plus one bonus token, matching how the scheduler
    derives num_accepted = len(sampled) - 1 (scheduler.py L1775 @ adc3e03517).
    Once a request's output_token_ids are exhausted, EOS is emitted.
    """

    def __init__(self, request_specs: dict[str, dict[str, Any]]):
        self.expected = {
            rid: list(s.get("output_token_ids", [])) for rid, s in request_specs.items()
        }
        self.cursor = {rid: 0 for rid in request_specs}

    def next_token(self, rid: str) -> int:
        exp = self.expected[rid]
        i = self.cursor[rid]
        tok = exp[i] if i < len(exp) else EOS_TOKEN_ID
        self.cursor[rid] = i + 1
        return tok

    def peek(self, rid: str, offset: int) -> int:
        exp = self.expected[rid]
        i = self.cursor[rid] + offset
        return exp[i] if i < len(exp) else EOS_TOKEN_ID

    def sample(self, rid: str, drafts: list[int]) -> list[int]:
        num_accepted = 0
        for d in drafts:
            if d != self.peek(rid, num_accepted):
                break
            num_accepted += 1
        return [self.next_token(rid) for _ in range(num_accepted + 1)]


def build_model_runner_output(
    out: SchedulerOutput, samples: dict[str, bool], model: MockModel
) -> ModelRunnerOutput:
    req_ids = list(out.num_scheduled_tokens.keys())
    sampled: list[list[int]] = []
    for rid in req_ids:
        if samples[rid]:
            drafts = out.scheduled_spec_decode_tokens.get(rid, [])
            sampled.append(model.sample(rid, list(drafts)))
        else:
            sampled.append([])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(scenario: dict[str, Any]) -> dict[str, Any]:
    cfg = scenario["config"]
    scheduler = build_scheduler(cfg)
    is_async = bool(cfg.get("async_scheduling", False))
    block_size = cfg["block_size"]

    request_specs = {r["request_id"]: r for r in scenario["requests"]}
    model = MockModel(request_specs)
    draft_queues = {
        rid: list(s.get("draft_token_ids_per_step", []))
        for rid, s in request_specs.items()
    }
    all_requests: dict[str, Request] = {}
    arrivals: dict[int, list[dict[str, Any]]] = {}
    for r in scenario["requests"]:
        arrivals.setdefault(int(r["arrival_step"]), []).append(r)
    last_arrival = max(arrivals) if arrivals else -1

    def push_drafts() -> None:
        ids, drafts = [], []
        for rid, q in draft_queues.items():
            req = scheduler.requests.get(rid)
            if req is None or req.is_finished() or req.is_prefill_chunk or not q:
                continue
            ids.append(rid)
            drafts.append(q.pop(0))
        if ids:
            scheduler.update_draft_token_ids(DraftTokenIds(ids, drafts))

    def update(out: SchedulerOutput, samples: dict[str, bool]) -> dict[str, Any]:
        mro = build_model_runner_output(out, samples, model)
        eco = scheduler.update_from_output(out, mro)
        outputs: dict[str, Any] = {}
        for client_outputs in (eco or {}).values():
            for o in client_outputs.outputs:
                outputs[o.request_id] = {
                    "new_token_ids": list(o.new_token_ids),
                    "finish_reason": None
                    if o.finish_reason is None
                    else str(o.finish_reason),
                }
        push_drafts()
        return {
            "model_runner_output": {
                "req_ids": mro.req_ids,
                "sampled_token_ids": mro.sampled_token_ids,
            },
            "outputs": outputs,
        }

    steps: list[dict[str, Any]] = []
    pending: tuple[int, SchedulerOutput, dict[str, bool]] | None = None
    max_steps = int(scenario.get("max_steps", 64))
    step = 0
    while step < max_steps:
        for spec in arrivals.get(step, []):
            req = build_request(spec, block_size, float(step))
            all_requests[req.request_id] = req
            scheduler.add_request(req)
        pending_has_work = (
            pending is not None and pending[1].total_num_scheduled_tokens > 0
        )
        if step > last_arrival and not scheduler.has_requests() and not pending_has_work:
            break

        record: dict[str, Any] = {"step": step, "arrivals": [
            r["request_id"] for r in arrivals.get(step, [])
        ]}
        out = scheduler.schedule()
        samples = {
            rid: not scheduler.requests[rid].is_prefill_chunk
            for rid in out.num_scheduled_tokens
        }
        record["scheduler_output"] = snapshot_scheduler_output(out)
        record["after_schedule"] = snapshot_state(scheduler)

        if is_async:
            # EngineCore.step_with_batch_queue with batch_queue_size=2:
            # schedule step N, then consume the output of step N-1.
            if pending is not None:
                prev_step, prev_out, prev_samples = pending
                record["update_for_step"] = prev_step
                record.update(update(prev_out, prev_samples))
            else:
                record["update_for_step"] = None
                record["outputs"] = {}
            pending = (step, out, samples)
        else:
            record["update_for_step"] = step
            record.update(update(out, samples))
        record["after_update"] = snapshot_state(scheduler)
        steps.append(record)
        step += 1

    if (
        is_async
        and pending is not None
        and pending[1].total_num_scheduled_tokens > 0
        and step < max_steps
    ):
        prev_step, prev_out, prev_samples = pending
        record = {"step": step, "arrivals": [], "scheduler_output": None,
                  "after_schedule": snapshot_state(scheduler),
                  "update_for_step": prev_step}
        record.update(update(prev_out, prev_samples))
        record["after_update"] = snapshot_state(scheduler)
        steps.append(record)

    final = {
        rid: {
            "status": req.status.name,
            "output_token_ids": list(req.output_token_ids),
            "num_output_tokens": req.num_output_tokens,
            "num_preemptions": req.num_preemptions,
        }
        for rid, req in all_requests.items()
    }
    return {
        "name": scenario["name"],
        "vllm_commit": VLLM_COMMIT,
        "scheduler_class": type(scheduler).__name__,
        "config": cfg,
        "requests": scenario["requests"],
        "num_steps": len(steps),
        "steps": steps,
        "final_requests": final,
        "unfinished": scheduler.get_num_unfinished_requests(),
    }


def summarize(trace: dict[str, Any]) -> str:
    lines = [f"[{trace['name']}] {trace['scheduler_class']} steps={trace['num_steps']} "
             f"unfinished={trace['unfinished']}"]
    for s in trace["steps"]:
        so = s["scheduler_output"]
        if so is None:
            lines.append(f"  step {s['step']}: (drain) update_for={s['update_for_step']}")
            continue
        new = [f"{r['req_id']}:{r['block_ids']}" for r in so["scheduled_new_reqs"]]
        lines.append(
            f"  step {s['step']}: sched={so['num_scheduled_tokens']} new={new} "
            f"cached={so['scheduled_cached_reqs']['req_ids']} "
            f"resumed={so['scheduled_cached_reqs']['resumed_req_ids']} "
            f"spec={so['scheduled_spec_decode_tokens']} "
            f"preempted={so['preempted']} finished={so['finished_req_ids']} "
            f"free={s['after_schedule']['num_free_blocks']}"
            f"->{s['after_update']['num_free_blocks']} "
            f"free_q={s['after_update']['free_queue_order']} "
            f"out={ {k: v['new_token_ids'] for k, v in s['outputs'].items()} }"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.stderr.write("warning: PYTHONHASHSEED != 0; fixtures may not be stable\n")
    with open(argv[1]) as f:
        scenario = json.load(f)
    trace = run(scenario)
    os.makedirs(os.path.dirname(os.path.abspath(argv[2])), exist_ok=True)
    with open(argv[2], "w") as f:
        json.dump(trace, f, indent=1, sort_keys=False)
        f.write("\n")
    print(summarize(trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
