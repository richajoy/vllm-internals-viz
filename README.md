# vLLM engine, step by step

**Live demo:** https://richajoy.github.io/vllm-internals-viz/

An interactive replay of vLLM's V1 engine loop: continuous batching, paged
attention, KV block allocation and freeing, the `free_block_queue` linked
list, slot mapping, prefix caching, chunked prefill, preemption, speculative
decoding, structured output and async scheduling. Every class, method and
field name on the page is a vLLM one, pinned to
[`vllm-project/vllm@adc3e03517`](https://github.com/vllm-project/vllm/tree/adc3e03517d2e7333a3bb2083bb4d394a2986876).

The scheduler, KV cache manager, block pool and async scheduler are ported to
TypeScript one module per vLLM file (`src/sim/`). They are checked against
vLLM's Python `Scheduler` by differential replay: `tools/trace_vllm.py` drives
`vllm.v1.core.sched.scheduler.Scheduler` with a mocked model and dumps
per-step traces to `tests/fixtures/`; `src/sim/differential.test.ts` requires
the TypeScript port to reproduce every block id, free-queue order, ref count,
status and placeholder count.

## Run

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # 44 tests incl. 9 differential fixtures
npm run typecheck && npm run lint
```

Keyboard: `←`/`→` phase, `↑`/`↓` step, `space` play. The URL hash
(`#s=<preset>&i=<snapshot>`) is a deep link.

## Regenerating fixtures

Requires the vLLM checkout at `~/dev/vllm` with CPU torch in its `.venv`
(`uv pip install --python .venv/bin/python torch -r requirements/common.txt`
was sufficient on macOS arm64; no compiled extensions are needed to import
the scheduler).

```bash
tools/gen_fixtures.sh    # runs every tools/scenarios/*.json -> tests/fixtures/
```

## What is simulated

Engine core: `Scheduler.schedule` / `update_from_output`, `AsyncScheduler`
placeholders, `KVCacheManager.allocate_slots` / `get_computed_blocks` /
`free`, `UnitaryKVCacheCoordinator`, `FullAttentionManager`, `BlockPool`
(lazy eviction, touch, hashless-first free order), `FreeKVCacheBlockQueue`,
chained block hashing, `step` / `step_with_batch_queue`, the n-gram
proposer's matching rule and the scheduler-side spec-decode rollback.

Model runner: the worker builds `input_ids`, `positions`, `slot_mapping`,
`query_start_loc`, `seq_lens`, `logits_indices` and a paged KV memory array.
The forward pass emits each request's scripted continuation with greedy
sampling instead of running attention.

Scope: a single full-attention KV group (no hybrid/Mamba, sliding window, KV
connectors, encoder inputs, LoRA, DP or PP). Block hashes are FNV-1a hex
rather than SHA-256 bytes.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`; the simulator ports vLLM (Apache 2.0).
