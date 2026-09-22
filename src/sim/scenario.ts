// Scenario definitions: engine config knobs + requests with arrival steps.
// Presets mirror the blog's worked examples and the differential fixtures.

import type { EngineConfig } from './engine_core'

export interface ScenarioRequest {
  id: string
  prompt: string
  arrival_step: number
  max_tokens: number
  priority?: number
  /** Structured output: the model must answer with one of these strings. */
  guided_choice?: string[]
  /** Scripted continuation (the "model" emits these tokens, then EOS). */
  continuation?: string
}

export interface Scenario {
  name: string
  title: string
  description: string
  config: EngineConfig
  requests: ScenarioRequest[]
  max_steps: number
}

export const BASE_CONFIG: EngineConfig = {
  max_num_seqs: 4,
  max_num_batched_tokens: 32,
  max_model_len: 128,
  enable_chunked_prefill: true,
  long_prefill_token_threshold: 0,
  policy: 'fcfs',
  scheduler_reserve_full_isl: true,
  async_scheduling: false,
  block_size: 4,
  num_gpu_blocks: 17,
  enable_prefix_caching: true,
  num_speculative_tokens: 0,
  num_lookahead_tokens: 0,
  spec: null,
  batch_queue_size: 1,
  grammar_compile_delay_steps: 1,
  seed: 7,
}

const FOX = 'The quick brown fox jumps over the lazy dog.'

export const PRESETS: Scenario[] = [
  {
    name: 'basic',
    title: 'Two prompts, one step',
    description: 'The blog\'s opening example: two prompts arrive together, are prefilled in the same step, then decode together as one flattened batch.',
    config: { ...BASE_CONFIG },
    requests: [
      { id: 'A', prompt: 'Hello, my name is', arrival_step: 0, max_tokens: 4, continuation: 'Aleksa and I like GPUs' },
      { id: 'B', prompt: 'The president of the United States is', arrival_step: 0, max_tokens: 3, continuation: 'not a GPU' },
    ],
    max_steps: 12,
  },
  {
    name: 'continuous_batching',
    title: 'Continuous batching',
    description: 'Requests arrive on different steps. A new prefill joins the decode batch mid-flight; finished requests leave and their blocks are recycled.',
    config: { ...BASE_CONFIG, max_num_batched_tokens: 16 },
    requests: [
      { id: 'A', prompt: 'Hi, my name is', arrival_step: 0, max_tokens: 5, continuation: 'Woosuk and I wrote vLLM' },
      { id: 'B', prompt: 'Today is a beautiful summer day', arrival_step: 1, max_tokens: 3, continuation: 'in Berkeley California' },
      { id: 'C', prompt: 'Hello there', arrival_step: 1, max_tokens: 4, continuation: 'general Kenobi !' },
      { id: 'D', prompt: 'Paged attention splits the KV cache into', arrival_step: 4, max_tokens: 3, continuation: 'fixed size blocks' },
    ],
    max_steps: 16,
  },
  {
    name: 'chunked_prefill',
    title: 'Chunked prefill',
    description: 'One long prompt against a small token budget: the prefill is split across steps and only the last chunk samples a token (blog Figure 5).',
    config: { ...BASE_CONFIG, max_num_batched_tokens: 8, enable_prefix_caching: false },
    requests: [
      { id: 'A', prompt: 'One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen', arrival_step: 0, max_tokens: 2, continuation: 'eighteen nineteen' },
      { id: 'B', prompt: 'short one', arrival_step: 1, max_tokens: 2, continuation: 'yes indeed' },
    ],
    max_steps: 12,
  },
  {
    name: 'prefix_cache',
    title: 'Prefix caching',
    description: 'Two prompts share an 8-token prefix. The second request hits two cached blocks (hash chain), touches them, and only computes its own tail (blog Figures 6-8).',
    config: { ...BASE_CONFIG, num_gpu_blocks: 17 },
    requests: [
      { id: 'A', prompt: 'Today is a nice and warm summer day! My name is', arrival_step: 0, max_tokens: 2, continuation: 'Aleksa' },
      { id: 'B', prompt: 'Today is a nice and warm summer day! His name is', arrival_step: 3, max_tokens: 2, continuation: 'Woosuk' },
    ],
    max_steps: 12,
  },
  {
    name: 'preemption',
    title: 'Preemption (recompute)',
    description: 'A tiny KV cache. When a decode needs a block and none is free, the last running request is preempted: its blocks are freed, num_computed_tokens resets to 0, and it re-prefills later.',
    config: { ...BASE_CONFIG, num_gpu_blocks: 7, enable_prefix_caching: false, scheduler_reserve_full_isl: false, max_num_batched_tokens: 64 },
    requests: [
      { id: 'A', prompt: 'alpha beta gamma delta eps', arrival_step: 0, max_tokens: 6, continuation: 'a1 a2 a3 a4 a5 a6' },
      { id: 'B', prompt: 'one two three four five', arrival_step: 0, max_tokens: 6, continuation: 'b1 b2 b3 b4 b5 b6' },
      { id: 'C', prompt: 'red green blue cyan pink', arrival_step: 0, max_tokens: 6, continuation: 'c1 c2 c3 c4 c5 c6' },
    ],
    max_steps: 24,
  },
  {
    name: 'priority',
    title: 'Priority scheduling',
    description: 'Same pressure as the preemption scenario, but with policy=priority: the waiting queue is a heap and the victim is max(priority, arrival_time), not the last running request.',
    config: { ...BASE_CONFIG, num_gpu_blocks: 7, enable_prefix_caching: false, scheduler_reserve_full_isl: false, max_num_batched_tokens: 64, policy: 'priority' },
    requests: [
      { id: 'low', prompt: 'alpha beta gamma delta eps', arrival_step: 0, max_tokens: 6, priority: 5, continuation: 'a1 a2 a3 a4 a5 a6' },
      { id: 'hi', prompt: 'one two three four five', arrival_step: 0, max_tokens: 6, priority: 0, continuation: 'b1 b2 b3 b4 b5 b6' },
      { id: 'mid', prompt: 'red green blue cyan pink', arrival_step: 0, max_tokens: 6, priority: 2, continuation: 'c1 c2 c3 c4 c5 c6' },
    ],
    max_steps: 24,
  },
  {
    name: 'spec_decode',
    title: 'Speculative decoding (n-gram)',
    description: 'The n-gram drafter proposes k tokens by matching the suffix against earlier text; the target model verifies all k+1 positions in one forward pass and the rejection sampler keeps the accepted prefix.',
    config: {
      ...BASE_CONFIG,
      enable_prefix_caching: false,
      num_speculative_tokens: 3,
      spec: { method: 'ngram', num_speculative_tokens: 3, prompt_lookup_min: 2, prompt_lookup_max: 4, acceptance_rate: null },
    },
    requests: [{ id: 'A', prompt: `${FOX} ${FOX} The quick`, arrival_step: 0, max_tokens: 12, continuation: `brown fox jumps over the lazy dog. The quick brown cat sleeps` }],
    max_steps: 16,
  },
  {
    name: 'structured_output',
    title: 'Structured output (grammar bitmask)',
    description: 'A choice grammar. The request waits for its grammar to compile, then every sampling step masks disallowed logits to -inf with a packed 32-bit bitmask computed on the CPU while the GPU runs.',
    config: { ...BASE_CONFIG, enable_prefix_caching: false },
    requests: [
      { id: 'A', prompt: 'This sucks . Sentiment :', arrival_step: 0, max_tokens: 3, guided_choice: ['Positive', 'Negative'], continuation: 'Meh whatever' },
      { id: 'B', prompt: 'The weather is beautiful . Sentiment :', arrival_step: 0, max_tokens: 3, guided_choice: ['Positive', 'Negative'], continuation: 'Positive' },
    ],
    max_steps: 12,
  },
  {
    name: 'async',
    title: 'Async scheduling + batch queue',
    description: 'Production default: step N+1 is scheduled with -1 placeholders before step N has sampled. The batch queue holds the in-flight step; update_from_output settles placeholders one step late.',
    config: { ...BASE_CONFIG, async_scheduling: true, batch_queue_size: 2, enable_prefix_caching: false },
    requests: [
      { id: 'A', prompt: 'Hello, my name is', arrival_step: 0, max_tokens: 3, continuation: 'Aleksa Gordic' },
      { id: 'B', prompt: 'The president of the United States is', arrival_step: 0, max_tokens: 3, continuation: 'not a GPU' },
    ],
    max_steps: 12,
  },
]

export function preset(name: string): Scenario {
  const p = PRESETS.find((s) => s.name === name)
  if (!p) throw new Error(`unknown preset ${name}`)
  return structuredClone(p)
}
