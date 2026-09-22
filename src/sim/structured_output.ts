// Toy stand-in for vllm/v1/structured_output/ (StructuredOutputManager with
// an xgrammar-style backend). Grammar = a choice between fixed strings; the
// "FSM" is the set of choices still consistent with the accepted prefix.
// Bitmask rows are packed 32-bit words exactly like xgrammar's.

import { type EventLog, NULL_LOG } from './events'
import type { Request } from './request'
import { REF } from './source_refs'
import { EOS_ID, type ToyTokenizer } from './tokenizer'

export class ChoiceGrammar {
  choices: number[][]
  accepted: number[] = []

  constructor(choices: number[][]) {
    this.choices = choices
  }

  private live(prefix: readonly number[]): number[][] {
    return this.choices.filter((c) => c.length >= prefix.length && prefix.every((t, i) => c[i] === t))
  }

  allowed_next(prefix: readonly number[] = this.accepted): Set<number> {
    const allowed = new Set<number>()
    for (const c of this.live(prefix)) {
      if (c.length === prefix.length) allowed.add(EOS_ID)
      else allowed.add(c[prefix.length])
    }
    return allowed
  }

  is_terminated(): boolean {
    return this.accepted.length > 0 && this.accepted[this.accepted.length - 1] === EOS_ID
  }

  accept_tokens(tokens: readonly number[]): boolean {
    const next = this.accepted.slice()
    for (const t of tokens) {
      if (!this.allowed_next(next).has(t)) return false
      next.push(t)
    }
    this.accepted = next
    return true
  }

  /** Truncate draft tokens at the first grammar violation. */
  validate_tokens(tokens: readonly number[]): number[] {
    const next = this.accepted.slice()
    const out: number[] = []
    for (const t of tokens) {
      if (!this.allowed_next(next).has(t)) break
      next.push(t)
      out.push(t)
    }
    return out
  }
}

export function pack_bitmask(allowed: Set<number>, vocab_size: number): number[] {
  const words = new Array<number>(Math.ceil(vocab_size / 32)).fill(0)
  for (const id of allowed) {
    if (id < vocab_size) words[id >> 5] |= 1 << (id & 31)
  }
  return words.map((w) => w >>> 0)
}

export function unpack_bitmask(words: readonly number[], vocab_size: number): boolean[] {
  return Array.from({ length: vocab_size }, (_, i) => ((words[i >> 5] >>> (i & 31)) & 1) === 1)
}

export class StructuredOutputManager {
  tokenizer: ToyTokenizer
  /** How many schedule() calls the "async compile" takes. */
  compile_delay_steps: number
  private grammars = new Map<string, ChoiceGrammar>()
  private ready_at = new Map<string, number>()
  private step = 0
  log: EventLog

  constructor(tokenizer: ToyTokenizer, compile_delay_steps = 1, log: EventLog = NULL_LOG) {
    this.tokenizer = tokenizer
    this.compile_delay_steps = compile_delay_steps
    this.log = log
  }

  /** Called from the engine each step so compile "finishes" asynchronously. */
  tick(step: number): void {
    this.step = step
  }

  grammar_init(request: Request): void {
    const choices = request.sampling_params.guided_choice ?? []
    this.grammars.set(request.request_id, new ChoiceGrammar(choices.map((c) => this.tokenizer.encode(c, false))))
    this.ready_at.set(request.request_id, this.step + this.compile_delay_steps)
    this.log.emit(
      'StructuredOutputManager',
      'grammar_init',
      `grammar_init(${request.request_id}): compile choice grammar [${choices.join(' | ')}] asynchronously (ready in ${this.compile_delay_steps} step(s))`,
      { request_id: request.request_id, choices },
      REF.StructuredOutputManager_grammar_bitmask,
    )
  }

  is_grammar_ready(request: Request): boolean {
    const at = this.ready_at.get(request.request_id)
    return at !== undefined && this.step >= at
  }

  should_advance(request: Request): boolean {
    return request.use_structured_output && this.is_grammar_ready(request) && !request.is_prefill_chunk
  }

  accept_tokens(request: Request, tokens: readonly number[]): boolean {
    const g = this.grammars.get(request.request_id)
    if (!g) return true
    const ok = g.accept_tokens(tokens)
    this.log.emit(
      'StructuredOutputManager',
      'accept_tokens',
      `${request.request_id}: grammar.accept_tokens([${tokens.join(', ')}]) -> ${ok ? 'advanced FSM' : 'REJECTED'}`,
      { request_id: request.request_id, tokens, ok },
      REF.StructuredOutputManager_grammar_bitmask,
    )
    return ok
  }

  validate_tokens(request: Request, tokens: readonly number[]): number[] {
    return this.grammars.get(request.request_id)?.validate_tokens(tokens) ?? tokens.slice()
  }

  allowed_next(request: Request): Set<number> {
    return this.grammars.get(request.request_id)?.allowed_next() ?? new Set()
  }

  /** One row per request plus one per scheduled draft token (speculative rows). */
  grammar_bitmask(requests: Map<string, Request>, request_ids: string[], spec_tokens: Record<string, number[]>): number[][] {
    const rows: number[][] = []
    for (const id of request_ids) {
      const g = this.grammars.get(id)
      const req = requests.get(id)
      if (!g || !req) continue
      const prefix = g.accepted.slice()
      rows.push(pack_bitmask(g.allowed_next(prefix), this.tokenizer.vocab_size))
      for (const t of spec_tokens[id] ?? []) {
        if (t < 0) break
        prefix.push(t)
        rows.push(pack_bitmask(g.allowed_next(prefix), this.tokenizer.vocab_size))
      }
    }
    return rows
  }

  remove(request_id: string): void {
    this.grammars.delete(request_id)
    this.ready_at.delete(request_id)
  }
}
