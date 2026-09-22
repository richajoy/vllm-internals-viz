// Port of vllm/v1/spec_decode/ngram_proposer.py
// (_find_longest_matched_ngram_and_propose_tokens). Finds the longest n-gram
// in [min_ngram, max_ngram] that matches the suffix of the context; on ties
// the EARLIEST occurrence in the sequence wins, and the k tokens following
// that occurrence become the draft.

export interface NgramConfig {
  prompt_lookup_min: number
  prompt_lookup_max: number
  num_speculative_tokens: number
  max_model_len: number
}

export interface NgramMatch {
  drafts: number[]
  /** Start index of the matched n-gram in the original sequence, or -1. */
  match_start: number
  ngram_len: number
}

export function find_longest_matched_ngram_and_propose_tokens(
  origin_tokens: readonly number[],
  min_ngram: number,
  max_ngram: number,
  max_model_len: number,
  k_in: number,
): NgramMatch {
  const total_token = origin_tokens.length
  const none: NgramMatch = { drafts: [], match_start: -1, ngram_len: 0 }
  if (total_token < min_ngram) return none
  let k = Math.min(k_in, max_model_len - total_token)
  if (k <= 0) return none

  const tokens = origin_tokens.slice().reverse()
  const lps = new Array<number>(max_ngram).fill(0)
  let longest_ngram = 0
  let position = 0
  let prev_lps = 0
  let i = 1
  while (i < total_token) {
    if (tokens[prev_lps] === tokens[i]) {
      prev_lps += 1
      if (prev_lps >= longest_ngram) {
        longest_ngram = prev_lps
        position = i
      }
      if (i < max_ngram) lps[i] = prev_lps
      if (prev_lps === max_ngram) prev_lps = lps[max_ngram - 1]
      i += 1
    } else if (prev_lps !== 0) {
      prev_lps = lps[prev_lps - 1]
    } else {
      i += 1
    }
  }
  if (longest_ngram < min_ngram) return none
  const start_position = total_token - 1 - position + longest_ngram
  k = Math.min(k, total_token - start_position)
  return {
    drafts: origin_tokens.slice(start_position, start_position + k),
    match_start: total_token - 1 - position,
    ngram_len: longest_ngram,
  }
}

export class NgramProposer {
  cfg: NgramConfig
  constructor(cfg: NgramConfig) {
    this.cfg = cfg
  }
  propose(context_token_ids: readonly number[]): NgramMatch {
    return find_longest_matched_ngram_and_propose_tokens(
      context_token_ids,
      this.cfg.prompt_lookup_min,
      this.cfg.prompt_lookup_max,
      this.cfg.max_model_len,
      this.cfg.num_speculative_tokens,
    )
  }
}
