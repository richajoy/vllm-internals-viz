// Deterministic word-level tokenizer so token ids, block hashes and slot
// mappings stay legible. Ids are assigned in first-seen order after the
// special tokens, mirroring the blog's `[1, 2, 3, 4, 5]` style examples.

export const PAD_ID = 0
export const BOS_ID = 1
export const EOS_ID = 2

export class ToyTokenizer {
  private vocab = new Map<string, number>([
    ['<pad>', PAD_ID],
    ['<s>', BOS_ID],
    ['</s>', EOS_ID],
  ])
  private inv: string[] = ['<pad>', '<s>', '</s>']

  get vocab_size(): number {
    return this.inv.length
  }

  token_to_id(tok: string): number {
    let id = this.vocab.get(tok)
    if (id === undefined) {
      id = this.inv.length
      this.vocab.set(tok, id)
      this.inv.push(tok)
    }
    return id
  }

  id_to_token(id: number): string {
    return this.inv[id] ?? `<${id}>`
  }

  /** Splits on whitespace and punctuation; keeps punctuation as tokens. */
  tokenize(text: string): string[] {
    return text.match(/[A-Za-z0-9_']+|[^\sA-Za-z0-9_']/g) ?? []
  }

  encode(text: string, add_bos = true): number[] {
    const ids = this.tokenize(text).map((t) => this.token_to_id(t))
    return add_bos ? [BOS_ID, ...ids] : ids
  }

  decode(ids: readonly number[]): string {
    return ids
      .filter((id) => id !== BOS_ID && id !== PAD_ID)
      .map((id) => this.id_to_token(id))
      .join(' ')
      .replace(/ ([,.!?;:])/g, '$1')
  }
}
