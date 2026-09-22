// UI-side helpers over the simulation result: request colours, phase order,
// cursor navigation.

import type { Phase } from '../sim/events'
import type { Snapshot } from '../sim/simulation'

export const PHASES: Phase[] = ['add_request', 'schedule', 'execute_model', 'grammar_bitmask', 'sample_tokens', 'update_from_output', 'post_step', 'output']

export const PHASE_TITLE: Record<Phase, string> = {
  add_request: 'add_request',
  schedule: 'schedule',
  execute_model: 'execute_model',
  grammar_bitmask: 'grammar_bitmask',
  sample_tokens: 'sample_tokens',
  update_from_output: 'update_from_output',
  post_step: 'post_step',
  output: 'output',
}

/** Stable colour per external request id, in first-seen order. */
export function requestColor(ext: string, order: string[]): string {
  const i = order.indexOf(ext)
  return `var(--req-${(i < 0 ? order.length : i) % 8})`
}

export function requestOrder(snapshots: Snapshot[]): string[] {
  const seen: string[] = []
  for (const s of snapshots) for (const ext of Object.values(s.id_map)) if (!seen.includes(ext)) seen.push(ext)
  return seen
}

export function extId(snap: Snapshot, internal: string): string {
  return snap.id_map[internal] ?? internal
}

/** Index of the first snapshot of the next/previous step relative to `i`. */
export function stepBoundary(snapshots: Snapshot[], i: number, dir: 1 | -1): number {
  const cur = snapshots[i].step
  if (dir === 1) {
    const j = snapshots.findIndex((s, k) => k > i && s.step > cur)
    return j === -1 ? snapshots.length - 1 : j
  }
  let j = i
  while (j > 0 && snapshots[j - 1].step === cur) j--
  if (j === i && j > 0) {
    const prev = snapshots[j - 1].step
    while (j > 0 && snapshots[j - 1].step === prev) j--
  }
  return j
}
