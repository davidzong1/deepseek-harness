/**
 * Conversation call configuration and freeze utilities. Provider routing,
 * model, reasoning effort, and sampling values are request-header state that
 * can affect cache reuse; request waterfalls replace them and the loop logs
 * changed snapshots instead of allowing silent per-call drift.
 * @module dsh-llm/call-config
 */

import type { GenerateOptions } from './types.ts'
import type { ReasoningEffortId } from './brand.ts'
import { CONTEXT_WINDOW_EXCEEDED_CODE, HarnessError } from './error.ts'

/** Process-local identities of request objects assembled by dsh-agent-loop. */
const AGENT_LOOP_REQUESTS = new WeakSet<GenerateOptions>()

// TODO(call-config-shape): Revisit which fields are epoch-level for cache reuse
// and where provider-specific request options belong.
/**
 * Provider, model, reasoning effort, and sampling scalars of one conversation's
 * requests. Every field maps 1:1 onto the same-named `GenerateOptions` field;
 * the loop builds requests from the logged header rather than accepting these
 * per call.
 */
export interface LlmCallConfig {
  provider: string
  model: string
  reasoningEffort?: ReasoningEffortId
  temperature?: number
  maxTokens?: number
  stop?: string[]
}

/**
 * Effective config fields supplied by exact-model adapter resolution rather
 * than by the caller's request proposal.
 */
export interface LlmCallConfigAdapterDefaults {
  reasoningEffort?: true
  maxTokens?: true
}

/**
 * Field-wise equality over {@link LlmCallConfig} — the comparison a caller
 * runs to decide whether a proposed configuration is a real change (worth a
 * logged header snapshot) or the held one restated.
 * @param a - one configuration.
 * @param b - the other.
 * @returns whether every field (including the `stop` list, element-wise) matches.
 */
export function callConfigEquals(a: LlmCallConfig, b: LlmCallConfig): boolean {
  if (
    a.provider !== b.provider
    || a.model !== b.model
    || a.reasoningEffort !== b.reasoningEffort
    || a.temperature !== b.temperature
    || a.maxTokens !== b.maxTokens
  ) return false
  if (a.stop === undefined || b.stop === undefined) return a.stop === b.stop
  return a.stop.length === b.stop.length && a.stop.every((s, i) => s === b.stop?.[i])
}

/**
 * Mark one exact request object as assembled by dsh-agent-loop.
 * @param request - loop-owned request envelope before LLM dispatch.
 * @returns the same request object marked as created by the process-local agent loop.
 */
export function markAgentLoopRequest<T extends GenerateOptions>(request: T): T {
  AGENT_LOOP_REQUESTS.add(request)
  return request
}

/**
 * Test whether the exact request object was assembled by dsh-agent-loop.
 * @param request - request envelope observed at the LLM waterfall.
 * @returns whether {@link markAgentLoopRequest} recorded this object.
 */
export function isAgentLoopRequest(request: GenerateOptions): boolean {
  return AGENT_LOOP_REQUESTS.has(request)
}

/**
 * Tokens kept back so a slightly low prompt measurement cannot ask for an
 * output cap the provider must reject as larger than the remaining window.
 */
const OUTPUT_FIT_MARGIN = 1_024

/**
 * Shrink a requested output cap so it fits beside the prompt in the context
 * window. The logged header keeps the configured cap; only the dispatched
 * request uses the fitted value. A prompt that already fills the window fails
 * as context overflow so recovery can compact instead of ending the turn as a
 * zero-output `max-tokens` stop.
 * @param requested - configured or adapter-default output cap.
 * @param contextWindow - model context capacity, when the adapter disclosed one.
 * @param promptTokens - tokens the prompt already occupies.
 * @returns the cap to send, or the request unchanged when either bound is absent.
 */
export function fitRequestMaxTokens(
  requested: number | undefined,
  contextWindow: number | undefined,
  promptTokens: number,
): number | undefined {
  if (requested === undefined || contextWindow === undefined) return requested
  if (!Number.isSafeInteger(requested) || requested < 1) return requested
  if (!Number.isSafeInteger(contextWindow) || contextWindow < 1) return requested
  const prompt = Number.isSafeInteger(promptTokens) && promptTokens > 0 ? promptTokens : 0
  const room = contextWindow - prompt - OUTPUT_FIT_MARGIN
  if (room < 1) {
    throw new HarnessError(
      `the ${contextWindow}-token context window has no room left for output after ${prompt} prompt tokens`,
      CONTEXT_WINDOW_EXCEEDED_CODE,
    )
  }
  return Math.min(requested, room)
}
