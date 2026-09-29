/**
 * call-config unit tests: field-wise LlmCallConfig equality (the real-change
 * detector behind logged changed headers) and the deepFreeze ownership helper
 * the loop applies to every built request.
 */

import { describe, expect, it } from 'vitest'
import { deepFreeze } from '@deepseek-ai/dsh-util-values'
import { callConfigEquals, fitRequestMaxTokens, isAgentLoopRequest, markAgentLoopRequest } from '../src/call-config.ts'
import { CONTEXT_WINDOW_EXCEEDED_CODE } from '../src/error.ts'
import { ReasoningEffortId } from '../src/brand.ts'
import type { GenerateOptions } from '../src/types.ts'

describe('callConfigEquals', () => {
  it('compares every field, including the stop list element-wise', () => {
    const base = { provider: 'p', model: 'm' }
    expect(callConfigEquals(base, base)).toBe(true)
    expect(callConfigEquals(base, { provider: 'x', model: 'm' })).toBe(false)
    expect(callConfigEquals(base, { provider: 'p', model: 'x' })).toBe(false)
    expect(callConfigEquals({ ...base, reasoningEffort: ReasoningEffortId('high') }, base)).toBe(false)
    expect(callConfigEquals(
      { ...base, reasoningEffort: ReasoningEffortId('high') },
      { ...base, reasoningEffort: ReasoningEffortId('high') },
    )).toBe(true)
    expect(callConfigEquals({ ...base, temperature: 0.5 }, base)).toBe(false)
    expect(callConfigEquals({ ...base, maxTokens: 1 }, { ...base, maxTokens: 2 })).toBe(false)
    expect(callConfigEquals({ ...base, stop: ['a'] }, base)).toBe(false)
    expect(callConfigEquals({ ...base, stop: ['a'] }, { ...base, stop: ['a', 'b'] })).toBe(false)
    expect(callConfigEquals({ ...base, stop: ['a'] }, { ...base, stop: ['b'] })).toBe(false)
    expect(callConfigEquals({ ...base, stop: ['a', 'b'] }, { ...base, stop: ['a', 'b'] })).toBe(true)
  })
})

describe('fitRequestMaxTokens', () => {
  it('keeps a cap that fits beside the prompt', () => {
    expect(fitRequestMaxTokens(256_000, 1_000_000, 10_000)).toBe(256_000)
    expect(fitRequestMaxTokens(undefined, 1_000_000, 10_000)).toBeUndefined()
    expect(fitRequestMaxTokens(1_000_000, undefined, 10_000)).toBe(1_000_000)
  })

  it('shrinks a cap equal to the whole window down to the room the prompt leaves', () => {
    expect(fitRequestMaxTokens(1_000_000, 1_000_000, 200_000)).toBe(1_000_000 - 200_000 - 1_024)
  })

  it('fails as context overflow when the prompt already fills the window', () => {
    expect(() => fitRequestMaxTokens(1_000_000, 1_000_000, 1_000_000)).toThrow(expect.objectContaining({
      code: CONTEXT_WINDOW_EXCEEDED_CODE,
    }))
  })
})

describe('deepFreeze', () => {
  it('freezes nested structure in place and returns the same reference', () => {
    const value = { a: { b: [1, { c: 'x' }] } }
    const frozen = deepFreeze(value)
    expect(frozen).toBe(value)
    expect(Object.isFrozen(value)).toBe(true)
    expect(Object.isFrozen(value.a)).toBe(true)
    expect(Object.isFrozen(value.a.b)).toBe(true)
    expect(Object.isFrozen(value.a.b[1])).toBe(true)
    // ESM runs in strict mode: mutation throws rather than silently failing.
    expect(() => { (value.a.b[1] as { c: string }).c = 'y' }).toThrow(TypeError)
  })

  it('never freezes an AbortSignal: the live cancellation channel keeps working', () => {
    const controller = new AbortController()
    const request = deepFreeze({ model: 'm', signal: controller.signal })
    expect(Object.isFrozen(request)).toBe(true)
    expect(Object.isFrozen(controller.signal)).toBe(false)
    let fired = false
    controller.signal.addEventListener('abort', () => { fired = true }, { once: true })
    controller.abort('stop')
    expect(fired).toBe(true)
    expect(controller.signal.aborted).toBe(true)
  })

  it('passes primitives through and terminates on cycles', () => {
    expect(deepFreeze(42)).toBe(42)
    expect(deepFreeze(null)).toBeNull()
    const cyclic = { self: undefined as unknown }
    cyclic.self = cyclic
    deepFreeze(cyclic)
    expect(Object.isFrozen(cyclic)).toBe(true)
  })

  it('freezes nesting deeper than the JavaScript call stack', () => {
    const depth = 5_000
    const root: unknown[] = []
    let cursor = root
    for (let index = 0; index < depth; index++) {
      const child: unknown[] = []
      cursor.push(child)
      cursor = child
    }

    deepFreeze(root)

    cursor = root
    for (let index = 0; index < depth; index++) {
      expect(Object.isFrozen(cursor)).toBe(true)
      cursor = cursor[0] as unknown[]
    }
    expect(Object.isFrozen(cursor)).toBe(true)
  })
})

describe('agent-loop request identity', () => {
  it('marks only the exact request object and preserves its identity', () => {
    const request: GenerateOptions = {
      provider: 'mock',
      model: 'model',
      messages: [],
    }
    const copy = { ...request }

    expect(isAgentLoopRequest(request)).toBe(false)
    expect(markAgentLoopRequest(request)).toBe(request)
    expect(isAgentLoopRequest(request)).toBe(true)
    expect(isAgentLoopRequest(copy)).toBe(false)
  })
})
