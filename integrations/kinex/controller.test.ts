import { describe, expect, test } from "bun:test"
import { z } from "zod"
import { define, type Context } from "../../src/tool/tool.ts"
import { controllerCompleted, executeController } from "./controller.ts"

const context: Context = {
  directory: "/unused", sessionID: "test", abort: new AbortController().signal,
  ask: async () => {}, questions: async () => ({}),
}

describe("frozen controller completion", () => {
  test("rejects returned failures, missing status and malformed output", () => {
    for (const output of ["{}", '{"completed":false}', '{"completed":true,"error":"stalled"}',
      '{"completed":1}', "not JSON", '{"completed":true}\ntruncated']) {
      expect(controllerCompleted({ title: "controller", output })).toBe(false)
    }
    expect(controllerCompleted(undefined)).toBe(false)
    expect(controllerCompleted({ title: "controller", output: '{"completed":true}', isError: true })).toBe(false)
    expect(controllerCompleted({ title: "controller", output: '{"completed":true,"error":null}' })).toBe(true)
  })

  test("checks the real JSON return before normal output truncation", async () => {
    const tool = define("controller", { description: "fixture", parameters: z.object({}),
      execute: async () => ({ title: "controller", output: JSON.stringify({
        completed: true, observations: "x".repeat(31_000),
      }) }),
    })
    const execution = await executeController(tool, {}, context)
    expect(execution.result.output).toContain("output truncated")
    expect(execution.raw?.output).not.toContain("output truncated")
    expect(execution.completed).toBe(true)
  })

  test("normal tool exception and argument checks remain authoritative", async () => {
    const tool = define("controller", { description: "fixture", parameters: z.object({ run: z.boolean() }),
      execute: async () => { throw new Error("controller stalled") },
    })
    for (const input of [{}, { run: true }]) {
      const execution = await executeController(tool, input, context)
      expect(execution.result.isError).toBe(true)
      expect(execution.completed).toBe(false)
    }
  })
})
