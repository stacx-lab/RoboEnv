import { describe, test } from "node:test"
import assert from "node:assert/strict"
import { z } from "zod"
import { define, type Context } from "@kinex/core/tool/tool"
import { controllerCompleted, executeController } from "./controller.ts"

const context: Context = {
  directory: "/unused", sessionID: "test", abort: new AbortController().signal,
  ask: async () => {}, questions: async () => ({}),
}

describe("frozen controller completion", () => {
  test("rejects returned failures, missing status and malformed output", () => {
    for (const output of ["{}", '{"completed":false}', '{"completed":true,"error":"stalled"}',
      '{"completed":1}', "not JSON", '{"completed":true}\ntruncated']) {
      assert.equal(controllerCompleted({ title: "controller", output }), false)
    }
    assert.equal(controllerCompleted(undefined), false)
    assert.equal(controllerCompleted({ title: "controller", output: '{"completed":true}', isError: true }), false)
    assert.equal(controllerCompleted({ title: "controller", output: '{"completed":true,"error":null}' }), true)
  })

  test("checks the real JSON return before normal output truncation", async () => {
    const tool = define("controller", { description: "fixture", parameters: z.object({}),
      execute: async () => ({ title: "controller", output: JSON.stringify({
        completed: true, observations: "x".repeat(31_000),
      }) }),
    })
    const execution = await executeController(tool, {}, context)
    assert.match(execution.result.output, /output truncated/)
    assert.doesNotMatch(execution.raw!.output, /output truncated/)
    assert.equal(execution.completed, true)
  })

  test("normal tool exception and argument checks remain authoritative", async () => {
    const tool = define("controller", { description: "fixture", parameters: z.object({ run: z.boolean() }),
      execute: async () => { throw new Error("controller stalled") },
    })
    for (const input of [{}, { run: true }]) {
      const execution = await executeController(tool, input, context)
      assert.equal(execution.result.isError, true)
      assert.equal(execution.completed, false)
    }
  })
})
