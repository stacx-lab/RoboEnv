import { test, mock } from "node:test"
import assert from "node:assert/strict"
import { requestEvidence, observeProvider } from "./provider-evidence.ts"

const expected = { id: "gpt-6-astra", effort: "medium" }
const body = {
  model: expected.id, reasoning: { effort: expected.effort },
  input: [{ role: "user", content: [{ type: "input_image", image_url: "data:image/png;base64,iVBORw0KGgo=" }] }],
}

test("actual model request evidence preserves image identity without credentials or pixels", () => {
  const evidence = requestEvidence(JSON.stringify(body), expected)
  assert.equal(evidence.model, expected.id)
  assert.equal(evidence.effort, expected.effort)
  assert.equal(evidence.images.length, 1)
  assert.equal(evidence.images[0]?.bytes, 8)
  assert.equal(evidence.images[0]?.sha256.length, 64)
  assert.doesNotMatch(JSON.stringify(evidence), /base64/)
})

test("Pi tool-result images retain the same evidence and reject remote references", () => {
  const input = [{ type: "function_call_output", call_id: "read-frame",
    output: body.input[0]!.content }]
  const evidence = requestEvidence(JSON.stringify({ ...body, input }), expected)
  assert.deepEqual(evidence, requestEvidence(JSON.stringify(body), expected))
  assert.throws(() => requestEvidence(JSON.stringify({ ...body, input: [{
    type: "function_call_output", call_id: "read-frame",
    output: [{ type: "input_image", image_url: "https://example.com/frame.png" }],
  }] }), expected), /encoded sensor bytes/)
})

test("provider evidence rejects substituted configuration and remote-only images", () => {
  assert.throws(() => requestEvidence(JSON.stringify({ ...body, model: "another-model" }), expected), /authorized/)
  assert.throws(() => requestEvidence(JSON.stringify({ ...body, reasoning: { effort: "high" } }), expected), /authorized/)
  assert.throws(() => requestEvidence(JSON.stringify({ ...body, input: [{ content: [
    { type: "input_image", image_url: "https://example.com/picture.png" },
  ] }] }), expected), /encoded sensor bytes/)
})

for (const variable of ["OPENAI_BASE_URL", "OPENAI_API_BASE"]) test(`provider evidence follows native ${variable} selection`, async () => {
  const previous = { base: process.env.OPENAI_BASE_URL, alias: process.env.OPENAI_API_BASE }
  delete process.env.OPENAI_BASE_URL
  delete process.env.OPENAI_API_BASE
  process.env[variable] = "http://gateway.example/v1/"
  const fetch = mock.method(globalThis, "fetch", async () => new Response("ok"))
  const observer = observeProvider(expected)
  try {
    await globalThis.fetch("http://gateway.example/v1/responses", { body: JSON.stringify(body) })
    await globalThis.fetch("http://sim:8123/status")
    assert.equal(observer.requests.length, 1)
    assert.equal(observer.requests[0]?.status, 200)
    assert.equal(observer.requests[0]?.model, expected.id)
    assert.equal(observer.requests[0]?.effort, expected.effort)
    await assert.rejects(globalThis.fetch("http://gateway.example/v1/responses", {
      body: JSON.stringify({ ...body, model: "wrong-model" }),
    }), /authorized/)
    assert.equal(fetch.mock.callCount(), 2)
  } finally {
    observer.stop()
    fetch.mock.restore()
    if (previous.base === undefined) delete process.env.OPENAI_BASE_URL
    if (previous.base !== undefined) process.env.OPENAI_BASE_URL = previous.base
    if (previous.alias === undefined) delete process.env.OPENAI_API_BASE
    if (previous.alias !== undefined) process.env.OPENAI_API_BASE = previous.alias
  }
})
