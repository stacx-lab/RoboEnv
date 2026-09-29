import assert from "node:assert/strict"
import { test } from "node:test"
import { zstdCompressSync } from "node:zlib"
import { requestEvidence, requestText } from "./provider-evidence.ts"

const expected = { id: "gpt-6-astra", effort: "medium" }
const image = { type: "input_image", detail: "auto", image_url: "data:image/png;base64,iVBORw0KGgo=" }
const body = {
  model: expected.id, reasoning: { effort: expected.effort },
  input: [{ role: "user", content: [image] }],
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

test("tool results deliver camera images in the function call output", () => {
  const input = [
    { type: "function_call_output", call_id: "call", output: [{ type: "input_text", text: "{}" }, image] },
    { type: "function_call_output", call_id: "text", output: "no image" },
  ]
  assert.equal(requestEvidence(JSON.stringify({ ...body, input }), expected).images.length, 1)
})

test("provider evidence rejects substituted configuration and remote-only images", () => {
  assert.throws(() => requestEvidence(JSON.stringify({ ...body, model: "another-model" }), expected), /authorized/)
  assert.throws(() => requestEvidence(JSON.stringify({ ...body, reasoning: { effort: "high" } }), expected), /authorized/)
  assert.throws(() => requestEvidence(JSON.stringify({ ...body, input: [{ content: [
    { type: "input_image", image_url: "https://example.com/picture.png" },
  ] }] }), expected), /encoded sensor bytes/)
})

test("request bodies are read as sent by the OpenAI SDK and the Codex backend", () => {
  const text = JSON.stringify(body)
  assert.equal(requestText({ body: text }), text)
  const compressed = new Uint8Array(zstdCompressSync(text))
  assert.equal(requestText({ body: compressed, headers: { "content-encoding": "zstd" } }), text)
  assert.throws(() => requestText({ body: text, headers: { "content-encoding": "gzip" } }), /JSON provider request/)
  assert.throws(() => requestText(undefined), /JSON provider request/)
})
