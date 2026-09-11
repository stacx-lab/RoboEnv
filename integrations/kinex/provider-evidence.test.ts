import { test, expect } from "bun:test"
import { requestEvidence } from "./provider-evidence.ts"

const expected = { id: "gpt-6-astra", effort: "medium" }
const body = {
  model: expected.id, reasoning: { effort: expected.effort },
  input: [{ role: "user", content: [{ type: "input_image", image_url: "data:image/png;base64,iVBORw0KGgo=" }] }],
}

test("actual model request evidence preserves image identity without credentials or pixels", () => {
  const evidence = requestEvidence(JSON.stringify(body), expected)
  expect(evidence.model).toBe(expected.id)
  expect(evidence.effort).toBe(expected.effort)
  expect(evidence.images).toHaveLength(1)
  expect(evidence.images[0]?.bytes).toBe(8)
  expect(evidence.images[0]?.sha256).toHaveLength(64)
  expect(JSON.stringify(evidence)).not.toContain("base64")
})

test("provider evidence rejects substituted configuration and remote-only images", () => {
  expect(() => requestEvidence(JSON.stringify({ ...body, model: "another-model" }), expected)).toThrow("authorized")
  expect(() => requestEvidence(JSON.stringify({ ...body, reasoning: { effort: "high" } }), expected)).toThrow("authorized")
  expect(() => requestEvidence(JSON.stringify({ ...body, input: [{ content: [
    { type: "input_image", image_url: "https://example.com/picture.png" },
  ] }] }), expected)).toThrow("encoded sensor bytes")
})
