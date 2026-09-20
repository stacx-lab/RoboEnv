// Observe the real request at the provider boundary without storing credentials.
import { createHash } from "node:crypto"
import { z } from "zod"

const Body = z.object({
  model: z.string(),
  reasoning: z.object({ effort: z.string() }),
  input: z.array(z.unknown()),
})
const Message = z.object({ content: z.array(z.unknown()) })
const Image = z.object({ type: z.literal("input_image"), image_url: z.string() })

export function requestEvidence(body: string, expected: { id: string; effort: string }) {
  const parsed = Body.parse(JSON.parse(body))
  if (parsed.model !== expected.id || parsed.reasoning.effort !== expected.effort)
    throw new Error("Actual provider request differs from the authorized model/effort")
  const images = parsed.input.flatMap((item) => {
    const message = Message.safeParse(item)
    if (!message.success) return []
    return message.data.content.flatMap((part) => {
      const image = Image.safeParse(part)
      if (!image.success) return []
      const match = /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\r\n]+)$/.exec(image.data.image_url)
      if (!match) throw new Error("Model image must contain encoded sensor bytes")
      const bytes = Buffer.from(match[2]!, "base64")
      if (!bytes.length) throw new Error("Model image is empty")
      return [{ mime: match[1]!, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }]
    })
  })
  return { model: parsed.model, effort: parsed.reasoning.effort, images }
}

export function observeProvider(expected: { id: string; effort: string }) {
  const original = globalThis.fetch
  const apiBase = process.env["OPENAI_BASE_URL"] || process.env["OPENAI_API_BASE"] || "https://api.openai.com/v1"
  const endpoints = new Set([
    apiBase.replace(/\/+$/, "") + "/responses",
    "https://chatgpt.com/backend-api/codex/responses",
  ])
  const requests: (ReturnType<typeof requestEvidence> & { status?: number })[] = []
  const observed: typeof fetch = async (...args: Parameters<typeof fetch>) => {
    const [input, init] = args
    const url = input instanceof Request ? input.url : String(input)
    if (!endpoints.has(url))
      return original(...args)
    if (typeof init?.body !== "string") throw new Error("Expected a JSON provider request body")
    const record: ReturnType<typeof requestEvidence> & { status?: number } = requestEvidence(init.body, expected)
    requests.push(record)
    const response = await original(...args)
    record.status = response.status
    return response
  }
  globalThis.fetch = observed
  return { requests, stop: () => { globalThis.fetch = original } }
}
