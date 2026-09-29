// Observe the real request at the provider boundary without storing credentials.
import { createHash } from "node:crypto"
import { zstdDecompressSync } from "node:zlib"
import { z } from "zod"

const Body = z.object({
  model: z.string(),
  reasoning: z.object({ effort: z.string() }),
  input: z.array(z.unknown()),
})
// Images reach the model in user messages and, for tool results such as camera
// reads, in the output of the function call item.
const Item = z.object({
  content: z.array(z.unknown()).optional(),
  output: z.union([z.string(), z.array(z.unknown())]).optional(),
})
const Image = z.object({ type: z.literal("input_image"), image_url: z.string() })
// The Responses endpoints of OpenAI's API and of the ChatGPT Codex backend.
const Endpoint = /^https:\/\/(?:api\.openai\.com\/v1|chatgpt\.com\/backend-api\/codex)\/responses$/

export function requestEvidence(body: string, expected: { id: string; effort: string }) {
  const parsed = Body.parse(JSON.parse(body))
  if (parsed.model !== expected.id || parsed.reasoning.effort !== expected.effort)
    throw new Error("Actual provider request differs from the authorized model/effort")
  const images = parsed.input.flatMap((entry) => {
    const item = Item.safeParse(entry)
    if (!item.success) return []
    const parts = [...(item.data.content ?? []), ...(Array.isArray(item.data.output) ? item.data.output : [])]
    return parts.flatMap((part) => {
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

// The OpenAI SDK sends JSON text; Codex compresses its request with zstd.
export function requestText(init: RequestInit | undefined): string {
  const encoding = new Headers(init?.headers).get("content-encoding")
  const body = init?.body
  if (typeof body === "string" && !encoding) return body
  if (body instanceof Uint8Array && (!encoding || encoding === "zstd"))
    return (encoding ? zstdDecompressSync(body) : Buffer.from(body)).toString("utf8")
  throw new Error("Expected a JSON provider request body")
}

export function observeProvider(expected: { id: string; effort: string }) {
  const original = globalThis.fetch
  const requests: (ReturnType<typeof requestEvidence> & { status?: number })[] = []
  globalThis.fetch = async (input, init) => {
    const url = input instanceof Request ? input.url : String(input)
    if (!Endpoint.test(url)) return original(input, init)
    const record: ReturnType<typeof requestEvidence> & { status?: number } = requestEvidence(requestText(init), expected)
    requests.push(record)
    const response = await original(input, init)
    record.status = response.status
    return response
  }
  return { requests, stop: () => { globalThis.fetch = original } }
}
