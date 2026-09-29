// Harbor agent-side composition of Kinex 0.9.0: the core runtime owns devices,
// recording and Python tools; the Kinex application runs live model sessions.
import path from "node:path"
import fs from "node:fs/promises"
import { randomUUID } from "node:crypto"
import { z } from "zod"
import { createKinexRuntime, type Context, type KinexRuntime } from "@kinex/core"
// The public runtime exposes neither content roots nor tool definitions, only
// model-facing schemas and results; the package exports these core modules.
import { ConfigPaths } from "@kinex/core/config/paths"
import { Global } from "@kinex/core/global"
import { pythonTools } from "@kinex/core/tool/python"
import { observeProvider } from "./provider-evidence.ts"
import { executeController } from "./controller.ts"

const Input = z.strictObject({
  mode: z.enum(["no-model", "develop"]),
  workspace: z.string(),
  output: z.string(),
  tool: z.string().default("integration_probe"),
  args: z.record(z.string(), z.unknown()).default({}),
  timeout_s: z.number().positive(),
  recording: z.record(z.string(), z.strictObject({ rate: z.number().positive() })),
  require_completion: z.boolean().default(false),
  prompt: z.string().optional(),
  // Kinex's thinking levels except off: development always states an effort.
  model: z.strictObject({
    provider: z.literal("openai"), id: z.string().min(1),
    effort: z.enum(["minimal", "low", "medium", "high", "xhigh"]),
  }).optional(),
})
type Input = z.infer<typeof Input>

const allowed = new Set(["device_act", "device_read", "python_tool", "python_tool_controller"])
type Receipt = Record<string, unknown>
type DeviceEvent = { device: string; action: string; error?: string; wall: number; latencyMs: number; origin?: string }

async function exists(file: string) {
  return fs.access(file).then(() => true, () => false)
}

// A failed stage must not prevent the remaining cleanup or its receipt.
async function settle(errors: string[], ...steps: (() => Promise<unknown>)[]) {
  for (const step of steps) {
    try { await step() } catch (error) { errors.push(String(error)) }
  }
}

// Kinex writes its content, settings and credentials below these roots; the
// user's defaults under $HOME must never be read or written by a trial.
function bridgeRoots() {
  for (const name of ["KINEX_CONFIG_DIR", "KINEX_DATA_DIR"])
    if (!process.env[name]) throw new Error(`${name} must select the bridge's own directory`)
}

async function checkRoots(directory: string) {
  const roots = await ConfigPaths.directories(directory)
  if (roots.some((root) => ![path.join(directory, ".kinex"), path.resolve(Global.Path.config)].includes(path.resolve(root))))
    throw new Error("Workspace inherits undeclared parent content")
}

// A headless session has no owner to approve requests, so Kinex's ask mode denies
// every permission these rules do not allow.
async function configureDevelopment() {
  const permission = Object.fromEntries([...allowed, "write", "edit", "shell"].map((name) => [name, "allow"]))
  await fs.mkdir(path.join(Global.Path.config, "agent"), { recursive: true })
  await fs.writeFile(path.join(Global.Path.config, "kinex.json"),
    JSON.stringify({ permission: { ...permission, calib_write: "deny", device_calibrate: "deny" } }, null, 2) + "\n", { flag: "wx" })
}

// The same environment as Kinex's own entrypoint (bin/kinex.mjs).
async function application() {
  process.env.PI_CODING_AGENT_DIR = path.join(Global.Path.config, "agent")
  process.env.PI_SKIP_VERSION_CHECK = "1"
  process.env.PI_TELEMETRY = "0"
  const [{ createApplication }, { runPrintMode }] = await Promise.all([
    import("../../dist/application.js"), import("@earendil-works/pi-coding-agent"),
  ])
  return { createApplication, runPrintMode }
}

async function prepareDevices(runtime: KinexRuntime, input: Input, sessionID: string, receipt: Receipt) {
  const devices = await runtime.devices.list()
  if (!devices.length || devices.some((device) => device.type === "model")) throw new Error("Require native devices without undeclared model services")
  if (devices.some((device) => !device.loaded)) throw new Error("Prepare loaded device profiles before freezing")
  await runtime.python.prepare()
  receipt.python = (await runtime.python.status()).executable
  await Promise.all((await runtime.devices.warmLoaded()).map((activation) => activation.ready))
  if (!Object.keys(input.recording).length) return undefined
  const started = await runtime.recording.start({ sessionID, specs: input.recording,
    instruction: input.prompt ?? "Integration probe; task success is evaluated by the Harbor verifier" })
  receipt.recording_started = started
  if (Object.keys(started.refusals).length || !(await runtime.recording.status())) throw new Error("Required recording streams were refused")
  return started.manifest.id
}

function observeDevices(runtime: KinexRuntime, events: DeviceEvent[]) {
  return runtime.subscribe((event) => {
    if (event.type !== "device") return
    const { device, action, error, wall, latencyMs, origin } = event.event
    events.push({ device: device.name, action, error, wall, latencyMs, origin })
  })
}

async function noModel(input: Input, directory: string, receipt: Receipt, abort: AbortSignal, cleanup: string[]) {
  // This path has no provider call. Also reject accidental harness fetches.
  globalThis.fetch = async () => { throw new Error("Network provider calls are disabled in no-model validation") }
  const context: Omit<Context, "sessionID" | "directory" | "abort"> = {
    ask: async (request) => {
      if (!allowed.has(request.permission)) throw new Error(`Permission unavailable in no-model: ${request.permission}`)
    },
    questions: async () => { throw new Error("Owner input is required; this run cannot answer agent questions") },
  }
  const runtime = createKinexRuntime({ directory, host: context })
  const sessionID = randomUUID()
  const events: DeviceEvent[] = []
  Object.assign(receipt, { session: sessionID, events })
  const unobserve = observeDevices(runtime, events)
  try {
    await prepareDevices(runtime, input, sessionID, receipt)
    // Normal discovery and execution, bound to this runtime's drivers and recorder.
    const found = await runtime.run(() => pythonTools(directory))
    if (found.problems.length) throw new Error(found.problems.join("\n"))
    const tool = found.tools.find((candidate) => candidate.id === input.tool)
    if (!tool) throw new Error(`Normal discovery did not find ${input.tool}`)
    const callID = randomUUID()
    runtime.observe({ type: "tool", callID, name: tool.id, status: "running" })
    const execution = await runtime.run(() => executeController(tool, input.args,
      { ...context, sessionID, directory, abort }))
    const result = execution.result
    runtime.observe({ type: "tool", callID, name: tool.id, status: result.isError ? "error" : "completed", isError: result.isError })
    receipt.tool_result = result
    if (input.require_completion) {
      receipt.raw_tool_result = execution.raw
      receipt.controller_completed = execution.completed
      if (!execution.completed) throw new Error("Controller did not return completed:true without an error")
    }
    if (result.isError) throw new Error(result.output)
    if (abort.aborted) throw new Error("Execution deadline exceeded")
    receipt.recording = await runtime.recording.stop()
  } finally {
    await settle(cleanup, () => runtime.recording.stop({ endedBy: "error" }), () => runtime.dispose())
    unobserve()
  }
}

async function develop(input: Input, directory: string, receipt: Receipt, abort: AbortSignal, cleanup: string[]) {
  const model = input.model!
  await configureDevelopment()
  // Observe before the application exists, so that every provider request is seen.
  const provider = observeProvider(model)
  receipt.provider_requests = provider.requests
  const { createApplication, runPrintMode } = await application()
  const events: DeviceEvent[] = []
  receipt.events = events
  const app = await createApplication({ cwd: directory, model: `${model.provider}/${model.id}`,
    thinkingLevel: model.effort, mode: "ask", report: (text: string) => console.error(text) })
  const cancel = () => { void app.host.session.abort() }
  abort.addEventListener("abort", cancel, { once: true })
  let unobserve = () => {}
  try {
    const runtime = app.owner.runtime
    if (!runtime) throw new Error("Kinex application did not open the workspace runtime")
    const session = app.host.session
    receipt.session = session.sessionId
    unobserve = observeDevices(runtime, events)
    const recording = await prepareDevices(runtime, input, session.sessionId, receipt)
    let final: { stopReason: string; errorMessage?: string } | undefined
    const unsubscribe = session.subscribe((event) => {
      if (event.type === "message_end" && "stopReason" in event.message) final = event.message
    })
    // Print mode ends by disposing the session; Kinex then stops the recording
    // and closes device drivers as it does for `kinex run --json`.
    const code = await runPrintMode(app.host, { mode: "json", initialMessage: input.prompt! }).finally(unsubscribe)
    if (abort.aborted) throw new Error("Execution deadline exceeded")
    if (code || app.owner.failure) throw new Error(app.owner.failure?.message ?? `Kinex print mode exited ${code}`)
    if (!final || ["error", "aborted"].includes(final.stopReason))
      throw new Error(`Development session ended without a final reply: ${final?.errorMessage ?? final?.stopReason ?? "none"}`)
    if (!provider.requests.length) throw new Error("Development did not issue a real provider request")
    if (!provider.requests.some((request) => request.images.length && request.status === 200))
      throw new Error("Development did not deliver sensor images to the live model")
    if (recording) {
      const manifest = JSON.parse(await fs.readFile(path.join(directory, ".kinex/recordings", recording, "recording.json"), "utf8"))
      receipt.recording = manifest
      if (manifest.status !== "stopped" || manifest.error) throw new Error("Recording did not stop cleanly")
    }
  } finally {
    abort.removeEventListener("abort", cancel)
    await settle(cleanup, () => app.dispose())
    unobserve()
    provider.stop()
  }
}

async function main() {
  const [option, argument] = process.argv.slice(2)
  if (option === "--help") {
    console.log("node integrations/roboenv/run.ts RUN.json\nnode integrations/roboenv/run.ts --prepare-python WORKSPACE\n"
      + "Run inside the packaged Kinex tree with public per-episode devices.")
    return
  }
  bridgeRoots()
  if (option === "--prepare-python") {
    if (!argument) throw new Error("Workspace directory is required")
    const runtime = createKinexRuntime({ directory: path.resolve(argument) })
    await runtime.python.prepare().finally(() => runtime.dispose())
    return
  }
  const input = Input.parse(JSON.parse(await fs.readFile(option ?? "", "utf8")))
  const directory = path.resolve(input.workspace)
  if (await exists(input.output)) throw new Error("Preserve existing receipts; select a fresh output")
  if (input.mode === "develop" && (!input.prompt || !input.model)) throw new Error("Development needs an authorized exact model and prompt")
  if (input.mode === "no-model" && (input.model || input.prompt)) throw new Error("No-model execution cannot contain model instructions")
  const receipt: Receipt = { schema: "kinex/simulator-execution/1", mode: input.mode, passed: false, timeout_s: input.timeout_s }
  const abort = new AbortController()
  const cancel = () => abort.abort()
  process.once("SIGTERM", cancel)
  const timer = setTimeout(cancel, input.timeout_s * 1000)
  const cleanup: string[] = []
  try {
    await checkRoots(directory)
    await (input.mode === "develop" ? develop : noModel)(input, directory, receipt, abort.signal, cleanup)
    receipt.passed = true
  } catch (error) {
    receipt.error = error instanceof Error ? error.message : String(error)
    throw error
  } finally {
    clearTimeout(timer)
    process.removeListener("SIGTERM", cancel)
    if (cleanup.length) {
      receipt.passed = false
      receipt.cleanup_errors = cleanup
    }
    await fs.writeFile(input.output, JSON.stringify(receipt, null, 2) + "\n", { flag: "wx", mode: 0o600 })
  }
  if (receipt.passed !== true) throw new Error("Execution or cleanup failed; inspect the receipt")
}

await main()
// Device and runtime cleanup above is explicit; imported file watchers are process scoped.
process.exit(0)
