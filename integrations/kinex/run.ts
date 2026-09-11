// Harbor agent-side composition of the normal Kinex tool/session/recording APIs.
import path from "node:path"
import fs from "node:fs/promises"
import { z } from "zod"
import { CalibrationStore } from "../../src/calibration/store.ts"
import { activateDevice, deactivateDevice } from "../../src/device/dispatch.ts"
import { observeDevices } from "../../src/device/observe.ts"
import { pythonTools } from "../../src/tool/python.ts"
import { execute, type Context } from "../../src/tool/tool.ts"
import { Recorder } from "../../src/record/recorder.ts"
import { Session } from "../../src/session/index.ts"
import { createRunner } from "../../src/session/runner.ts"
import { createBus } from "../../src/bus/index.ts"
import { toResponsesBody } from "../../src/provider/openai.ts"
import { pythonEnvironment } from "../../src/python/environment.ts"
import { ConfigPaths } from "../../src/config/paths.ts"
import { Global } from "../../src/global.ts"
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
  model: z.strictObject({ provider: z.literal("openai"), id: z.string(), effort: z.string() }).optional(),
})

async function main() {
  if (process.argv[2] === "--help") {
    console.log("bun integrations/roboenv/run.ts RUN.json\nRun inside the packaged Kinex tree with public per-episode devices.")
    return
  }
  const input = Input.parse(await Bun.file(process.argv[2] ?? "").json())
  const directory = path.resolve(input.workspace)
  if (await Bun.file(input.output).exists()) throw new Error("Preserve existing receipts; select a fresh output")
  if (input.mode === "develop" && (!input.prompt || !input.model)) throw new Error("Development needs an authorized exact model and prompt")
  if (input.mode === "no-model" && (input.model || input.prompt)) throw new Error("No-model execution cannot contain model instructions")
  if (!(await pythonEnvironment(directory)).ready || !(await Bun.file(path.join(directory, ".kinex/python/uv.lock")).exists()))
    throw new Error("Run kinex python before activation; the Python lock is mandatory")
  const devices = await CalibrationStore.listDevices(directory)
  if (!devices.length || devices.some((device) => device.type === "model")) throw new Error("Require native devices without undeclared model services")
  if (devices.some((device) => !device.loaded)) throw new Error("Prepare loaded device profiles before freezing")
  const roots = await ConfigPaths.directories(directory)
  if (roots.some((root) => ![path.join(directory, ".kinex"), path.resolve(Global.Path.config)].includes(path.resolve(root))))
    throw new Error("Workspace inherits undeclared parent content")
  const session = await Session.create(directory)
  const bus = createBus()
  Recorder.attach(bus)
  const abort = new AbortController()
  const events: { device: string; action: string; error?: string; wall: number; latencyMs: number; origin?: string }[] = []
  const unobserve = observeDevices((event) => {
    events.push({ device: event.device.name, action: event.action, error: event.error,
      wall: event.wall, latencyMs: event.latencyMs, origin: event.origin })
  })
  const allowed = new Set(["device_act", "device_read", "python_tool", "python_tool_controller"])
  const context: Context = {
    sessionID: session.id, directory, abort: abort.signal,
    ask: async (request) => {
      if (allowed.has(request.permission)) return
      if (input.mode === "develop" && ["write", "edit", "shell"].includes(request.permission)) return
      throw new Error(`Permission unavailable in ${input.mode}: ${request.permission}`)
    },
    questions: async () => { throw new Error("Owner input is required; this run cannot answer agent questions") },
  }
  const runner = createRunner(bus)
  const cancel = () => { abort.abort(); void runner.cancel(session.id) }
  process.once("SIGTERM", cancel)
  const timer = setTimeout(() => { abort.abort(); void runner.cancel(session.id) }, input.timeout_s * 1000)
  const receipt: Record<string, unknown> = { schema: "kinex/simulator-execution/1", mode: input.mode,
    passed: false, session: session.id, timeout_s: input.timeout_s, events,
    python: await pythonEnvironment(directory).then((value) => value.executable) }
  const provider = input.mode === "develop" ? observeProvider(input.model!) : undefined
  if (provider) receipt.provider_requests = provider.requests
  try {
    for (const device of devices) await activateDevice(directory, device)
    if (Object.keys(input.recording).length) {
      const started = await Recorder.start(directory, { sessionID: session.id,
        instruction: input.prompt ?? "Integration probe; task success is evaluated by the Harbor verifier",
        specs: input.recording })
      receipt.recording_started = started
      if (Object.keys(started.refusals).length || !Recorder.active()) throw new Error("Required recording streams were refused")
    }
    if (input.mode === "develop") {
      const model = input.model!
      // The same model/effort is passed to createRunner and checked at serialization.
      receipt.provider_request = toResponsesBody({ model: model.id, effort: model.effort,
        system: "configuration preflight", messages: [], tools: [] })
      await runner.submit(session.id, { directory, providerID: model.provider, model: model.id,
        effort: model.effort, ask: context.ask, questions: context.questions }, input.prompt!)
      await runner.wait(session.id)
      if (!provider!.requests.length) throw new Error("Development did not issue a real provider request")
      if (!provider!.requests.some((request) => request.images.length && request.status === 200))
        throw new Error("Development did not deliver sensor images to the live model")
    }
    if (input.mode === "no-model") {
      // This path has no provider call. Also reject accidental harness fetches.
      const disabled = () => { throw new Error("Network provider calls are disabled in no-model validation") }
      globalThis.fetch = Object.assign(async () => disabled(), { preconnect: disabled })
      const found = await pythonTools(directory)
      if (found.problems.length) throw new Error(found.problems.join("\n"))
      const tool = found.tools.find((candidate) => candidate.id === input.tool)
      if (!tool) throw new Error(`Normal discovery did not find ${input.tool}`)
      const execution = input.require_completion
        ? await executeController(tool, input.args, context)
        : { result: await execute(tool, input.args, context), raw: undefined, completed: false }
      const result = execution.result
      receipt.tool_result = result
      if (input.require_completion) {
        receipt.raw_tool_result = execution.raw
        receipt.controller_completed = execution.completed
        if (!execution.completed) throw new Error("Controller did not return completed:true without an error")
      }
      if (result.isError) throw new Error(result.output)
    }
    if (abort.signal.aborted) throw new Error("Execution deadline exceeded")
    receipt.recording = await Recorder.stop({ endedBy: "stop" })
    receipt.passed = true
  } catch (error) {
    receipt.error = error instanceof Error ? error.message : String(error)
    throw error
  } finally {
    clearTimeout(timer)
    const errors: string[] = []
    // A failed stage must not prevent the remaining cleanup or its receipt.
    for (const finish of [() => runner.cancel(session.id), () => Recorder.stop({ endedBy: "error" })]) {
      try { await finish() } catch (error) { errors.push(String(error)) }
    }
    const cleanup = await Promise.allSettled(devices.map((device) => deactivateDevice(directory, device)))
    const failures = cleanup.filter((result) => result.status === "rejected")
    errors.push(...failures.map((result) => String(result.reason)))
    if (errors.length) {
      receipt.passed = false
      receipt.cleanup_errors = errors
    }
    provider?.stop()
    process.removeListener("SIGTERM", cancel)
    unobserve()
    await fs.writeFile(input.output, JSON.stringify(receipt, null, 2) + "\n", { flag: "wx", mode: 0o600 })
  }
  if (receipt.passed !== true) throw new Error("Execution or cleanup failed; inspect the receipt")
}

await main()
// Device/recorder cleanup above is explicit; imported file watchers are process scoped.
process.exit(0)
