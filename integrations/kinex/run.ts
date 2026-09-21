// Harbor agent-side composition of the normal Kinex tool/session/recording APIs.
import path from "node:path"
import { randomUUID } from "node:crypto"
import fs from "node:fs/promises"
import { z } from "zod"
import { deactivateDevice } from "@kinex/core/device/dispatch"
import { observeDevices } from "@kinex/core/device/observe"
import { pythonTools } from "@kinex/core/tool/python"
import { execute, type Context } from "@kinex/core/tool/tool"
import { Recorder } from "@kinex/core/record/recorder"
import { createKinexRuntime, type KinexRuntime } from "@kinex/core"
import { createApplication } from "../../dist/application.js"
import { pythonEnvironment } from "@kinex/core/python/environment"
import { ConfigPaths } from "@kinex/core/config/paths"
import { Global } from "@kinex/core/global"
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
  model: z.strictObject({ provider: z.literal("openai"), id: z.string(), effort: z.enum(["off", "minimal", "low", "medium", "high", "xhigh"]) }).optional(),
})

async function main() {
  if (process.argv[2] === "--prepare-python") {
    const directory = process.argv[3]
    if (!directory) throw new Error("Workspace directory is required")
    const runtime = createKinexRuntime({ directory })
    try { await runtime.python.prepare() }
    finally { await runtime.dispose() }
    return
  }
  if (process.argv[2] === "--help") {
    console.log("node integrations/roboenv/run.ts RUN.json\nRun inside the packaged Kinex tree with public per-episode devices.")
    return
  }
  const input = Input.parse(JSON.parse(await fs.readFile(process.argv[2] ?? "", "utf8")))
  const directory = path.resolve(input.workspace)
  if (await fs.access(input.output).then(() => true, () => false)) throw new Error("Preserve existing receipts; select a fresh output")
  if (input.mode === "develop" && (!input.prompt || !input.model)) throw new Error("Development needs an authorized exact model and prompt")
  if (input.mode === "no-model" && (input.model || input.prompt)) throw new Error("No-model execution cannot contain model instructions")
  const app = input.mode === "develop" ? await createApplication({ cwd: directory,
    model: `${input.model!.provider}/${input.model!.id}`, thinkingLevel: input.model!.effort,
  }) : undefined
  const runtime = app?.owner.runtime ?? createKinexRuntime({ directory })
  try { await runtime.run(() => executeRun(input, runtime, app)) }
  finally { if (app) await app.dispose(); else await runtime.dispose() }
}

async function executeRun(input: z.infer<typeof Input>, runtime: KinexRuntime, app?: Awaited<ReturnType<typeof createApplication>>) {
  const directory = runtime.directory
  const devices = await runtime.devices.list()
  if (!devices.length || devices.some((device) => device.type === "model")) throw new Error("Require native devices without undeclared model services")
  if (devices.some((device) => !device.loaded)) throw new Error("Prepare loaded device profiles before freezing")
  const roots = await ConfigPaths.directories(directory)
  if (roots.some((root) => ![path.join(directory, ".kinex"), path.resolve(Global.Path.config)].includes(path.resolve(root))))
    throw new Error("Workspace inherits undeclared parent content")
  const session = { id: app?.host.session.sessionId ?? randomUUID() }
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
  if (app) {
    app.owner.interaction.host.ask = context.ask
    app.owner.interaction.questions = context.questions
    await app.host.session.bindExtensions({ mode: "print", onError: error => { throw new Error(error.error) } })
  }
  const cancel = () => {
    abort.abort()
    void app?.host.session.abort()
    // Abort activation/Python preparation as well as an already running tool.
    // Cleanup errors are awaited again below and included in the receipt.
    void runtime.dispose().catch(() => {})
  }
  process.once("SIGTERM", cancel)
  const timer = setTimeout(cancel, input.timeout_s * 1000)
  const receipt: Record<string, unknown> = { schema: "kinex/simulator-execution/1", mode: input.mode,
    passed: false, session: session.id, timeout_s: input.timeout_s, max_steps: null, events,
    python: await pythonEnvironment(directory).then((value) => value.executable) }
  const originalFetch = globalThis.fetch
  const provider = input.mode === "develop" ? observeProvider(input.model!) : undefined
  if (provider) receipt.provider_requests = provider.requests
  try {
    const activations = await runtime.devices.warmLoaded()
    await Promise.all(activations.map((activation) => activation.ready))
    if (Object.keys(input.recording).length) {
      const started = await Recorder.start(directory, { sessionID: session.id,
        instruction: input.prompt ?? "Integration probe; task success is evaluated by the Harbor verifier",
        specs: input.recording })
      receipt.recording_started = started
      if (Object.keys(started.refusals).length || !Recorder.active()) throw new Error("Required recording streams were refused")
    }
    if (input.mode === "develop") {
      // observeProvider checks the actual serialized request's model and effort.
      await app!.host.session.prompt(input.prompt!)
      const last = app!.host.session.messages.findLast(message => message.role === "assistant")
      if (app!.owner.failure) throw app!.owner.failure
      if (last?.role === "assistant" && (last.stopReason === "error" || last.stopReason === "aborted"))
        throw new Error(last.errorMessage ?? last.stopReason)
      if (!provider!.requests.length) throw new Error("Development did not issue a real provider request")
      if (!provider!.requests.some((request) => request.images.length && request.status === 200))
        throw new Error("Development did not deliver sensor images to the live model")
    }
    if (input.mode === "no-model") {
      // This path has no provider call. Also reject accidental harness fetches.
      const disabled = () => { throw new Error("Network provider calls are disabled in no-model validation") }
      globalThis.fetch = async () => disabled()
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
    for (const finish of [() => app?.host.session.abort(), () => Recorder.stop({ endedBy: "error" })]) {
      try { await finish() } catch (error) { errors.push(String(error)) }
    }
    const cleanup = await Promise.allSettled(devices.map((device) => deactivateDevice(directory, device)))
    const failures = cleanup.filter((result) => result.status === "rejected")
    errors.push(...failures.map((result) => String(result.reason)))
    if (errors.length) {
      receipt.passed = false
      receipt.cleanup_errors = errors
    }
    if (app?.host.session.sessionFile) {
      try {
        const native = path.join(path.dirname(input.output), "session.jsonl")
        await fs.copyFile(app.host.session.sessionFile, native, fs.constants.COPYFILE_EXCL)
        receipt.native_session = native
      } catch (error) { receipt.passed = false; errors.push(String(error)); receipt.cleanup_errors = errors }
    }
    await (app ? app.dispose() : runtime.dispose()).catch((error: unknown) => {
      receipt.passed = false
      errors.push(String(error))
      receipt.cleanup_errors = errors
    })
    provider?.stop()
    globalThis.fetch = originalFetch
    process.removeListener("SIGTERM", cancel)
    unobserve()
    await fs.writeFile(input.output, JSON.stringify(receipt, null, 2) + "\n", { flag: "wx", mode: 0o600 })
  }
  if (receipt.passed !== true) throw new Error("Execution or cleanup failed; inspect the receipt")
}

await main()
