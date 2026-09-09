// CPU-only integration against RoboEnv's real worker and shared portable driver.
import path from "node:path"
const [root, cwd, python] = process.argv.slice(2)
const { CalibrationStore } = await import(path.join(root, "src/calibration/store.ts"))
const { activateDevice, deactivateDevice, dispatchDevice } = await import(path.join(root, "src/device/dispatch.ts"))
const { applyDeviceReload } = await import(path.join(root, "src/tui/command/device.ts"))
const { ToolManifest } = await import(path.join(root, "src/tool/python.ts"))
const { runPyTool } = await import(path.join(root, "src/tool/python-runner.ts"))
const { isServing } = await import(path.join(root, "src/device/driver.ts"))
const { Recorder } = await import(path.join(root, "src/record/recorder.ts"))
await CalibrationStore.writeManifest(cwd, { identity: { id: "cpu", name: "CPU", platform: "sim", kind: "arm" } })
const device = await CalibrationStore.readDevice(cwd, "controller")
if (!device.access.driver.persistent) throw new Error("Driver must default to persistent")
await Bun.write(path.join(cwd, "tool.py"), 'from kinex import device,finish\nfinish(device("controller"))\n')
const info = { directory: cwd, manifest: ToolManifest.parse({ name: "probe", kind: "perception",
  description: "CPU integration", interpreter: python, timeout_s: 5 }) }
const ctx = { directory: cwd, sessionID: "cpu", abort: new AbortController().signal,
  ask: async () => {}, questions: async () => ({}) }
try {
  await activateDevice(cwd, device)
  for (const _ of [0, 1]) {
    const result = await runPyTool(cwd, info, {}, ctx)
    if (result.isError) throw new Error(result.output)
  }
  const snap = await Recorder.snap(cwd, { sessionID: "cpu", devices: ["controller"] })
  if (!snap.readings.controller?.data || snap.readings.controller.error
    || Object.keys(snap.refusals).length) throw new Error(JSON.stringify(snap))
  if (!await Bun.file(path.join(snap.dir, "recording.json")).exists()) throw new Error("Missing recording")
  const reload = await applyDeviceReload(cwd, "controller")
  await reload.follow
  await dispatchDevice(cwd, device, "read", {})
} finally {
  await Recorder.stop()
  await deactivateDevice(cwd, device)
}
if (await isServing(cwd, "controller")) throw new Error("Driver survived unload")
