// Exercise current Kinex discovery, dispatch, controller and recording without a model.
import path from "node:path"

const [kinexRoot, cwd, output] = process.argv.slice(2)
const { CalibrationStore } = await import(path.join(kinexRoot, "src/calibration/store.ts"))
const { dispatchDevice } = await import(path.join(kinexRoot, "src/device/dispatch.ts"))
const { listPyTools } = await import(path.join(kinexRoot, "src/tool/python.ts"))
const { runPyTool } = await import(path.join(kinexRoot, "src/tool/python-runner.ts"))
const { Recorder } = await import(path.join(kinexRoot, "src/record/recorder.ts"))
const { Session } = await import(path.join(kinexRoot, "src/session/index.ts"))
const devices = await CalibrationStore.listDevices(cwd)
if (devices.length !== 5) throw new Error(`Expected five devices, found ${devices.length}`)
const arm = devices.find((d: { name: string }) => d.name === "arm")
const gripper = devices.find((d: { name: string }) => d.name === "gripper")
const first = await dispatchDevice(cwd, arm, "read", {})
const target = [...first.data.positions]
target[0] += 0.05
const session = await Session.create(cwd)
const ctx = { sessionID: session.id, directory: cwd, abort: new AbortController().signal,
  ask: async () => {}, questions: async () => ({}) }
const found = await listPyTools(cwd)
const servo = found.tools.find((t: { manifest: { name: string } }) => t.manifest.name === "servo_to")
if (!servo) throw new Error("Current Kinex servo_to was not discovered")
const started = await Recorder.start(cwd, { sessionID: session.id,
  instruction: "LIBERO integration device smoke; no task success claim",
  specs: { camera: { rate: 3 }, camera_wrist: { rate: 3 }, arm: { rate: 5 } } })
try {
  const motion = await runPyTool(cwd, servo, { device: "arm", joints: target, timeout_s: 25 }, ctx)
  const arrived = JSON.parse(motion.output)
  if (!arrived.reached) throw new Error(motion.output)
  await dispatchDevice(cwd, gripper, "write", { target: [1] })
  await dispatchDevice(cwd, gripper, "stop", {})
  const snap = await Recorder.snap(cwd, { sessionID: session.id, devices: ["camera", "camera_wrist"] })
  await dispatchDevice(cwd, arm, "stop", {})
  const stopped = await Recorder.stop({ summary: "Device smoke passed; no task attempt" })
  await Bun.write(output, JSON.stringify({ passed: true, devices: devices.map((d: { name: string }) => d.name),
    motion: arrived, started, snap, recording: stopped }, null, 2))
} finally {
  await Recorder.stop()
  await dispatchDevice(cwd, arm, "stop", {})
  await dispatchDevice(cwd, gripper, "stop", {})
}

// Imported harness watchers are process-scoped; all device/recording cleanup is above.
process.exit(0)
