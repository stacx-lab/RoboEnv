// SIMPLE through Kinex's normal tool runner, local IK, devices, and recording.
import path from "node:path"
const [kinexRoot, cwd, output, clockMode] = process.argv.slice(2)
const { CalibrationStore } = await import(path.join(kinexRoot, "src/calibration/store.ts"))
const { activateDevice, deactivateDevice } = await import(path.join(kinexRoot, "src/device/dispatch.ts"))
const { dispatchDevice } = await import(path.join(kinexRoot, "src/device/dispatch.ts"))
const { pythonTools } = await import(path.join(kinexRoot, "src/tool/python.ts"))
const { execute } = await import(path.join(kinexRoot, "src/tool/tool.ts"))
const { Recorder } = await import(path.join(kinexRoot, "src/record/recorder.ts"))
const { Session } = await import(path.join(kinexRoot, "src/session/index.ts"))
const devices = await CalibrationStore.listDevices(cwd)
if (devices.length !== 8) throw new Error(`Expected eight devices, found ${devices.length}`)
const active = devices
for (const d of active) {
  await activateDevice(cwd, d)
  await CalibrationStore.writeDevice(cwd, { name: d.name, loaded: true })
}
const names = ["base", "left_arm", "right_arm", "left_hand", "right_hand"]
async function readings() {
  const entries = await Promise.all(names.map(async name => {
    const device = await CalibrationStore.readDevice(cwd, name)
    if (!device) throw new Error(`Missing device ${name}`)
    const result = await dispatchDevice(cwd, device, "read", {})
    return [name, result.data]
  }))
  return Object.fromEntries(entries)
}
// Compose the measured torso and wrist poses into pelvis coordinates. These
// goals force Kinex IK to traverse the live base:torso frame on every iteration.
function compose(a: number[], b: number[]): number[] {
  const [x, y, z, w] = a.slice(3)
  const [u, v, s, t] = b.slice(3)
  const [px, py, pz] = b
  const cx = 2 * (y * pz - z * py)
  const cy = 2 * (z * px - x * pz)
  const cz = 2 * (x * py - y * px)
  return [a[0] + px + w * cx + y * cz - z * cy,
    a[1] + py + w * cy + z * cx - x * cz,
    a[2] + pz + w * cz + x * cy - y * cx,
    w * u + x * t + y * s - z * v,
    w * v - x * s + y * t + z * u,
    w * s + x * v - y * u + z * t,
    w * t - x * u - y * v - z * s]
}
const session = await Session.create(cwd)
try {
  const before = await readings()
  const tools = (await pythonTools(cwd)).tools
  const handTool = tools.find((t: { id: string }) => t.id === "hand_geometry_probe")
  if (!handTool) throw new Error("Missing hand geometry probe")
  const handResult = await execute(handTool, {}, {
    directory: cwd, sessionID: session.id, abort: new AbortController().signal,
    ask: async () => {}, questions: async () => ({}),
  })
  if (handResult.isError) throw new Error(handResult.output)
  const handGeometry = JSON.parse(handResult.output)
  if (!handGeometry.passed) throw new Error(handResult.output)
  const tool = tools.find((t: { id: string }) => t.id === "device_composition_probe")
  if (!tool) throw new Error("Missing device_composition_probe")
  if (!before.base.frame_poses?.torso) throw new Error("Missing measured torso frame")
  // Both exported arm bases coincide with torso_link; the chains include shoulders.
  const goals = Object.fromEntries(["left_arm", "right_arm"].map((n) =>
    [n, { frame: "base", pose: compose(before.base.frame_poses.torso,
      before[n].frame_poses.flange) }]))
  const started = await Recorder.start(cwd, { sessionID: session.id,
    instruction: "SIMPLE integration probe; no task success claim",
    specs: { camera_head_left: { rate: 3 }, camera_head_right: { rate: 3 }, base: { rate: 5 } } })
  const result = await execute(tool, { steps: 30, goals, body: {
    mode: "locomotion", velocity: { vx: 0, vy: 0, wz: 0 },
    posture: { height: .74, roll: .03, pitch: .05, yaw: .08 },
  } }, {
    directory: cwd, sessionID: session.id, abort: new AbortController().signal,
    ask: async () => {}, questions: async () => ({}),
  })
  if (result.isError) throw new Error(result.output)
  const motion = JSON.parse(result.output)
  if (motion.steps !== 30) throw new Error(result.output)
  const after = await readings()
  const elapsed = after.base.sample_time_s - before.base.sample_time_s
  if (clockMode === "sync" ? Math.abs(elapsed - 3.0) > 1e-8 : elapsed <= 0)
    throw new Error(`Unexpected simulation progress ${elapsed}`)
  await Recorder.snap(cwd, { sessionID: session.id, devices: ["camera_head_left", "camera_head_right"] })
  const recording = await Recorder.stop({ summary: "No-model SIMPLE probe" })
  await Bun.write(output, JSON.stringify({ passed: true, before, after, handGeometry, motion, started, recording }, null, 2))
} finally {
  await Recorder.stop()
  await Promise.all(active.map((d: unknown) => deactivateDevice(cwd, d)))
}
process.exit(0)
