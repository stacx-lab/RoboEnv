// SIMPLE through Kinex's normal tool runner, local IK, devices, and recording.
import path from "node:path"
import { writeFile } from "node:fs/promises"
const [kinexRoot, cwd, output, clockMode] = process.argv.slice(2)
const { createKinexRuntime } = await import(path.join(kinexRoot, "packages/core/dist/index.js"))
const runtime = createKinexRuntime({ directory: cwd, host: { ask: async () => {} } })
const names = ["base", "left_arm", "right_arm", "left_hand", "right_hand"]
async function readings() {
  const entries = await Promise.all(names.map(async name => {
    const result = await runtime.devices.dispatch(name, "read")
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
const sessionID = "simple-smoke"
try {
  const devices = await runtime.devices.list()
  if (devices.length !== 8) throw new Error(`Expected eight devices, found ${devices.length}`)
  for (const device of devices.filter((device) => device.type !== "world"))
    await (await runtime.devices.load(device.name)).ready
  const before = await readings()
  const handResult = await runtime.tools.call("hand_geometry_probe", {}, { sessionID })
  if (handResult.isError) throw new Error(handResult.output)
  const handGeometry = JSON.parse(handResult.output)
  if (!handGeometry.passed) throw new Error(handResult.output)
  if (!before.base.frame_poses?.torso) throw new Error("Missing measured torso frame")
  // Both exported arm bases coincide with torso_link; the chains include shoulders.
  const goals = Object.fromEntries(["left_arm", "right_arm"].map((n) =>
    [n, { frame: "base", pose: compose(before.base.frame_poses.torso,
      before[n].frame_poses.flange) }]))
  const started = await runtime.recording.start({ sessionID,
    instruction: "SIMPLE integration probe; no task success claim",
    specs: { camera_head_left: { rate: 3 }, camera_head_right: { rate: 3 }, base: { rate: 5 } } })
  const result = await runtime.tools.call("device_composition_probe", { steps: 30, goals, body: {
    mode: "locomotion", velocity: { vx: 0, vy: 0, wz: 0 },
    posture: { height: .74, roll: .03, pitch: .05, yaw: .08 },
  } }, { sessionID })
  if (result.isError) throw new Error(result.output)
  const motion = JSON.parse(result.output)
  if (motion.steps !== 30) throw new Error(result.output)
  const after = await readings()
  const elapsed = after.base.sample_time_s - before.base.sample_time_s
  if (clockMode === "sync" ? Math.abs(elapsed - 3.0) > 1e-8 : elapsed <= 0)
    throw new Error(`Unexpected simulation progress ${elapsed}`)
  await runtime.recording.snap({ sessionID, devices: ["camera_head_left", "camera_head_right"] })
  const recording = await runtime.recording.stop({ summary: "No-model SIMPLE probe" })
  await writeFile(output, JSON.stringify({ passed: true, before, after, handGeometry, motion, started, recording }, null, 2))
} finally {
  await runtime.dispose()
}
