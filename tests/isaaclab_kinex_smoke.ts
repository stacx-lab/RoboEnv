// Native acceptance: copied Go2 driver, local FK/IK, generic Kinex controller.
import path from "node:path"

const [kinexRoot, cwd, output] = process.argv.slice(2)
const { CalibrationStore } = await import(path.join(kinexRoot, "src/calibration/store.ts"))
const { dispatchDevice, activateDevice, deactivateDevice } = await import(path.join(kinexRoot, "src/device/dispatch.ts"))
const { pythonTools } = await import(path.join(kinexRoot, "src/tool/python.ts"))
const { execute } = await import(path.join(kinexRoot, "src/tool/tool.ts"))
const body = await CalibrationStore.readDevice(cwd, "body")
if (!body || body.type !== "legged") throw new Error("Missing legged body")
await activateDevice(cwd, body)
await CalibrationStore.writeDevice(cwd, { name: "body", loaded: true })
try {
  const initial = (await dispatchDevice(cwd, body, "read", {})).data
  const forward = (await dispatchDevice(cwd, body, "fk", { joints: initial.positions })).data.pose
  for (const [i, leg] of body.intrinsics.legs.entries()) {
    const measured = initial.frame_poses[leg.foot_frame]
    if (Math.hypot(...forward[i].position.map((v: number, j: number) => v - measured[j])) > .0001)
      throw new Error(`FK position differs from native ${leg.foot_frame}`)
    const dot = forward[i].orientation.reduce((sum: number, v: number, j: number) => sum + v*measured[j+3], 0)
    if (2*Math.acos(Math.min(1, Math.abs(dot))) > .001)
      throw new Error(`FK orientation differs from native ${leg.foot_frame}`)
  }
  const target = body.intrinsics.kinematics.model.joints.map((joint: { name: string }, i: number) =>
    body.intrinsics.home[joint.name] + (i % 3 === 1 ? .1 : i % 3 === 2 ? -.2 : 0))
  const pose = (await dispatchDevice(cwd, body, "fk", { joints: target })).data.pose
  await dispatchDevice(cwd, body, "ik", { pose, seed: initial.positions })
  const afterIK = (await dispatchDevice(cwd, body, "read", {})).data
  if (afterIK.sample_time_s !== initial.sample_time_s || JSON.stringify(afterIK.positions) !== JSON.stringify(initial.positions))
    throw new Error("FK/IK changed native state or time")
  const tool = (await pythonTools(cwd)).tools.find((tool: { id: string }) => tool.id === "servo_to")
  if (!tool) throw new Error("Missing servo_to")
  const result = await execute(tool, { device: "body", pose, joint_tolerance: .05,
    quiet_velocity: .05, stable_s: .5, max_steps: 150, timeout_s: 110 }, {
    directory: cwd, sessionID: "isaaclab-native", abort: new AbortController().signal,
    ask: async () => {}, questions: async () => ({}),
  })
  if (result.isError) throw new Error(result.output)
  const motion = JSON.parse(result.output)
  if (!motion.reached) throw new Error(result.output)
  if (Math.abs(motion.state.sample_time_s - initial.sample_time_s - motion.writes*.02) > 1e-8)
    throw new Error("Only writes may advance the native clock")
  await Bun.write(output, JSON.stringify({ fk_matches_native: true, ik_read_only: true, motion }, null, 2))
} finally {
  await deactivateDevice(cwd, body)
}
process.exit(0)
