// Explicit integration check: copied G1 drivers and the workspace controller.
import path from "node:path"

const [kinexRoot, cwd, output] = process.argv.slice(2)
const { CalibrationStore } = await import(path.join(kinexRoot, "src/calibration/store.ts"))
const { activateDevice, deactivateDevice } = await import(path.join(kinexRoot, "src/device/dispatch.ts"))
const { snapshotDevices } = await import(path.join(kinexRoot, "src/device/group.ts"))
const { pythonTools } = await import(path.join(kinexRoot, "src/tool/python.ts"))
const { execute } = await import(path.join(kinexRoot, "src/tool/tool.ts"))
const names = ["base", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
const devices = await Promise.all(names.map(async (name) => {
  const device = await CalibrationStore.readDevice(cwd, name)
  if (!device) throw new Error(`Missing ${name}`)
  await activateDevice(cwd, device)
  await CalibrationStore.writeDevice(cwd, { name, loaded: true })
  return device
}))
try {
  const before = await snapshotDevices(cwd, names)
  const tool = (await pythonTools(cwd)).tools.find((tool: { id: string }) => tool.id === "g1_motion")
  if (!tool) throw new Error("Missing g1_motion")
  const goals = Object.fromEntries(["left_arm", "right_arm"].map((name) =>
    [name, { frame: `${name}:base`, pose: before.states[name].flange_pose }]))
  const result = await execute(tool, { steps: 3, goals }, {
    directory: cwd, sessionID: "g1-native", abort: new AbortController().signal,
    ask: async () => {}, questions: async () => ({}),
  })
  if (result.isError) throw new Error(result.output)
  const motion = JSON.parse(result.output)
  if (motion.steps !== 3) throw new Error(result.output)
  const after = await snapshotDevices(cwd, names)
  if (Math.abs(after.sample_time_s - before.sample_time_s - .06) > 1e-8)
    throw new Error("Three coordinated writes must advance exactly 60 ms")
  await Bun.write(output, JSON.stringify({ before, after, motion }, null, 2))
} finally {
  await Promise.all(devices.map((device) => deactivateDevice(cwd, device)))
}
process.exit(0)
