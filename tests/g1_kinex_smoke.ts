// Explicit integration check: copied G1 drivers and the workspace controller.
import path from "node:path"

const [kinexRoot, cwd, output] = process.argv.slice(2)
const { CalibrationStore } = await import(path.join(kinexRoot, "src/calibration/store.ts"))
const { activateDevice, deactivateDevice, dispatchDevice } = await import(path.join(kinexRoot, "src/device/dispatch.ts"))
const { pythonTools } = await import(path.join(kinexRoot, "src/tool/python.ts"))
const { execute } = await import(path.join(kinexRoot, "src/tool/tool.ts"))
const names = ["world", "base", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
const devices = await Promise.all(names.map(async (name) => {
  const device = await CalibrationStore.readDevice(cwd, name)
  if (!device) throw new Error(`Missing ${name}`)
  await activateDevice(cwd, device)
  await CalibrationStore.writeDevice(cwd, { name, loaded: true })
  return device
}))
try {
  const before = Object.fromEntries(await Promise.all(devices.map(async (device) => [device.name, (await dispatchDevice(cwd, device, "read", {})).data])))
  const tool = (await pythonTools(cwd)).tools.find((tool: { id: string }) => tool.id === "device_composition_probe")
  if (!tool) throw new Error("Missing device_composition_probe")
  const goals = Object.fromEntries(["left_arm", "right_arm"].map((name) =>
    [name, { frame: name, pose: before[name].frame_poses.flange }]))
  const result = await execute(tool, { steps: 3, goals }, {
    directory: cwd, sessionID: "g1-native", abort: new AbortController().signal,
    ask: async () => {}, questions: async () => ({}),
  })
  if (result.isError) throw new Error(result.output)
  const motion = JSON.parse(result.output)
  if (motion.steps !== 3) throw new Error(result.output)
  const after = Object.fromEntries(await Promise.all(devices.map(async (device) => [device.name, (await dispatchDevice(cwd, device, "read", {})).data])))
  if (motion.elapsed_s < .3 - 1e-8)
    throw new Error("The loop must wait at least 0.3 world seconds")
  await Bun.write(output, JSON.stringify({ before, after, motion }, null, 2))
} finally {
  await Promise.all(devices.map((device) => deactivateDevice(cwd, device)))
}
process.exit(0)
