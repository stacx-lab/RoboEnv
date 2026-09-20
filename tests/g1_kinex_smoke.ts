// Explicit integration check: copied G1 drivers and the workspace controller.
import path from "node:path"
import { writeFile } from "node:fs/promises"

const [kinexRoot, cwd, output] = process.argv.slice(2)
const { createKinexRuntime } = await import(path.join(kinexRoot, "packages/core/dist/index.js"))
const runtime = createKinexRuntime({ directory: cwd, host: { ask: async () => {} } })
const names = ["base", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
try {
  for (const name of names) await (await runtime.devices.load(name)).ready
  const before = Object.fromEntries(await Promise.all(names.map(async (name) => [name, (await runtime.devices.dispatch(name, "read")).data])))
  const goals = Object.fromEntries(["left_arm", "right_arm"].map((name) =>
    [name, { frame: name, pose: before[name].frame_poses.flange }]))
  const result = await runtime.tools.call("device_composition_probe", { steps: 3, goals }, { sessionID: "g1-native" })
  if (result.isError) throw new Error(result.output)
  const motion = JSON.parse(result.output)
  if (motion.steps !== 3) throw new Error(result.output)
  const after = Object.fromEntries(await Promise.all(names.map(async (name) => [name, (await runtime.devices.dispatch(name, "read")).data])))
  if (Math.abs(after.base.sample_time_s - before.base.sample_time_s - .02 * names.length * 3) > 1e-8)
    throw new Error("Each component write advances its own control interval")
  await writeFile(output, JSON.stringify({ before, after, motion }, null, 2))
} finally {
  await runtime.dispose()
}
