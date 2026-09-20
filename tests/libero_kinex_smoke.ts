// Native LIBERO smoke through the public core runtime; no model calls.
import path from "node:path"
import { pathToFileURL } from "node:url"
import { writeFile } from "node:fs/promises"

const [kinexRoot, cwd, output] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(path.join(kinexRoot, "packages/core/dist/index.js")).href)
const runtime = createKinexRuntime({ directory: cwd, host: { ask: async () => {} } })
try {
  const devices = await runtime.devices.list()
  if (devices.length !== 5) throw new Error(`Expected five devices, found ${devices.length}`)
  const activations = await runtime.devices.warmLoaded()
  await Promise.all(activations.map((activation) => activation.ready))
  const first = await runtime.devices.dispatch("arm", "read")
  const target = [...first.data.positions]
  target[0] += 0.05
  const sessionID = "libero-smoke"
  const started = await runtime.recording.start({ sessionID,
    instruction: "LIBERO integration device smoke; no task success claim",
    specs: { camera: { rate: 3 }, camera_wrist: { rate: 3 }, arm: { rate: 5 } } })
  const motion = await runtime.tools.call("servo_to", { device: "arm", joints: target, timeout_s: 25 }, { sessionID })
  if (motion.isError) throw new Error(motion.output)
  const arrived = JSON.parse(motion.output)
  if (!arrived.reached) throw new Error(motion.output)
  await runtime.devices.dispatch("gripper", "write", { target: [1] })
  await runtime.devices.dispatch("gripper", "stop")
  const snap = await runtime.recording.snap({ sessionID, devices: ["camera", "camera_wrist"] })
  await runtime.devices.dispatch("arm", "stop")
  const stopped = await runtime.recording.stop({ summary: "Device smoke passed; no task attempt" })
  await writeFile(output, JSON.stringify({ passed: true, devices: devices.map((device) => device.name),
    motion: arrived, started, snap, recording: stopped }, null, 2))
} finally {
  await runtime.dispose()
}
