// CPU-only integration against RoboEnv's real worker and shared portable driver.
import path from "node:path"
import { mkdir, writeFile, access } from "node:fs/promises"
const [root, cwd, python] = process.argv.slice(2)
const { createKinexRuntime } = await import(path.join(root, "packages/core/dist/index.js"))
const runtime = createKinexRuntime({ directory: cwd, host: { ask: async () => {} } })
try {
  await runtime.profiles.writeManifest({ identity: { id: "cpu", name: "CPU", platform: "sim", kind: "arm" } })
  const device = (await runtime.devices.list()).find((device) => device.name === "controller")
  if (!device?.access.driver.persistent) throw new Error("Driver must default to persistent")
  const tool = path.join(cwd, ".kinex/tool/probe")
  await mkdir(tool, { recursive: true })
  await writeFile(path.join(tool, "tool.py"), 'from kinex import device,finish\nfinish(device("controller"))\n')
  await writeFile(path.join(tool, "tool.yaml"), JSON.stringify({ name: "probe", kind: "perception",
    description: "CPU integration", interpreter: python, timeout_s: 5 }))
  await (await runtime.devices.load("controller")).ready
  for (const _ of [0, 1]) {
    const result = await runtime.tools.call("probe", {}, { sessionID: "cpu" })
    if (result.isError) throw new Error(result.output)
  }
  const snap = await runtime.recording.snap({ sessionID: "cpu", devices: ["controller"] })
  if (!snap.readings.controller?.data || snap.readings.controller.error
    || Object.keys(snap.refusals).length) throw new Error(JSON.stringify(snap))
  await access(path.join(snap.dir, "recording.json"))
  await (await runtime.devices.reload("controller")).ready
  await runtime.devices.dispatch("controller", "read")
} finally {
  await runtime.dispose()
}
