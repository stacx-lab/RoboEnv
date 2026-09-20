// Validate copied rigs with the selected Kinex schema, action contract and DAG.
import path from "node:path"

const [kinexArgument, ...workspaces] = process.argv.slice(2)
const kinexRoot = path.resolve(kinexArgument)
const { createKinexRuntime } = await import(path.join(kinexRoot, "packages/core/dist/index.js"))
const { missingActions } = await import(path.join(kinexRoot, "packages/core/dist/device/interface.js"))
const results = []
for (const workspace of workspaces) {
  const root = path.resolve(workspace)
  const runtime = createKinexRuntime({ directory: root })
  try {
    const { robot, devices } = await runtime.profiles.snapshot()
    if (!robot || devices.length === 0) throw new Error(`No valid rig in ${root}`)
    for (const device of devices) {
      const missing = missingActions(device, device.access?.driver?.actions ?? [])
      if (missing.length) throw new Error(`${device.name}: missing ${missing.join(", ")}`)
    }
    const plan = await runtime.calibration.plan()
    const unresolved = plan.items.filter((item) => !item.optional && item.status !== "done")
    if (unresolved.length) throw new Error(JSON.stringify({ workspace: root, unresolved }))
    results.push({ workspace: root, profiles: devices.length, calibration: plan.counts })
  } finally {
    await runtime.dispose()
  }
}
console.log(JSON.stringify({ passed: true, results }))
