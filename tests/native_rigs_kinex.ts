// Validate copied rigs with the selected Kinex schema, action contract and DAG.
import path from "node:path"

const [kinexArgument, ...workspaces] = process.argv.slice(2)
const kinexRoot = path.resolve(kinexArgument)
const { Device, RobotManifest } = await import(path.join(kinexRoot, "src/device/profile/schema.ts"))
const { missingActions } = await import(path.join(kinexRoot, "src/device/interface.ts"))
const { calibrationPlan } = await import(path.join(kinexRoot, "src/calibration/dag.ts"))
const results = []
for (const workspace of workspaces) {
  const root = path.resolve(workspace)
  const robot = RobotManifest.parse(Bun.YAML.parse(await Bun.file(path.join(root, ".kinex/workspace.yaml")).text()))
  const devices = []
  for await (const relative of new Bun.Glob("*/profile.yaml").scan(path.join(root, ".kinex/devices"))) {
    const file = path.join(root, ".kinex/devices", relative)
    const raw = Bun.YAML.parse(await Bun.file(file).text())
    const device = Device.parse({ ...raw, loaded: true })
    const missing = missingActions(device, device.access?.driver?.actions ?? [])
    if (missing.length) throw new Error(`${file}: missing ${missing.join(", ")}`)
    devices.push(device)
  }
  if (devices.length === 0) throw new Error(`No devices in ${root}`)
  const plan = await calibrationPlan(root, { robot, devices })
  const unresolved = plan.items.filter((item: { status: string; optional: boolean }) => !item.optional && item.status !== "done")
  if (unresolved.length) throw new Error(JSON.stringify({ workspace: root, unresolved }))
  results.push({ workspace: root, profiles: devices.length, calibration: plan.counts })
}
console.log(JSON.stringify({ passed: true, results }))
process.exit(0)
