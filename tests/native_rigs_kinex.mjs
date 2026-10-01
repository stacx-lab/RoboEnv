// Validate copied rigs using the selected packaged Kinex core; no device activation.
import path from 'node:path'
import { readdir, readFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'

const [kinexArgument, ...workspaces] = process.argv.slice(2)
const core = path.join(path.resolve(kinexArgument), 'packages/core/dist')
const { createKinexRuntime } = await import(pathToFileURL(path.join(core, 'index.js')))
const { parse: parseYaml } = createRequire(path.join(core, '../package.json'))('yaml')
const { Device, RobotManifest } = await import(pathToFileURL(path.join(core, 'device/profile/schema.js')))
const { missingActions } = await import(pathToFileURL(path.join(core, 'device/interface.js')))
const { readinessPlan } = await import(pathToFileURL(path.join(core, 'device/profile/readiness.js')))
const results = []
for (const workspace of workspaces) {
  const root = path.resolve(workspace)
  const runtime = createKinexRuntime({ directory: root })
  try {
    const snapshot = await runtime.profiles.snapshot()
    snapshot.robot = RobotManifest.parse(parseYaml(await readFile(path.join(root, '.kinex/workspace.yaml'), 'utf8')))
    const entries = await readdir(path.join(root, '.kinex/devices'), { withFileTypes: true })
    const devices = []
    for (const entry of entries.filter(entry => entry.isDirectory())) {
      const raw = parseYaml(await readFile(path.join(root, '.kinex/devices', entry.name, 'profile.yaml'), 'utf8'))
      devices.push(Device.parse({ ...raw, loaded: true }))
    }
    if (!devices.length) throw new Error(`No devices in ${root}`)
    for (const device of devices) {
      const missing = missingActions(device, device.access?.driver?.actions ?? [])
      if (missing.length) throw new Error(`${device.name}: missing ${missing.join(', ')}`)
    }
    const plan = await readinessPlan(root, { ...snapshot, devices })
    const unresolved = plan.items.filter(item => !item.optional && item.status !== 'done')
    if (unresolved.length) throw new Error(JSON.stringify({ workspace: root, unresolved }))
    results.push({ workspace: root, profiles: devices.length, calibration: plan.counts })
  } finally {
    await runtime.dispose()
  }
}
console.log(JSON.stringify({ passed: true, results }))
