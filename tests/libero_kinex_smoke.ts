// LIBERO device and recording smoke through the public core runtime.
import assert from 'node:assert/strict'
import { writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [kinexRoot, cwd, output] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(path.join(kinexRoot, 'packages/core/dist/index.js')).href)
const runtime = createKinexRuntime({ directory: cwd, host: {
  ask: async () => {}, questions: async () => ({}),
} })
const invocation = { sessionID: 'libero-smoke', signal: AbortSignal.timeout(120_000) }
async function call(name: string, args: object) {
  const result = await runtime.tools.call(name, args, invocation)
  assert.ok(!result.isError, result.output)
  return JSON.parse(result.output)
}
try {
  const devices = await runtime.profiles.listDevices()
  for (const device of devices) await (await runtime.devices.load(device.name)).ready
  const first = (await runtime.devices.dispatch('arm', 'read')).data
  const target = [...first.positions]
  target[0] += .05
  const started = await runtime.recording.start({ sessionID: invocation.sessionID,
    instruction: 'LIBERO device smoke; no task success claim',
    specs: { camera: { rate: 3 }, camera_wrist: { rate: 3 }, arm: { rate: 5 } } })
  const motion = await call('servo', { device: 'arm', joints: target, duration_s: 2 })
  await runtime.devices.dispatch('world', 'wait', { duration_s: 2 })
  const measured = (await runtime.devices.dispatch('arm', 'read')).data
  assert.ok(Math.abs(measured.positions[0] - target[0]) < .01)
  await runtime.devices.dispatch('gripper', 'write', { actions: [1], duration_s: .2 })
  await runtime.devices.dispatch('world', 'wait', { duration_s: .2 })
  await runtime.devices.dispatch('gripper', 'stop')
  const snap = await runtime.recording.snap({ sessionID: invocation.sessionID, devices: ['camera', 'camera_wrist'] })
  const recording = await runtime.recording.stop({ summary: 'Device smoke passed; no task attempt' })
  await writeFile(output, JSON.stringify({ passed: true, motion, measured, started, snap, recording }, null, 2))
} finally {
  await runtime.dispose()
}
