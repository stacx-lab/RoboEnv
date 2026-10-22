// Real Kinex lifecycle and Python tools; no model/provider calls.
import assert from 'node:assert/strict'
import { writeFile } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'

const [root, cwd, output] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(root + '/packages/core/dist/index.js'))
const runtime = createKinexRuntime({ directory: cwd, host: {
  ask: async () => {}, questions: async () => ({}),
} })
try {
  await runtime.python.prepare()
  const devices = await runtime.devices.list()
  assert.deepEqual(devices.map(d => d.name).sort(), ['instrument', 'world'])
  for (const device of devices) await (await runtime.devices.load(device.name)).ready
  const result = await runtime.tools.call('chemlab_device_probe', {}, {
    sessionID: 'chemlab-no-model', signal: AbortSignal.timeout(120000),
  })
  assert.ok(!result.isError, result.output)
  const data = JSON.parse(result.output)
  assert.equal(data.after.observations['light.on'].value, true)
  assert.ok(Math.abs(data.after.observations['sash.height_m'].value - .457) < .001)
  await writeFile(output, JSON.stringify({ passed: true, result: data }, null, 2))
} finally {
  await runtime.dispose()
}
