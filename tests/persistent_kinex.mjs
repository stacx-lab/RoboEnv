// CPU integration through the public Kinex runtime and portable driver.
import assert from 'node:assert/strict'
import path from 'node:path'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'

const [root, directory, python] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(path.join(root, 'packages/core/dist/index.js')))
const runtime = createKinexRuntime({ directory, enableRecording: true, host: { ask: async () => {} } })
const boots = async () => (await readFile(path.join(directory, '.kinex/devices/controller/boots'), 'utf8')).trim().split('\n')
try {
  await runtime.profiles.writeManifest({ identity: { id: 'cpu', name: 'CPU', platform: 'sim', kind: 'arm' } })
  const tool = path.join(directory, '.kinex/tools/probe')
  await mkdir(tool, { recursive: true })
  await writeFile(path.join(tool, 'tool.yaml'), `name: probe\nkind: perception\ndescription: CPU integration\ninterpreter: ${python}\ntimeout_s: 5\n`)
  await writeFile(path.join(tool, 'tool.py'), 'from kinex import device,finish\nfinish(device("controller"))\n')
  await (await runtime.devices.load('controller')).ready
  for (const _ of [0, 1]) {
    const result = await runtime.tools.call('probe', {}, { sessionID: 'cpu' })
    assert.ok(!result.isError, result.output)
  }
  const snap = await runtime.recording.snap({ sessionID: 'cpu', devices: ['controller'] })
  assert.ok(snap.readings.controller?.data, JSON.stringify(snap))
  assert.deepEqual(snap.refusals, {})
  assert.equal((await boots()).length, 1)
  await (await runtime.devices.reload('controller')).ready
  await runtime.devices.dispatch('controller', 'read')
  const pids = await boots()
  assert.equal(pids.length, 2)
  assert.notEqual(pids[0], pids[1])
  await runtime.devices.unload('controller')
  for (const pid of pids) assert.throws(() => process.kill(Number(pid), 0), { code: 'ESRCH' })
} finally {
  await runtime.dispose()
}
