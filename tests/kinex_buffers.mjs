import assert from 'node:assert/strict'
import path from 'node:path'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'

const [kinex, directory, interpreter, endpoint] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(path.join(kinex, 'packages/core/dist/index.js')))
const runtime = createKinexRuntime({ directory, host: { ask: async () => {} } })
try {
  await runtime.profiles.writeManifest({ identity: { id: 'buffers', name: 'Buffers', platform: 'sim', kind: 'arm' } })
  for (const name of ['left', 'right']) {
    await runtime.profiles.writeDevice({ name, type: 'joint', loaded: true,
      access: { connection: { kind: 'url', url: endpoint, device: name }, driver: { interpreter, persistent: true,
        actions: ['load', 'unload', 'read', 'write', 'status', 'stop'], timeout_s: 5 } },
      intrinsics: {
        model: { format: 'joints', joints: [{ name: 'j', type: 'revolute', limits: { min: -2, max: 2 }, max_velocity: null, max_effort: null }] },
        home: null, control: { mode: 'position' },
        feedback: { velocities: true, efforts: false, clock: 'simulation' },
        stop: { behavior: 'hold', affected_devices: [name] },
      },
    })
    await (await runtime.devices.load(name)).ready
  }
  const tool = path.join(directory, '.kinex/tool/buffer-probe')
  await mkdir(tool, { recursive: true })
  await writeFile(path.join(tool, 'tool.yaml'), `name: buffer_probe\nkind: controller\ndescription: Execute concurrent buffers.\ninterpreter: ${interpreter}\nparameters: {type: object, properties: {}}\n`)
  await writeFile(path.join(tool, 'tool.py'), `from kinex import devices, finish
commands = [{'device': name, 'action': 'write', 'args': {'actions': values, 'freq_hz': 3, 'for_s': 1}} for name, values in {'left': [[.1], [.2], [.3]], 'right': [[-.1], [-.2], [-.3]]}.items()]
finish({'results': devices(commands)})
`)
  const result = await runtime.tools.call('buffer_probe', {}, { sessionID: 'buffers' })
  assert.ok(!result.isError, result.output)
  for (const outcome of JSON.parse(result.output).results) {
    assert.ok(outcome.ok)
    assert.equal(outcome.data.completed_actions, 3)
    assert.equal(outcome.data.elapsed_s, 1)
  }
  const rounded = await runtime.devices.dispatch('left', 'write', { actions: [[.1], [.3]], freq_hz: 3 })
  assert.equal(rounded.data.completed_actions, 2)
  assert.equal(rounded.data.elapsed_s, .66)
  const fallback = path.join(directory, '.kinex/tool/fallback-probe')
  await mkdir(fallback, { recursive: true })
  await writeFile(path.join(fallback, 'tool.yaml'), `name: fallback_probe\nkind: controller\ndescription: Exercise upper-layer goal planning.\ninterpreter: ${interpreter}\nparameters: {type: object}\n`)
  await writeFile(path.join(fallback, 'tool.py'), await readFile(path.join(kinex, 'packages/core/assets/builtin/tool/move-to/tool.py')))
  const reached = await runtime.tools.call('fallback_probe', { device: 'right', joints: [-.5], timeout_s: 3 }, { sessionID: 'fallback' })
  assert.ok(!reached.isError, reached.output)
  const goal = JSON.parse(reached.output)
  assert.equal(goal.reached, true)
  assert.ok(goal.elapsed_s <= 3)
  assert.deepEqual(goal.state.positions, [-.5])
} finally {
  await runtime.dispose()
}
