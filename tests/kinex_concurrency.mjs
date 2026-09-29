import assert from 'node:assert/strict'
import path from 'node:path'
import { readFile } from 'node:fs/promises'
import { pathToFileURL } from 'node:url'
import { setTimeout as delay } from 'node:timers/promises'

const [kinex, directory, interpreter] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(path.join(kinex, 'packages/core/dist/index.js')))
const runtime = createKinexRuntime({ directory, enableRecording: true })
try {
  await runtime.profiles.writeManifest({ identity: { id: 'concurrency', name: 'Concurrency', platform: 'sim', kind: 'arm' } })
  for (const name of ['left', 'right']) {
    await runtime.profiles.writeDevice({ name, type: 'joint', loaded: true,
      access: { connection: { kind: 'mock' }, driver: { interpreter, persistent: true,
        actions: ['load', 'unload', 'read', 'write', 'status', 'stop'], timeout_s: 5 } },
      intrinsics: {
        model: { format: 'joints', joints: [{ name: 'j0', type: 'revolute', limits: { min: -2, max: 2 }, max_velocity: 2, max_effort: null }] },
        home: null, control: { mode: 'position' },
        feedback: { velocities: true, efforts: false, clock: 'simulation' },
        stop: { behavior: 'hold', affected_devices: [name] },
      },
    })
    await (await runtime.devices.load(name)).ready
  }
  const { endpoint } = JSON.parse(await readFile(path.join(directory, '.kinex/devices/world/connection.json'), 'utf8'))
  await runtime.profiles.writeDevice({ name: 'world', type: 'world', loaded: true,
    access: { connection: { kind: 'url', url: endpoint, device: 'world' }, driver: { interpreter, persistent: true,
      actions: ['load', 'unload', 'read', 'wait', 'status'], timeout_s: 5 } } })
  await (await runtime.devices.load('world')).ready
  const samples = []
  runtime.subscribe(event => {
    if (event.type === 'device' && event.event.origin === 'sampler' && event.event.reading) samples.push(event.event)
  })
  await runtime.recording.start({ specs: { left: { rate: 30 }, right: { rate: 30 } }, sessionID: 'test' })
  await runtime.devices.dispatch('left', 'write', { actions: [[.2]], duration_s: 1 })
  await runtime.devices.dispatch('right', 'write', { actions: [[.6]], duration_s: .2 })
  const waiting = runtime.devices.dispatch('world', 'wait', { duration_s: 1 })
  await delay(100)
  assert.ok(samples.filter(event => event.device.name === 'left').length >= 2)
  await runtime.recording.stop()
  await waiting
  assert.deepEqual((await runtime.devices.dispatch('left', 'read')).data.positions, [.2])
  const cancelled = new AbortController()
  const long = runtime.devices.dispatch('world', 'wait', { duration_s: 5 }, cancelled.signal)
  const rejected = assert.rejects(long)
  const peer = runtime.devices.dispatch('world', 'wait', { duration_s: .2 })
  await delay(20)
  cancelled.abort(new Error('cancel one world wait'))
  await rejected
  assert.ok((await peer).data.elapsed_s >= .2 - 1e-9)
  const clock = (await runtime.devices.dispatch('world', 'read')).data
  assert.ok(Number.isFinite(clock.time_s))
  // Kinex validates the informational budget as {elapsed_s, max_s} seconds; this fixture is uncapped.
  assert.deepEqual(Object.keys(clock.budget).sort(), ['elapsed_s', 'max_s'])
  assert.ok(Math.abs(clock.budget.elapsed_s - clock.time_s) < 1e-6)
  assert.equal(clock.budget.max_s, null)
  const status = (await runtime.devices.dispatch('world', 'status')).data
  assert.deepEqual(Object.keys(status.budget).sort(), ['elapsed_s', 'max_s'])
  assert.ok(status.budget.elapsed_s >= clock.budget.elapsed_s)

} finally {
  await runtime.dispose()
}
