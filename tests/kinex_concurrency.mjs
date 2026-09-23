import assert from 'node:assert/strict'
import path from 'node:path'
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
  const samples = []
  runtime.subscribe(event => {
    if (event.type === 'device' && event.event.origin === 'sampler' && event.event.reading) samples.push(event.event)
  })
  await runtime.recording.start({ specs: { left: { rate: 30 }, right: { rate: 30 } }, sessionID: 'test' })
  let finished = false
  const left = runtime.devices.dispatch('left', 'write', { actions: [[.2]], freq_hz: 1 }).then(value => { finished = true; return value })
  while (!(await runtime.devices.dispatch('left', 'read')).data.commands.length) await delay(5)
  const right = await runtime.devices.dispatch('right', 'write', { actions: [[.6]], freq_hz: 5 })
  assert.equal(finished, false)
  assert.ok(right.data.elapsed_s >= .2)
  assert.ok(samples.filter(event => event.device.name === 'left').length >= 2)
  await runtime.recording.stop()
  assert.equal(finished, false)
  await left
} finally {
  await runtime.dispose()
}
