// ABC uses the existing Kinex lifecycle, tools and recorder; no provider calls.
import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [kinex, workspace, output] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(path.join(kinex, 'packages/core/dist/index.js')))
const runtime = createKinexRuntime({ directory: workspace, host: {
  ask: async () => {}, questions: async () => ({}),
} })
const invocation = { sessionID: 'abc-smoke', signal: AbortSignal.timeout(180_000) }
async function call(name, args) {
  const result = await runtime.tools.call(name, args, invocation)
  assert.ok(!result.isError, result.output)
  return JSON.parse(result.output)
}
try {
  const devices = await runtime.devices.list()
  assert.equal(devices.length, 8)
  for (const device of devices) await (await runtime.devices.load(device.name)).ready
  const cameras = ['camera_top', 'camera_left', 'camera_right']
  for (const name of cameras) {
    const camera = devices.find(device => device.name === name)
    assert.equal(camera.type, 'rgbd')
    const reading = await runtime.devices.dispatch(name, 'read', { output: 'file' })
    assert.ok(Number.isFinite(reading.data.timestamp))
    const depth = await readFile(reading.data.depth_file)
    const header = Buffer.from('P5\n640 480\n65535\n')
    assert.ok(depth.subarray(0, header.length).equals(header))
    assert.equal(depth.length, header.length + 640 * 480 * 2)
    assert.ok(depth.subarray(header.length).some(value => value !== 0))
  }
  const before = (await runtime.devices.dispatch('left_arm', 'read')).data
  const target = [...before.positions]
  target[0] += .05
  const started = await runtime.recording.start({ sessionID: invocation.sessionID,
    instruction: 'ABC integration probe; no task success claim',
    specs: { camera_top: { rate: 3 }, camera_left: { rate: 3 }, camera_right: { rate: 3 }, left_arm: { rate: 5 } } })
  const motion = await call('servo', { device: 'left_arm', joints: target, block: false, speedFraction: .2 })
  await runtime.devices.dispatch('world', 'wait', { duration_s: Math.max(1, motion.duration_s + .3) })
  const after = (await runtime.devices.dispatch('left_arm', 'read')).data
  assert.ok(Math.abs(after.positions[0] - target[0]) < .01)
  const fk = (await runtime.devices.dispatch('left_arm', 'fk', {
    joints: after.positions, frame: 'flange',
  })).data
  const ik = (await runtime.devices.dispatch('left_arm', 'ik', {
    pose: fk.pose, seed: after.positions, frame: 'flange',
  })).data
  await runtime.devices.dispatch('left_gripper', 'write', { actions: [.3], duration_s: .2 })
  await runtime.devices.dispatch('world', 'wait', { duration_s: .2 })
  await runtime.devices.dispatch('left_gripper', 'stop')
  // Allow the wall-clock recorder to capture several frames while stepped physics is idle.
  await new Promise(resolve => setTimeout(resolve, 1200))
  await runtime.recording.snap({ sessionID: invocation.sessionID, devices: cameras })
  const recording = await runtime.recording.stop({ summary: 'ABC device probe complete' })
  await writeFile(output, JSON.stringify({ passed: true, rgbd_cameras: cameras, before, after, fk, ik, motion, started, recording }, null, 2))
} finally {
  await runtime.dispose()
}
