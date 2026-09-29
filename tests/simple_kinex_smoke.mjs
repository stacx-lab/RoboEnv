// SIMPLE through the packaged Kinex runtime, local IK, devices and recording.
import assert from 'node:assert/strict'
import { writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [kinexRoot, cwd, output, clockMode] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(path.join(kinexRoot, 'packages/core/dist/index.js')))
const runtime = createKinexRuntime({ directory: cwd, host: {
  ask: async () => {}, questions: async () => ({}),
} })
const invocation = { sessionID: 'simple-smoke', signal: AbortSignal.timeout(120_000) }
const names = ['base', 'left_arm', 'right_arm', 'left_hand', 'right_hand']
async function readings() {
  return Object.fromEntries(await Promise.all(names.map(async name => {
    const result = await runtime.devices.dispatch(name, 'read')
    assert.ok(result.data && !result.data.faults?.length, JSON.stringify(result))
    return [name, result.data]
  })))
}
async function call(name, args) {
  const result = await runtime.tools.call(name, args, invocation)
  assert.ok(!result.isError, result.output)
  return JSON.parse(result.output)
}
// Compose the measured torso and wrist poses into pelvis coordinates. These
// goals force Kinex IK to traverse the live base:torso frame on every iteration.
function compose(a, b) {
  const [x, y, z, w] = a.slice(3)
  const [u, v, s, t] = b.slice(3)
  const [px, py, pz] = b
  const cx = 2 * (y * pz - z * py)
  const cy = 2 * (z * px - x * pz)
  const cz = 2 * (x * py - y * px)
  return [a[0] + px + w * cx + y * cz - z * cy,
    a[1] + py + w * cy + z * cx - x * cz,
    a[2] + pz + w * cz + x * cy - y * cx,
    w * u + x * t + y * s - z * v,
    w * v - x * s + y * t + z * u,
    w * s + x * v - y * u + z * t,
    w * t - x * u - y * v - z * s]
}

try {
  const devices = await runtime.devices.list()
  assert.equal(devices.length, 8)
  for (const device of devices) await (await runtime.devices.load(device.name)).ready
  const before = await readings()
  const handGeometry = await call('hand_geometry_probe', {})
  assert.ok(handGeometry.passed)
  assert.ok(before.base.frame_poses?.torso)
  const goals = Object.fromEntries(['left_arm', 'right_arm'].map(n =>
    [n, { frame: 'base', pose: compose(before.base.frame_poses.torso, before[n].frame_poses.flange) }]))
  const started = await runtime.recording.start({ sessionID: invocation.sessionID,
    instruction: 'SIMPLE integration probe; no task success claim',
    specs: { camera_head_left: { rate: 3 }, camera_head_right: { rate: 3 }, base: { rate: 5 } } })
  const motion = await call('device_composition_probe', { steps: 30, goals, body: {
    mode: 'locomotion', velocity: { vx: 0, vy: 0, wz: 0 },
    posture: { height: .74, roll: .03, pitch: .05, yaw: .08 },
  } })
  assert.equal(motion.steps, 30)
  const after = await readings()
  const elapsed = after.base.sample_time_s - before.base.sample_time_s
  assert.ok(clockMode === 'stepped' ? Math.abs(elapsed - 3) < 1e-8 : elapsed > 0)
  await runtime.recording.snap({ sessionID: invocation.sessionID, devices: ['camera_head_left', 'camera_head_right'] })
  const recording = await runtime.recording.stop({ summary: 'No-model SIMPLE probe' })
  await writeFile(output, JSON.stringify({ passed: true, before, after, handGeometry, motion, started, recording }, null, 2))
} finally {
  await runtime.dispose()
}
