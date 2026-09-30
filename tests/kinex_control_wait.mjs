// Native motion smoke through Kinex's permission-checked tools; no model calls.
import assert from 'node:assert/strict'
import { writeFile } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [kinex, workspace, mode, output] = process.argv.slice(2)
const { createKinexRuntime } = await import(pathToFileURL(path.join(kinex, 'packages/core/dist/index.js')))
const runtime = createKinexRuntime({ directory: workspace, host: {
  ask: async () => {}, questions: async () => ({}),
} })
const invocation = { sessionID: 'control-wait-smoke', signal: AbortSignal.timeout(120_000) }
async function call(name, args) {
  const result = await runtime.tools.call(name, args, invocation)
  assert.ok(!result.isError, result.output)
  return JSON.parse(result.output)
}
try {
  await call('device', { deviceID: 'arm', action: 'load', compact: true })
  await call('device', { deviceID: 'world', action: 'load', compact: true })
  const before = await call('device', { deviceID: 'arm', action: 'read', compact: true })
  const target = [...before.positions]
  target[0] += 0.04
  const started = performance.now()
  const motion = await call('servo', { device: 'arm', joints: target, speedFraction: 0.2 })
  const waited = await call('device', { deviceID: 'world', action: 'wait', compact: true, args: { duration_s: Math.max(2, motion.duration_s + 0.2) } })
  const after = await call('device', { deviceID: 'arm', action: 'read', compact: true })
  const wall_s = (performance.now() - started) / 1000
  assert.ok(motion.actions_submitted > 0)
  assert.ok(waited.elapsed_s >= 2 - 1e-9)
  assert.ok(Math.abs(after.positions[0] - target[0]) < .01)
  const report = { mode, wall_s, motion, model_calls: 0, task_success_claimed: false }
  await writeFile(output, JSON.stringify(report, null, 2) + '\n')
  console.log(JSON.stringify({ mode, wall_s, elapsed_s: waited.elapsed_s }))
} finally {
  await runtime.dispose()
}
