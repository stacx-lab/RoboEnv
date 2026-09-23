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
  const before = await call('device', { deviceID: 'arm', action: 'read', compact: true })
  const target = [...before.positions]
  target[0] += 0.04
  const started = performance.now()
  const motion = await call('device', { deviceID: 'arm', action: 'set', compact: true, args: { target, for_s: 15, arrive: { joint_tolerance: target.map(() => .01) } } })
  const wall_s = (performance.now() - started) / 1000
  assert.ok(motion.elapsed_s > 0, JSON.stringify(motion))
  assert.ok(motion.reached, JSON.stringify(motion))
  assert.ok(wall_s < 60, `write hit the old timeout: ${wall_s}`)
  assert.ok(Math.abs(motion.state.positions[0] - before.positions[0]) > 0.01)
  const report = { mode, wall_s, motion, model_calls: 0, task_success_claimed: false }
  await writeFile(output, JSON.stringify(report, null, 2) + '\n')
  console.log(JSON.stringify({ mode, wall_s, elapsed_s: motion.elapsed_s, rounds: motion.rounds,
    reached: motion.reached }))
} finally {
  await runtime.dispose()
}
