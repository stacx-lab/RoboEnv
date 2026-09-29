import { z } from "zod"
import type { Context, Def, Result } from "@kinex/core"
// The public runtime returns only the truncated, model-facing result; the
// package exports the normal execution wrapper as an internal module.
import { execute } from "@kinex/core/tool/tool"

const Completion = z.looseObject({ completed: z.literal(true), error: z.null().optional() })

export function controllerCompleted(result: Result | undefined): boolean {
  if (!result || result.isError) return false
  try {
    return Completion.safeParse(JSON.parse(result.output)).success
  } catch {
    return false
  }
}

// Retain the original return value before normal tool execution truncates its
// model-facing text. A caught controller failure must still fail acceptance.
export async function executeController(tool: Def, input: unknown, context: Context) {
  const returned: Result[] = []
  const observed: Def = { ...tool, execute: async (args, ctx) => {
    const result = await tool.execute(args, ctx)
    returned.push(result)
    return result
  } }
  const result = await execute(observed, input, context)
  return { result, raw: returned[0], completed: controllerCompleted(returned[0]) }
}
