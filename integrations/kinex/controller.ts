import { z } from "zod"
import { execute, type Context, type Def, type Result } from "@kinex/core/tool/tool"

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
