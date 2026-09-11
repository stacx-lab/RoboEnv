# Simulator development and testing skills

RoboEnv has two repository skills for work spanning RoboEnv and Kinex. They share
one acceptance contract and use normal project tools. Skills guide a coding agent;
they do not replace code, installation recipes or runners, and their presence
does not certify a backend.

| Skill | Use it for | Result |
| --- | --- | --- |
| [integrate-simulator-kinex](../.agents/skills/integrate-simulator-kinex/SKILL.md) | Add or migrate a simulator, rig or native device integration | Code, isolated preparation, portable catalog and complete applicable acceptance |
| [test-simulator-kinex](../.agents/skills/test-simulator-kinex/SKILL.md) | Run jobs or inspect acceptance on an existing integration | Evidence for the requested probe, development, frozen validation, full acceptance or Harbor scope |

Integration continues into testing before completion, within the same task and
authorization/evidence history. The user does not need a second prompt. A
standalone test starts with the testing skill and does not imply migration,
paid model development or deployment.

Integration readiness and task performance have separate completion criteria.
Integration requires working installation, native devices, transport, real visual
agent interaction, original outcome reporting, recording and command replay.
A difficult task may fail while that integration is complete. Inspect sessions,
device events and video to distinguish controller failures from integration defects;
do not assume every failure comes from model reasoning. Frozen 3/3 task success is
an optional performance check, required only when that scope is requested.

```text
.agents/skills/
  integrate-simulator-kinex/
    SKILL.md                         implementation and end-to-end ownership
    references/backends.md           backend migration details
    references/acceptance.md         forwarding link for existing callers
  test-simulator-kinex/
    SKILL.md                         executable job and test workflow
    references/acceptance.md         single detailed acceptance contract
```

Implementation is explained in [Integrating a simulator](simulator-integration.md).
Job modes, commands and results are explained in
[Testing simulator integration with Kinex](simulator-kinex-testing.md).
Keep host paths, credentials, assets and results outside instruction files.

## Example requests

- "Use integrate-simulator-kinex to migrate RoboDojo to typed source-backed
  devices and complete native and Kinex integration acceptance; report task outcomes separately."
- "Use test-simulator-kinex to run a no-model smoke job on RoboTwin
  robotwin/stack-blocks-three and verify recording and command replay."
- "Use test-simulator-kinex to validate this frozen controller on these three
  cases, without model calls or controller changes."
- "Use test-simulator-kinex to inspect existing receipts after this driver
  change and identify valid checks; do not launch new jobs."

Model/GPU/license authorization and development limits carry across continuations.
Clarify only unresolved prerequisites. Full integration remains incomplete until
all required gates pass; report a smoke job at its narrower scope.
