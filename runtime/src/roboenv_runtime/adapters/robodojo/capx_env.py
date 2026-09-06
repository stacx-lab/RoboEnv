"""High-level code-execution task envs for RoboDojo's dual ARX X5 arms.

Importing this module registers the ARX X5 APIs (see
`capx.integrations.arx_x5`), so RoboDojo configs need no edits to any existing
CaP-X registry file.

Prompts are assembled from named PromptSpec sections (situation / method /
facts / task / skills / contract) so experiments can swap or ablate one
section from YAML, and lessons learned live in the skill library
(`capx/skills/arx_x5/`) instead of being pasted into prompt strings.

Knowledge-injection modes (per-run, from YAML):
    documented: true      -> inject "hardware-spec" skills (the datasheet)
    skills: true          -> inject the whole skill library
    skills: [name, ...]   -> inject selected skills by file stem
    (neither)             -> pure discovery mode
"""

# Moved from third_party/cap-x/capx/envs/tasks/arx_x5/robodojo_tasks.py on 2026-08-26 (refactor P2); cap-x keeps a shim.

from capx.envs.tasks.base import CodeExecutionEnvBase

import roboenv_runtime.adapters.robodojo.api  # noqa: F401  (registers RoboDojoPlannerPrivilegedApi)
from roboenv_runtime.prompt import PromptSpec
from roboenv_runtime.skills.store import SkillStore, default_store

SKILL_DOMAIN = "arx_x5"

SITUATION = """
You are controlling a dual-arm robot (two ARX X5 6-DoF arms with parallel
grippers) in a tabletop scene, by writing Python code against the API below.

Coordinate system (world frame):
- Z-axis is up; the tabletop surface is at z ~= 0.765.
- The LEFT arm base sits at x ~= -0.3 and the RIGHT arm base at x ~= +0.3, both
  on the negative-Y side of the table (y = -0.45). Objects spawn roughly within
  x in [-0.45, 0.45], y in [-0.25, 0.0].
- All quaternions are WXYZ order: [qw, qx, qy, qz].
"""

METHOD = """
Working method:
- Call list_objects() first, then get_object_pose() / get_object_bbox_extent()
  for the ones you need. Extents are full side lengths (half-height is
  extent[2]/2). Poses are ground truth, so USE THEM TO VERIFY every action;
  never assume an action worked without checking.
- Read the API docs below carefully - the gripper's orientation convention and
  the fingertip-frame semantics of goto_pose() are specific to this robot.
- goto_pose() raises when a pose is unreachable or reached only approximately.
  Wrap moves in try/except and fall back to a nearby pose, another approach
  direction, or the other arm instead of letting the program die.
- Both arms can work; each has limited reach around its own base, overlapping
  near the table middle. Objects only one arm can reach may still end up
  anywhere the OTHER arm can place them - think about cooperation.
- After releasing an object, call wait() so it settles before you read poses.
"""

FACTS = """
Environment mechanics:
- The episode has a fixed simulation-step budget (see get_step_budget());
  every motion and wait consumes it, and the episode ends mid-action when it
  runs out.
- Carried objects hang below your fingertips.
"""

CONTRACT = """
You may write python code comments for reasoning but ONLY write executable
Python code, and do not wrap it in code fences. The API functions below are
already imported. If you want numpy, import it explicitly.
"""

TASK_PACK = """
Task: Several loose objects are scattered on the table next to an open cardboard
box. Pick each object up and place it inside the box.

Scoring is graded per object. An object counts only when it rests inside the
box with its long axis running along the box's long axis and its front/head
end pointing toward world +X. The box must stay upright and where it is. Full
completion additionally requires both grippers open and both arms returned
home (go_home) at the end. Pack one object at a time and keep going even if
one fails.
"""

BASE_SPEC = PromptSpec(
    situation=SITUATION,
    task="",
    method=METHOD,
    facts=FACTS,
    skills="",
    contract=CONTRACT,
)


def _select_skills(documented: bool, skills) -> list:
    store = default_store()
    picked = []
    if documented:
        picked += store.shared([SKILL_DOMAIN], kinds=["hardware-spec"])
    if skills is True:
        picked += store.shared([SKILL_DOMAIN])
    elif isinstance(skills, (list, tuple)):
        picked += store.shared([SKILL_DOMAIN], names=list(skills))
    seen: set[str] = set()
    return [s for s in picked if not (s.name in seen or seen.add(s.name))]


def render_skills(skills: list) -> str:
    """The prompt section for the selected skills (cap-x launcher wording)."""
    return SkillStore.render(
        skills, header="Field notes from previous work on this platform (hard-won; trust them):"
    )


class RoboDojoCodeEnv(CodeExecutionEnvBase):
    """Generic RoboDojo task env; per-task subclasses override `prompt_spec`.

    YAML-tunable knowledge injection: ``documented`` (hardware spec sheet),
    ``skills`` (True = whole library, or a list of skill names).
    """

    prompt_spec = BASE_SPEC

    def __init__(self, *args, documented: bool = False, skills=None, **kwargs):
        spec = type(self).prompt_spec
        chosen = _select_skills(documented, skills)
        if chosen:
            spec = spec.with_(skills=render_skills(chosen))
        self.prompt = spec.render()
        super().__init__(*args, **kwargs)


class RoboDojoPackObjectsCodeEnv(RoboDojoCodeEnv):
    """RoboDojo `pack_objects_into_box` (and its `_random` variant)."""

    prompt_spec = BASE_SPEC.with_(task=TASK_PACK)


__all__ = ["RoboDojoCodeEnv", "RoboDojoPackObjectsCodeEnv"]
