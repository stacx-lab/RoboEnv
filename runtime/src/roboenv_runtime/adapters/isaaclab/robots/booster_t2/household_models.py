"""Authored household solids, passive mechanisms and functional geometry in metres.

Visual and collision parts share geometry. Cavities are compound solids, never
convex hulls of a hollow object. This module has no simulator dependency.
"""

from dataclasses import dataclass, field
import math


PALETTE = {
    "oak": ((0.48, 0.28, 0.13), 0.55, 0.0),
    "oak_light": ((0.72, 0.52, 0.29), 0.5, 0.0),
    "walnut": ((0.19, 0.10, 0.055), 0.48, 0.0),
    "ceramic": ((0.86, 0.83, 0.72), 0.3, 0.0),
    "teal": ((0.07, 0.28, 0.28), 0.62, 0.0),
    "fabric": ((0.27, 0.39, 0.40), 0.9, 0.0),
    "cream": ((0.73, 0.69, 0.59), 0.85, 0.0),
    "metal": ((0.43, 0.46, 0.48), 0.32, 0.7),
    "black": ((0.035, 0.045, 0.05), 0.5, 0.1),
    "red": ((0.66, 0.075, 0.045), 0.46, 0.0),
    "blue": ((0.075, 0.23, 0.59), 0.5, 0.0),
    "green": ((0.23, 0.36, 0.08), 0.55, 0.0),
    "paper": ((0.88, 0.85, 0.74), 0.85, 0.0),
    "rubber": ((0.045, 0.05, 0.045), 0.95, 0.0),
    "brass": ((0.57, 0.39, 0.12), 0.3, 0.65),
    "cardboard": ((0.55, 0.36, 0.18), 0.9, 0.0),
}


@dataclass
class Part:
    name: str
    shape: str
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    rpy: tuple = (0.0, 0.0, 0.0)
    material: str = "oak"
    collision: bool = True


@dataclass
class Link:
    name: str
    parts: list = field(default_factory=list)
    mass: float = 0.1
    pos: tuple = (0.0, 0.0, 0.0)
    linear_damping: float = 0.05
    angular_damping: float = 0.08


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    axis: str
    anchor: tuple
    limits: tuple
    damping: float = 0.02
    friction: float = 0.01


@dataclass
class Model:
    links: list
    joints: list = field(default_factory=list)
    fixed: bool = False
    # Physical regions are in the named link frame: center, half extents.
    regions: dict = field(default_factory=dict)
    radius: float = 0.05
    height: float = 0.1


def box(name, size, pos=(0.0, 0.0, 0.0), material="oak", rpy=(0.0, 0.0, 0.0), collision=True):
    return Part(name, "box", tuple(size), tuple(pos), tuple(rpy), material, collision)


def cylinder(
    name, radius, height, pos=(0.0, 0.0, 0.0), material="metal", rpy=(0.0, 0.0, 0.0), collision=True
):
    return Part(name, "cylinder", (radius, height), tuple(pos), tuple(rpy), material, collision)


def sphere(
    name, radius, pos=(0.0, 0.0, 0.0), material="red", scale=(1.0, 1.0, 1.0), collision=True
):
    return Part(name, "sphere", (radius, *scale), tuple(pos), (0.0, 0.0, 0.0), material, collision)


def ring(name, radius, thickness, height, z, material, count=40):
    """Overlapping tangent blocks preserve a circular cavity in PhysX."""
    return [
        box(
            f"{name}_{i}",
            (thickness, 2 * radius * math.tan(math.pi / count) + 0.002, height),
            (radius * math.cos(a), radius * math.sin(a), z),
            material,
            (0.0, 0.0, a),
        )
        for i in range(count)
        for a in [2 * math.pi * i / count]
    ]


def tray_parts(radius=0.18, height=0.025, material="ceramic"):
    return [
        cylinder("floor", radius, 0.012, (0, 0, 0.006), material),
        *ring("rim", radius - 0.006, 0.012, height, height / 2, material),
    ]


def basin_parts(size=(0.36, 0.26), wall=0.06, material="oak"):
    x, y = size
    return [
        box("floor", (x, y, 0.014), (0, 0, 0.007), material),
        box("left", (0.012, y, wall), (-x / 2, 0, wall / 2), material),
        box("right", (0.012, y, wall), (x / 2, 0, wall / 2), material),
        box("back", (x, 0.012, wall), (0, y / 2, wall / 2), material),
        box("front", (x, 0.012, wall), (0, -y / 2, wall / 2), material),
    ]


def table_parts(size=(1.4, 0.8), height=0.82, material="oak_light"):
    x, y = size
    return [
        box("top", (x, y, 0.045), (0, 0, height - 0.0225), material),
        *[
            box(
                f"leg_{i}",
                (0.065, 0.065, height - 0.045),
                (a * (x / 2 - 0.08), b * (y / 2 - 0.08), (height - 0.045) / 2),
                "walnut",
            )
            for i, (a, b) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))
        ],
        box("apron", (x - 0.12, 0.04, 0.11), (0, y / 2 - 0.055, height - 0.10), "oak"),
    ]


def model(name):
    """Return a new definition; mutations cannot affect a later episode."""
    if name in ("can_tomato", "can_beans"):
        color = "red" if name == "can_tomato" else "blue"
        parts = [
            cylinder("body", 0.034, 0.108, (0, 0, 0.054), color),
            cylinder("bottom", 0.0345, 0.004, (0, 0, 0.002), "metal"),
            cylinder("lid", 0.0345, 0.004, (0, 0, 0.110), "metal"),
            *ring("rim", 0.032, 0.003, 0.003, 0.113, "metal", 32),
        ]
        if name == "can_tomato":
            parts += [
                sphere("tomato_picture", 0.018, (0, -0.033, 0.06), "red", (1, 0.12, 0.9), False),
                box("label", (0.052, 0.001, 0.067), (0, -0.034, 0.061), "paper", collision=False),
                sphere("tomato_icon", 0.017, (0, -0.035, 0.06), "red", (1, 0.12, 0.9), False),
            ]
        else:
            parts += [
                box("label", (0.052, 0.001, 0.067), (0, -0.034, 0.061), "paper", collision=False),
                *[
                    sphere(f"bean_icon_{i}", 0.008, (x, -0.035, z), "walnut", (1, 0.2, 1.4), False)
                    for i, (x, z) in enumerate(((-0.012, 0.05), (0.012, 0.06), (0, 0.078)))
                ],
            ]
        return Model([Link("body", parts, 0.12)], radius=0.035, height=0.115)
    if name == "apple":
        return Model(
            [
                Link(
                    "body",
                    [
                        sphere("fruit", 0.043, (0, 0, 0.041), "red", (1, 1, 0.94)),
                        cylinder("stem", 0.003, 0.018, (0, 0, 0.084), "walnut"),
                        sphere("leaf", 0.012, (0.009, 0, 0.087), "green", (1, 0.35, 0.18), False),
                    ],
                    0.09,
                )
            ],
            radius=0.043,
            height=0.093,
        )
    if name == "cracker_box":
        return Model(
            [
                Link(
                    "body",
                    [
                        box("carton", (0.11, 0.065, 0.18), (0, 0, 0.09), "red"),
                        box(
                            "front_label",
                            (0.085, 0.001, 0.105),
                            (0, -0.033, 0.10),
                            "paper",
                            collision=False,
                        ),
                        *[
                            sphere(
                                f"cracker_{i}",
                                0.018,
                                (x, -0.035, z),
                                "oak_light",
                                (1, 0.08, 1),
                                False,
                            )
                            for i, (x, z) in enumerate(
                                ((-0.022, 0.075), (0.02, 0.10), (-0.015, 0.13))
                            )
                        ],
                    ],
                    0.12,
                )
            ],
            radius=0.065,
            height=0.18,
        )
    if name in ("cylinder_block", "cube_block", "blue_block", "red_block", "small_block"):
        size = 0.044 if name == "small_block" else 0.055 if name == "cylinder_block" else 0.065
        color = "blue" if name == "blue_block" else "red" if name == "red_block" else "oak_light"
        part = (
            cylinder("shape", size / 2, size, (0, 0, size / 2), color)
            if name == "cylinder_block"
            else box("shape", (size, size, size), (0, 0, size / 2), color)
        )
        return Model(
            [Link("body", [part], 0.05)],
            radius=size / 2 if name == "cylinder_block" else size / math.sqrt(2),
            height=size,
        )
    if name in ("plate", "dish", "bowl"):
        radius, height = {"plate": (0.19, 0.025), "dish": (0.13, 0.022), "bowl": (0.13, 0.09)}[name]
        parts = tray_parts(radius, height)
        return Model(
            [Link("body", parts, 0.18)],
            regions={
                "inside": ("body", (0, 0, height / 2), (radius - 0.016, radius - 0.016, height / 2))
            },
            radius=radius,
            height=height,
        )
    if name == "tray":
        return Model(
            [Link("body", basin_parts((0.42, 0.31), 0.03, "teal"), 0.25)],
            regions={"inside": ("body", (0, 0, 0.025), (0.19, 0.135, 0.04))},
            radius=0.26,
            height=0.03,
        )
    if name in ("round_mat", "rectangular_mat"):
        part = (
            cylinder("mat", 0.20, 0.002, (0, 0, 0.001), "teal")
            if name == "round_mat"
            else box("mat", (0.40, 0.29, 0.002), (0, 0, 0.001), "teal")
        )
        return Model(
            [Link("body", [part], 0.12)],
            regions={
                "top": (
                    "body",
                    (0, 0, 0.002),
                    (0.18, 0.125 if name == "rectangular_mat" else 0.18, 0.01),
                )
            },
            radius=0.20,
            height=0.002,
        )
    if name in ("book", "open_book", "remote", "parcel"):
        sizes = {
            "book": (0.15, 0.20, 0.035),
            "open_book": (0.31, 0.21, 0.024),
            "remote": (0.055, 0.15, 0.028),
            "parcel": (0.34, 0.24, 0.20),
        }
        x, y, z = sizes[name]
        color = {"book": "teal", "open_book": "paper", "remote": "black", "parcel": "cardboard"}[
            name
        ]
        parts = [box("body", (x, y, z), (0, 0, z / 2), color)]
        if "book" in name:
            parts += [
                box(
                    "pages",
                    (x - 0.008, y - 0.012, z - 0.008),
                    (0, 0, z / 2),
                    "paper",
                    collision=False,
                )
            ]
            if name == "open_book":
                parts += [
                    box(
                        f"line_{i}_{int(side > 0)}",
                        (0.105, 0.001, 0.0005),
                        (side * 0.077, -0.065 + i * 0.012, z + 0.0004),
                        "walnut",
                        collision=False,
                    )
                    for side in (-1, 1)
                    for i in range(12)
                ]
        elif name == "remote":
            parts += [
                cylinder(
                    f"button_{i + 1}_{j}",
                    0.005,
                    0.002,
                    (i * 0.014, -0.04 + j * 0.025, z + 0.001),
                    "red" if j == 0 else "cream",
                    collision=False,
                )
                for i in (-1, 0, 1)
                for j in range(4)
            ]
        else:
            parts += [
                box("tape", (0.055, y + 0.001, z + 0.001), (0, 0, z / 2), "cream", collision=False)
            ]
        return Model(
            [Link("body", parts, 0.08 if name == "remote" else 0.18)],
            radius=math.hypot(x, y) / 2,
            height=z,
        )
    if name == "sorter":
        parts = basin_parts((0.43, 0.28), 0.17)
        # A round aperture on the left; a square aperture on the right. The
        # segmented annulus fills its half-lid without sealing the opening.
        parts += [
            box("lid_front", (0.43, 0.095, 0.018), (0, -0.0925, 0.17)),
            box("lid_back", (0.43, 0.095, 0.018), (0, 0.0925, 0.17)),
            box("lid_left", (0.067, 0.09, 0.018), (-0.1815, 0, 0.17)),
            box("lid_middle", (0.120, 0.09, 0.018), (0, 0, 0.17)),
            box("lid_right", (0.065, 0.09, 0.018), (0.1825, 0, 0.17)),
        ]
        for i in range(48):
            a = 2 * math.pi * i / 48
            parts.append(
                box(
                    f"round_entry_{i}",
                    (0.025, 0.010, 0.018),
                    (-0.104 + 0.0525 * math.cos(a), 0.0525 * math.sin(a), 0.1701),
                    "oak_light",
                    (0, 0, a),
                )
            )
        return Model(
            [Link("body", parts, 2.0)],
            fixed=True,
            regions={
                "round": ("body", (-0.104, 0, 0.17), (0.040, 0.040, 0.012)),
                "square": ("body", (0.105, 0, 0.17), (0.045, 0.045, 0.012)),
                "inside": ("body", (0, 0, 0.08), (0.198, 0.123, 0.067)),
            },
            radius=0.26,
            height=0.18,
        )
    if name == "mug":
        parts = [
            cylinder("bottom", 0.045, 0.012, (0, 0, 0.006), "ceramic"),
            *ring("wall", 0.043, 0.007, 0.105, 0.057, "ceramic"),
        ]
        # Handle lies in XZ. Its open center is at x=.074, z=.060.
        for i in range(40):
            a = 2 * math.pi * i / 40
            parts.append(
                sphere(
                    f"handle_{i}",
                    0.008,
                    (0.070 + 0.032 * math.cos(a), 0, 0.060 + 0.039 * math.sin(a)),
                    "ceramic",
                )
            )
        return Model(
            [Link("body", parts, 0.15, angular_damping=0.8)],
            regions={"handle": ("body", (0.07, 0, 0.06), (0.022, 0.014, 0.029))},
            radius=0.105,
            height=0.115,
        )
    if name == "mug_rack":
        return Model(
            [
                Link(
                    "body",
                    [
                        cylinder("base", 0.13, 0.025, (0, 0, 0.0125), "walnut"),
                        cylinder("post", 0.018, 0.33, (0, 0, 0.18), "walnut"),
                        cylinder(
                            "hook", 0.009, 0.15, (0, -0.075, 0.32), "brass", (math.pi / 2, 0, 0)
                        ),
                        sphere("hook_tip", 0.012, (0, -0.146, 0.327), "brass"),
                    ],
                    2.0,
                )
            ],
            fixed=True,
            regions={"hook": ("body", (0, -0.105, 0.32), (0.015, 0.045, 0.02))},
            height=0.35,
        )
    if name in ("scraper", "mallet"):
        parts = [cylinder("handle", 0.018, 0.19, (0, 0, 0.12), "walnut")]
        if name == "scraper":
            parts += [box("blade", (0.18, 0.018, 0.075), (0, 0, 0.245), "metal")]
        else:
            parts += [cylinder("head", 0.039, 0.14, (0, 0, 0.24), "oak_light", (0, math.pi / 2, 0))]
        return Model([Link("body", parts, 0.13)], radius=0.10, height=0.285)
    if name == "tool_stand":
        return Model(
            [Link("body", basin_parts((0.25, 0.35), 0.025, "walnut"), 0.4)],
            fixed=True,
            regions={"inside": ("body", (0, 0, 0.10), (0.12, 0.17, 0.15))},
            radius=0.22,
            height=0.025,
        )
    if name == "collection_pan":
        parts = [
            p for p in basin_parts((0.34, 0.29), 0.05, "teal") if p.name not in ("front", "floor")
        ]
        parts += [
            box("floor", (0.34, 0.29, 0.004), (0, 0, 0.002), "teal"),
            box("lip", (0.34, 0.035, 0.003), (0, -0.16, 0.0015), "metal"),
        ]
        return Model(
            [Link("body", parts, 0.3)],
            fixed=True,
            regions={"inside": ("body", (0, 0, 0.05), (0.15, 0.125, 0.07))},
            height=0.05,
        )
    if name in ("jar", "jar_lid"):
        if name == "jar":
            parts = [
                cylinder("bottom", 0.061, 0.01, (0, 0, 0.005), "teal"),
                *ring("wall", 0.058, 0.009, 0.14, 0.074, "teal"),
            ]
            return Model([Link("body", parts, 0.15)], radius=0.063, height=0.15)
        parts = [
            cylinder("lid", 0.064, 0.015, (0, 0, 0.0075), "oak_light"),
            cylinder("knob", 0.019, 0.026, (0, 0, 0.026), "walnut"),
        ]
        return Model([Link("body", parts, 0.055)], radius=0.064, height=0.04)
    if name.startswith("weight_"):
        n = int(name[-1])
        radius = {1: 0.023, 2: 0.029, 4: 0.037}[n]
        return Model(
            [
                Link(
                    "body",
                    [
                        cylinder("weight", radius, 0.055, (0, 0, 0.0275), "brass"),
                        sphere("grip", 0.018, (0, 0, 0.065), "brass"),
                    ],
                    0.06 * n,
                )
            ],
            radius=radius,
            height=0.083,
        )
    if name == "balance_scale":
        def suspended_pan():
            # Keep the loaded center of mass below the suspension pivot.
            # A pivot just above the empty floor tips when a weight is added.
            return tray_parts(0.105, 0.025, "metal") + [
                cylinder(f"hanger_{i}", 0.003, 0.155, (0, y, 0.0775), "brass")
                for i, y in enumerate((-0.099, 0.099))
            ] + [
                cylinder("hanger_top", 0.003, 0.198, (0, 0, 0.155), "brass", (math.pi / 2, 0, 0))
            ]

        links = [
            Link(
                "base",
                [
                    box("plinth", (0.47, 0.24, 0.025), (0, 0, 0.0125), "walnut"),
                    box("pillar", (0.035, 0.05, 0.27), (0, 0, 0.15), "brass"),
                ],
                2.0,
            ),
            Link(
                # A center of mass below the pivot restores the beam toward
                # level once the loads match, as on a physical balance scale.
                "beam",
                [box("beam", (0.46, 0.025, 0.018), (0, 0, -0.025), "brass")],
                0.12,
                (0, 0, 0.285),
            ),
            Link(
                "reference",
                suspended_pan()
                + [cylinder("reference_load", 0.05, 0.04, (0, 0, 0.034), "brass")],
                0.24,
                (-0.22, 0, 0.13),
            ),
            Link("pan", suspended_pan(), 0.06, (0.22, 0, 0.13)),
        ]
        joints = [
            Joint(
                "balance", "base", "beam", "revolute", "Y", (0, 0, 0.285), (-0.24, 0.24), 0.03, 0.0
            ),
            Joint(
                "reference_pivot",
                "beam",
                "reference",
                "revolute",
                "Y",
                (-0.22, 0, 0.285),
                (-0.5, 0.5),
                0.004,
                0.0,
            ),
            Joint(
                "pan_pivot",
                "beam",
                "pan",
                "revolute",
                "Y",
                (0.22, 0, 0.285),
                (-0.5, 0.5),
                0.004,
                0.0,
            ),
        ]
        return Model(
            links, joints, True, {"pan": ("pan", (0, 0, 0.045), (0.087, 0.087, 0.07))}, height=0.34
        )
    if name == "pounding_bench":
        links = [
            Link(
                "base",
                [
                    box("top_front", (0.40, 0.072, 0.025), (0, -0.069, 0.12), "oak_light"),
                    box("top_back", (0.40, 0.072, 0.025), (0, 0.069, 0.12), "oak_light"),
                    box("top_middle", (0.13, 0.066, 0.025), (0, 0, 0.12), "oak_light"),
                    box("top_left", (0.075, 0.066, 0.025), (-0.1625, 0, 0.12), "oak_light"),
                    box("top_right", (0.075, 0.066, 0.025), (0.1625, 0, 0.12), "oak_light"),
                    box("left_leg", (0.035, 0.21, 0.12), (-0.18, 0, 0.06), "oak"),
                    box("right_leg", (0.035, 0.21, 0.12), (0.18, 0, 0.06), "oak"),
                ],
                2.0,
            )
        ]
        joints = []
        for n, x in (("peg_a", -0.095), ("peg_b", 0.095)):
            links.append(
                Link(
                    n,
                    [cylinder("peg", 0.024, 0.095, (0, 0, 0), "blue" if n == "peg_a" else "red")],
                    0.025,
                    (x, 0, 0.085),
                )
            )
            joints.append(
                Joint(n, "base", n, "prismatic", "Z", (x, 0, 0.085), (0, 0.07), 0.4, 0.35)
            )
        return Model(links, joints, True, height=0.20)
    if name == "drawer_cabinet":
        base = [
            box("left", (0.035, 0.55, 0.82), (-0.40, 0, 0.41), "oak"),
            box("right", (0.035, 0.55, 0.82), (0.40, 0, 0.41), "oak"),
            box("back", (0.80, 0.025, 0.82), (0, 0.27, 0.41), "walnut"),
            box("top", (0.85, 0.58, 0.035), (0, 0, 0.84), "oak_light"),
            box("bottom", (0.80, 0.55, 0.035), (0, 0, 0.58), "oak"),
        ]
        drawer = basin_parts((0.71, 0.45), 0.13, "oak_light") + [
            box("face", (0.80, 0.035, 0.19), (0, -0.24, 0.065), "oak"),
            cylinder("pull", 0.012, 0.23, (0, -0.29, 0.08), "brass", (0, math.pi / 2, 0)),
        ]
        return Model(
            [Link("base", base, 8.0), Link("drawer", drawer, 0.8, (0, 0, 0.61))],
            [
                Joint(
                    "slide", "base", "drawer", "prismatic", "Y", (0, 0, 0.61), (-0.38, 0), 2.0, 0.1
                )
            ],
            True,
            {"inside": ("drawer", (0, 0, 0.075), (0.335, 0.205, 0.065))},
            height=0.86,
        )
    if name == "trash_bin":
        body = basin_parts((0.32, 0.30), 0.77, "teal")
        lid = [
            box("lid", (0.35, 0.33, 0.025), (0, -0.15, 0), "oak_light"),
            cylinder("handle", 0.012, 0.12, (0, -0.24, 0.026), "brass", (0, math.pi / 2, 0)),
        ]
        return Model(
            [Link("base", body, 3.0), Link("lid", lid, 0.12, (0, 0.15, 0.79))],
            [
                Joint(
                    "hinge", "base", "lid", "revolute", "X", (0, 0.15, 0.79), (-1.7, 0), 0.08, 0.02
                )
            ],
            True,
            height=0.83,
        )
    if name == "door":
        panel = [
            box("panel", (0.045, 1.38, 2.12), (0, 0.69, 1.06), "oak_light"),
            box("inset", (0.048, 1.13, 1.65), (0, 0.69, 1.12), "cream", collision=False),
            cylinder("handle", 0.014, 0.16, (-0.06, 1.18, 1.0), "brass", (math.pi / 2, 0, 0)),
        ]
        return Model(
            [
                Link("base", [box("hinge_post", (0.11, 0.11, 2.2), (0, 0, 1.1), "walnut")], 10.0),
                Link("panel", panel, 3.0),
            ],
            [Joint("hinge", "base", "panel", "revolute", "Z", (0, 0, 0), (-1.8, 0), 0.3, 0.1)],
            True,
            height=2.2,
        )
    if name == "lamp":
        base = [
            cylinder("foot", 0.13, 0.025, (0, 0, 0.0125), "black"),
            cylinder("stem", 0.015, 0.32, (0, 0, 0.18), "brass"),
        ]
        head = [
            cylinder("shade", 0.075, 0.055, (0, -0.045, 0), "teal", (math.pi / 2, 0, 0)),
            cylinder("emitter", 0.063, 0.003, (0, -0.074, 0), "paper", (math.pi / 2, 0, 0), False),
        ]
        switch = [box("switch", (0.055, 0.040, 0.015), (0, 0, 0), "cream")]
        return Model(
            [
                Link("base", base, 2.0),
                Link("head", head, 0.08, (0, 0, 0.34)),
                Link("switch", switch, 0.015, (0.07, -0.04, 0.04)),
            ],
            [
                Joint("aim", "base", "head", "revolute", "X", (0, 0, 0.34), (0.0, 1.4), 0.03, 0.03),
                Joint(
                    "power",
                    "base",
                    "switch",
                    "revolute",
                    "Y",
                    (0.07, -0.04, 0.04),
                    (-0.35, 0.35),
                    0.01,
                    0.02,
                ),
            ],
            True,
            height=0.42,
        )
    if name in ("cart", "chair"):
        cart = name == "cart"
        base = [
            box("deck", (0.55, 0.48, 0.035), (0, 0, 0.22), "oak_light"),
            box("back", (0.55, 0.035, 0.44), (0, 0.22, 0.46), "teal"),
        ]
        if cart:
            base += [
                box("upper_shelf", (0.55, 0.48, 0.025), (0, 0, 0.62), "oak_light"),
                *[
                    cylinder(f"handle_post_{i}", 0.014, 0.27, (x, -0.25, 0.75), "metal")
                    for i, x in enumerate((-0.23, 0.23))
                ],
                cylinder("handle", 0.017, 0.46, (0, -0.25, 0.89), "metal", (0, math.pi / 2, 0)),
            ]
        else:
            base += [
                box("seat", (0.51, 0.43, 0.07), (0, -0.025, 0.49), "fabric"),
                box("backrest", (0.51, 0.07, 0.37), (0, 0.19, 0.73), "fabric"),
            ]
        links = [Link("base", base, 2.2 if cart else 2.0)]
        joints = []
        for i, (x, y) in enumerate(((-0.23, -0.19), (0.23, -0.19), (-0.23, 0.19), (0.23, 0.19))):
            pivot = f"caster_{i}"
            wheel = f"wheel_{i}"
            links += [
                Link(
                    pivot,
                    [box("fork", (0.035, 0.045, 0.06), (0, 0, -0.015), "metal")],
                    0.04,
                    (x, y, 0.13),
                ),
                Link(
                    wheel,
                    [cylinder("tire", 0.06, 0.025, (0, 0, 0), "rubber", (0, math.pi / 2, 0))],
                    0.04,
                    (x, y - 0.025, 0.06),
                ),
            ]
            joints += [
                Joint(
                    pivot,
                    "base",
                    pivot,
                    "revolute",
                    "Z",
                    (x, y, 0.13),
                    (-1000.0, 1000.0),
                    0.005,
                    0.001,
                ),
                Joint(
                    wheel,
                    pivot,
                    wheel,
                    "revolute",
                    "X",
                    (x, y - 0.025, 0.06),
                    (-1000.0, 1000.0),
                    0.002,
                    0.001,
                ),
            ]
        return Model(
            links,
            joints,
            False,
            {"handle": ("base", (0, -0.25, 0.89), (0.25, 0.025, 0.025))} if cart else {},
            radius=0.38,
            height=0.93,
        )
    raise ValueError(f"Unknown household asset: {name}")


MODEL_NAMES = frozenset(
    (
        "can_tomato",
        "can_beans",
        "apple",
        "cracker_box",
        "cylinder_block",
        "cube_block",
        "blue_block",
        "red_block",
        "small_block",
        "plate",
        "dish",
        "bowl",
        "tray",
        "round_mat",
        "rectangular_mat",
        "book",
        "open_book",
        "remote",
        "parcel",
        "sorter",
        "mug",
        "mug_rack",
        "scraper",
        "mallet",
        "tool_stand",
        "collection_pan",
        "jar",
        "jar_lid",
        "weight_1",
        "weight_2",
        "weight_4",
        "balance_scale",
        "pounding_bench",
        "drawer_cabinet",
        "trash_bin",
        "door",
        "lamp",
        "cart",
        "chair",
    )
)

ROOM_BOUNDS = ((-5.2, -4.5), (7.6, 4.5))
PORTAL_X = 5.0
CART_BAY_INTERIOR = ((3.205, -2.595), (3.995, -1.775))

SURFACES = {
    "activity_table": ((-2.0, 2.0), (1.5, 0.8), 0.82),
    "dining_table": ((2.0, 2.0), (1.5, 0.85), 0.82),
    "reading_desk": ((-2.0, -2.5), (1.5, 0.8), 0.82),
    "kitchen_island": ((2.0, -2.3), (1.5, 0.85), 0.90),
    "sideboard": ((-4.0, 0.0), (0.65, 1.5), 0.85),
    "book_shelf": ((0.0, 3.65), (1.2, 0.38), 0.95),
    "parcel_console": ((3.75, 0.9), (0.65, 1.2), 0.85),
    "entry_console": ((6.8, 0.9), (0.65, 1.2), 0.85),
}
