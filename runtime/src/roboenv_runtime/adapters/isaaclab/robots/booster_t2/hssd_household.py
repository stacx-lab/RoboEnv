"""Version-5 functional models in the complete HSSD residence.

The four calibrated scans keep their version-4 physical dimensions. Mechanism
names and contact regions are stable; the door is sized to the new aperture.
"""

import math

from .household_models import box, cylinder, ring
from .simple_household import model as calibrated_model


def model(name):
    definition = calibrated_model(name)
    if name == "trash_bin":
        # A continuous brushed-metal vessel, with an equally round physical
        # cavity. The accepted lid pivot/span and joint contract are retained.
        definition.links[0].parts = [
            cylinder("floor", .16, .014, (0, 0, .007), "metal"),
            *ring("wall", .154, .012, .77, .385, "metal", 64),
        ]
        definition.links[1].parts[0] = cylinder("lid", .17, .025, (0, -.15, 0), "metal")
    if name in ("cart", "chair"):
        parts = definition.links[0].parts
        parts[:] = [p for p in parts if p.name != "back" and (name == "cart" or p.name != "deck")]
        if name == "cart":
            parts += [cylinder(f"shelf_post_{i}", .012, .40, (x, y, .42), "metal")
                      for i, (x, y) in enumerate(((-.25, -.22), (.25, -.22), (-.25, .22), (.25, .22)))]
        else:
            parts.append(cylinder("column", .032, .31, (0, 0, .305), "metal"))
            for i, (x, y) in enumerate(((-.23, -.19), (.23, -.19), (-.23, .19), (.23, .19))):
                parts.append(box(f"spoke_{i}", (math.hypot(x, y), .03, .025),
                                 (x/2, y/2, .145), "metal", (0, 0, math.atan2(y, x))))
    if name == "door":
        # Panel dimensions are shared with the full-house doorway fixture.
        from .scenes import load_scene

        portal = load_scene("hssd_household").portal
        width, height = portal["width"], portal["height"]
        for link in definition.links:
            for part in link.parts:
                if part.name == "panel":
                    part.size = (.045, width, height)
                    part.pos = (0, width / 2, height / 2)
                elif part.name == "inset":
                    part.size = (.048, width - .20, height - .35)
                    part.pos = (0, width / 2, height / 2)
                elif part.name == "handle":
                    part.pos = (-.06, width - .16, 1.)
                elif part.name == "hinge_post":
                    # The scene owns the jamb. A second full-width base post
                    # would silently narrow the measured aperture.
                    part.shape, part.size = "cylinder", (.012, .20)
                    part.pos, part.material, part.collision = (0, 0, 1.0), "brass", False
        panel = definition.links[1].parts
        for side in (-1, 1):
            for level, z in enumerate((.53, 1.49)):
                for edge, y in enumerate((.13, width-.13)):
                    panel.append(box(f"moulding_{side}_{level}_{edge}".replace("-", "m"),
                                     (.006, .025, .70), (side*.025, y, z), "oak_light", collision=False))
                for edge, dz in enumerate((-.35, .35)):
                    panel.append(box(f"rail_{side}_{level}_{edge}".replace("-", "m"),
                                     (.006, width-.235, .025), (side*.025, width/2, z+dz),
                                     "oak_light", collision=False))
        definition.height = 2.2
    return definition
