"""Version-4 asset bindings in the accepted household layout (metres, Z up).

SIMPLE supplies the scanned props and HSSD furniture/materials. Task-specific
mechanisms retain their native joints and functional geometry. The downloaded
house is a parts source, not a claim that this is its original floor plan.
"""

import math

from .household_models import Part, cylinder, model as authored_model

# GraspNet points are already metres/Z-up despite their USD metadata. Rotate
# the cracker box so its broad front is local -Y, then fit the accepted size.
SCANS = {
    "can_tomato": (2, 0, (0.069, 0.069, 0.115)),
    "bowl": (6, 0, (0.260, 0.260, 0.090)),
    "apple": (12, 0, (0.086, 0.086, 0.093)),
    "cracker_box": (0, -90, (0.110, 0.065, 0.180)),
}

FURNITURE = {
    "table": "eeaf34edd2065a3fa2af3fc021cd343ca029f696",
    "island": "af102d5ce3df9cc58b991eb618a86f66c5901efe",
    "sideboard": "acfe3955254d50d2134fa4ff31493b7d3f83cb0e",
    "console": "node_dc4fc2ba44e670900db84b7fe9e7e98c8c7b338",
    "sofa": "node_dac60e35797491cce17b7c74decb712f9ea17a9",
}
SURFACE_ASSETS = {
    "activity_table": ("table", 0),
    "dining_table": ("table", 0),
    "reading_desk": ("table", 0),
    "kitchen_island": ("island", 0),
    "sideboard": ("sideboard", 90),
    "parcel_console": ("console", 90),
    "entry_console": ("console", 90),
}
# Diffuse textures from the pinned HSSD house, never normal/ORM maps.
TEXTURES = {"oak": 128, "oak_light": 195, "walnut": 145, "fabric": 121,
            "cream": 246, "cardboard": 131}


def model(name):
    """Functional geometry; joints and semantic part names remain stable."""
    definition = authored_model(name)
    if name == "bowl":
        # Radial cross-section measured from GraspNet 006, fitted to the accepted
        # bowl diameter/height. Convex wedges preserve the open cavity. The
        # bottom is a separate load-bearing part used by the containment test.
        profile = ((.004, .0735, .0755), (.006, .07537, .07719),
                   (.010, .07932, .08128), (.015, .08454, .08659),
                   (.025, .09349, .09527), (.035, .10034, .10214),
                   (.045, .10571, .10735), (.055, .11029, .11197),
                   (.065, .11368, .11526), (.075, .11583, .11734),
                   (.080, .11684, .11831), (.084, .118, .129),
                   (.087, .121, .130), (.090, .125, .128))
        parts = [cylinder("floor", .0758, .004, (0, 0, .002), "ceramic")]
        for level, (lower, upper) in enumerate(zip(profile, profile[1:])):
            for i in range(40):
                # The working wall follows measured triangle cross-sections
                # within 3 mm. The scan's irregular rolled lip is approximated
                # separately; it is never a valid containment support surface.
                angles = (2 * math.pi * i / 40 - 0.0005,
                          2 * math.pi * (i + 1) / 40 + 0.0005)
                points = tuple((r * math.cos(a), r * math.sin(a), z)
                               for z, inner, outer in (lower, upper)
                               for r in (inner, outer)
                               for a in angles)
                parts.append(Part(f"wall_{level}_{i}", "hull", points, material="ceramic"))
        definition.links[0].parts = parts
        definition.regions["inside"] = ("body", (0, 0, 0.045), (0.116, 0.116, 0.045))
    return definition
