"""Pinned SIMPLE scan, furniture and material bindings for the HSSD house."""

# GraspNet points are already metres/Z-up despite their USD metadata. Rotate
# the cracker box so its broad front is local -Y, then fit the accepted size.
SCANS = {
    "can_tomato": (2, 0, (0.069, 0.069, 0.115)),
    "bowl": (6, 0, (0.260, 0.260, 0.090)),
    "apple": (12, 0, (0.086, 0.086, 0.093)),
    "cracker_box": (0, -90, (0.110, 0.065, 0.180)),
}

FURNITURE = {"console": "node_dc4fc2ba44e670900db84b7fe9e7e98c8c7b338"}
# Diffuse textures from the pinned HSSD house, never normal/ORM maps.
TEXTURES = {"oak": 128, "oak_light": 195, "walnut": 145, "fabric": 121,
            "cream": 246, "cardboard": 131}
