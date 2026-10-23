"""Load released visual components omitted or flattened by the URDF export."""

import json

import numpy as np
from roboenv.provisioning.chemlab_assets import resource

from .geometry import component_links


def supplement(root, row, contract, model):
    import trimesh

    scene = model.scene
    directory = resource(root, row["contract"]).parent
    path = directory / "geometry_manifest.json"
    evidence, groups = {}, {}
    if not path.exists():
        return evidence, groups
    manifest = json.loads(path.read_text())
    components = manifest.get("components", {})
    if isinstance(components, list):
        components = {c["id"]: c for c in components}
    required = {b.get("component_id") for b in contract.get("model_bindings", {}).values()}
    for component in sorted(required - {None}):
        if any(
            scene.graph.transforms.parents[node] in component_links(contract, component)
            for node in scene.graph.nodes_geometry
        ):
            continue
        spec = components.get(component, {})
        origin, bounds = spec.get("origin_m"), spec.get("evaluated_bounds_world_m")
        local_bounds = spec.get("bounding_box_component_local")
        if bounds is None and origin is not None and local_bounds:
            bounds = {
                axis: (np.asarray(local_bounds[f"{axis}_m"]) + origin).tolist()
                for axis in ("min", "max")
            }
        mesh_path = directory / "meshes" / f"{component}.obj"
        if origin is None or not isinstance(bounds, dict) or not mesh_path.is_file():
            continue
        mesh_path = resource(root, mesh_path.relative_to(root).as_posix())
        meshes = trimesh.load(mesh_path, force="scene")
        # Confirm the published local-mesh/world-origin convention before use.
        expected = np.asarray([bounds["min"], bounds["max"]])
        if not np.allclose(meshes.bounds + origin, expected, atol=1e-5):
            evidence[component] = {"status": "unresolved", "reason": "manifest bounds differ"}
            continue
        parent = spec.get("parent", "root")
        parent = parent if parent in scene.graph.nodes else scene.graph.base_frame
        frame = np.eye(4)
        frame[:3, 3] = origin
        frame = np.linalg.inv(scene.graph[parent][0]) @ frame
        scene.graph.update(frame_to=component, frame_from=parent, matrix=frame)
        for index, node in enumerate(meshes.graph.nodes_geometry):
            transform, geometry = meshes.graph[node]
            scene.add_geometry(
                meshes.geometry[geometry],
                node_name=f"supplement_{component}_{index}",
                geom_name=f"supplement_{component}_{index}",
                parent_node_name=component,
                transform=transform,
            )
        evidence[component] = {
            "status": "loaded",
            "source": mesh_path.relative_to(root).as_posix(),
            "default_visible": spec.get("default_visible", True),
            "collision": "not present in released URDF",
        }
    if row["id"] == "instrument.ziath_dp5_mirage_2023":
        # The URDF flattens only the default 96-well pair into the carriage.
        # All four released OBJ variants use the same carriage frame. Restore
        # named base components too, including the process-controlled power LED.
        scene.delete_geometry(
            [
                scene.graph[node][1]
                for node in scene.graph.nodes_geometry
                if scene.graph.transforms.parents[node] in {"base_link", "rack_carriage"}
            ]
        )
        variants = [(v.name, "base_link", True) for v in model.link_map["base_link"].visuals]
        variants += [
            (f"{kind}_{size}", "rack_carriage", size == 96)
            for size in (24, 48, 96, 240)
            for kind in ("rack", "tubes")
        ]
        for component, parent, default_visible in variants:
            spec = components[component]
            mesh_path = resource(root, (directory / spec["path"]).relative_to(root).as_posix())
            mesh = trimesh.load(mesh_path, force="scene")
            expected = np.asarray([spec["bbox"][f"{axis}_m"] for axis in ("min", "max")])
            if not np.allclose(mesh.bounds, expected, atol=1e-5):
                raise ValueError(f"Released rack component bounds disagree: {component}")
            scene.graph.update(frame_to=component, frame_from=parent, matrix=np.eye(4))
            for index, node in enumerate(mesh.graph.nodes_geometry):
                transform, geometry = mesh.graph[node]
                target = f"supplement_{component}_{index}"
                scene.add_geometry(
                    mesh.geometry[geometry],
                    node_name=target,
                    geom_name=target,
                    parent_node_name=component,
                    transform=transform,
                )
            evidence[component] = {
                "status": "loaded",
                "source": mesh_path.relative_to(root).as_posix(),
                "default_visible": default_visible,
                "collision": "released URDF default-format proxy remains unchanged",
            }
    if row["id"] == "instrument.eppendorf_5910ri":
        # Reuse named objects from the release OBJ files. The public Blender
        # adapter selects the same large/fixed rotor and sample-carrier groups.
        pieces = {
            link: trimesh.Scene()
            for link in ("rotor", "rotor_slots", "sample_carriers", "display", "lid_lock")
        }
        for name in (
            "rotor",
            "rotor_fixed_angle",
            "rotor_slots",
            "sample_carriers",
            "display",
            "lid_lock",
        ):
            mesh = trimesh.load(
                directory / "meshes" / f"{name}.obj",
                force="scene",
                split_objects=True,
                group_material=False,
            )
            for node in mesh.graph.nodes_geometry:
                link = (
                    "rotor_slots"
                    if node.startswith("rotor_fixed_socket_")
                    else "rotor"
                    if name == "rotor_fixed_angle"
                    else name
                )
                transform, geometry = mesh.graph[node]
                pieces[link].add_geometry(
                    mesh.geometry[geometry], node_name=node, geom_name=node, transform=transform
                )
        for link, mesh in pieces.items():
            spec = components[link]
            expected = np.asarray(
                [spec["evaluated_bounds_world_m"][axis] for axis in ("min", "max")]
            )
            if not np.allclose(mesh.bounds + spec["origin_m"], expected, atol=1e-5):
                raise ValueError(f"Released centrifuge component bounds disagree: {link}")
            old = [
                scene.graph[node][1]
                for node in scene.graph.nodes_geometry
                if scene.graph.transforms.parents[node] == link
            ]
            scene.delete_geometry(old)
            for node in mesh.graph.nodes_geometry:
                transform, geometry = mesh.graph[node]
                target = "split_" + node
                scene.add_geometry(
                    mesh.geometry[geometry],
                    node_name=target,
                    geom_name=target,
                    parent_node_name=link,
                    transform=transform,
                )
                groups[target] = node
        evidence["rotor_configuration"] = {
            "status": "loaded",
            "source": row["path"] + "/meshes",
            "mesh_count": len(groups),
            "collision": "released URDF proxy remains unchanged",
        }
    return evidence, groups
