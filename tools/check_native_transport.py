"""Exercise public HTTP fencing and concurrent native worlds through RoboEnv launch."""

import argparse
import copy
import dataclasses
import http.client
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import roboenv
import yaml
from roboenv.client import Client, RemoteError
from roboenv.tasks.resolve import resolve_task


def require(condition, message):
    if not condition:
        raise ValueError(message)


def owner_client(owner):
    record = owner.record
    return Client(record["socket"], session=record["session"], generation=record["generation"])


def reject(call, code=None):
    try:
        call()
    except RemoteError as error:
        if code is not None:
            require(error.code == code, f"Unexpected rejection: {error.code}")
        return error.code
    raise ValueError("Public transport accepted a prohibited request")


def target(client, name, offset):
    values = client.device(name).read()["data"]["positions"]
    joints = client.device(name).describe()["intrinsics"]["model"]["joints"]
    values = [
        min(j["limits"]["max"], max(j["limits"]["min"], q))
        for j, q in zip(joints, values, strict=True)
    ]
    for index, joint in enumerate(joints):
        limits = joint["limits"]
        for delta in (offset, -offset):
            if limits["min"] <= values[index] + delta <= limits["max"]:
                values[index] += delta
                return values
    raise ValueError("No joint has room for the bounded transport test target")


def camera_name(client, requested=None):
    cameras = sorted(
        item["name"] for item in client.devices()["devices"] if item["type"] in {"rgb", "rgbd"}
    )
    require(bool(cameras), "Transport checks require a native color camera")
    if requested is not None:
        require(requested in cameras, "Requested camera is not a native color camera")
        return requested
    return "head_camera" if "head_camera" in cameras else cameras[0]


def select_cases(task, seeds=(0, 1), instances=None):
    require(
        len(seeds) == 2 and all(type(seed) is int and seed >= 0 for seed in seeds),
        "Specify two nonnegative seeds",
    )
    if task.backend == "behavior":
        require(
            instances is not None
            and len(instances) == 2
            and all(type(value) is int and value >= 0 for value in instances)
            and instances[0] != instances[1],
            "BEHAVIOR requires two distinct explicit activity instance IDs; "
            "seeds do not select them",
        )
    else:
        require(instances is None, "Activity instance IDs are supported only for BEHAVIOR")
        require(seeds[0] != seeds[1], "Specify two distinct seeds for native layout selection")
    result = []
    for index, seed in enumerate(seeds):
        kwargs = dict(task.backend_kwargs)
        if instances is not None:
            kwargs["activity_instance_id"] = instances[index]
        result.append(dataclasses.replace(task, seed=seed, backend_kwargs=kwargs))
    return result


def exercise(owner, *, camera=None):
    public, private = owner.client, owner_client(owner)
    baseline = private.request("observe")
    camera = camera_name(public, camera)
    description = public.describe()
    require(
        not {"seed", "backend_kwargs", "owner_socket"}.intersection(description.get("task", {})),
        "Public task leaked owner configuration",
    )
    private.request("observe", request_id="private-state")
    private.request("operations.get", {"request_id": "private-state"}, request_id="owner-lookup")
    reject(
        lambda: public.request(
            "operations.get", {"request_id": "private-state"}, request_id="owner-lookup"
        ),
        "unsupported_op",
    )
    require(
        public.operation("owner-lookup")["state"] == "unknown",
        "Public operation lookup leaked owner result",
    )
    for op in ("observe", "score.peek", "call"):
        reject(lambda op=op: public.request(op))
    privileged = Client(owner.endpoint, session=public.session, generation=public.generation)
    privileged.allowed_ops |= {"submit", "shutdown", "reset", "restore"}
    for op in ("submit", "shutdown", "reset", "restore"):
        reject(lambda op=op: privileged.request(op))
    for name in ("left_arm", "right_arm", camera):
        public.device(name).call("load")
        public.device(name).read()
        public.device(name).call("status")
    require(
        private.request("observe")["state"] == baseline["state"], "Discovery/reads advanced physics"
    )
    goal = target(public, "left_arm", 0.1)
    # Duplicate concurrent requests share one native operation across both transports.
    with ThreadPoolExecutor(max_workers=3) as pool:
        replies = list(
            pool.map(
                lambda client: client.device("left_arm").write(
                    duration_s=len([goal]) / 50, request_id="dedup-motion", actions=[goal]
                ),
                [public, private, public],
            )
        )
    require(replies[0] == replies[1] == replies[2], "Duplicate request replies differ")
    public.device("world").wait(0.02)
    require(private.request("observe")["budget"]["used"] == 1, "Duplicate request advanced twice")
    reject(
        lambda: public.device("left_arm").write(
            duration_s=len([[0.0] * len(goal)]) / 1,
            request_id="dedup-motion",
            actions=[[0.0] * len(goal)],
        ),
        "request_id_conflict",
    )
    # Lose a real HTTP reply after sending; recover by request ID without redoing motion.
    identity = "lost-reply"
    request = {
        "protocol": "roboenv/1",
        "session": public.session,
        "generation": public.generation,
        "request_id": identity,
        "deadline_s": 60.0,
        "op": "devices.call",
        "args": {
            "device": "left_arm",
            "action": "write",
            "args": {"actions": [goal], "duration_s": len([goal]) / 50},
        },
    }
    body = (json.dumps(request) + "\n").encode()
    url = urlsplit(owner.endpoint)
    header = (
        f"POST / HTTP/1.1\r\nHost: {url.netloc}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode()
    with socket.create_connection((url.hostname, url.port), timeout=10) as connection:
        connection.sendall(header + body)
        connection.shutdown(socket.SHUT_WR)
        # Reading no response deliberately leaves the caller uncertain.
    deadline = time.monotonic() + 30
    while public.operation(identity)["state"] not in {"succeeded", "failed"}:
        require(time.monotonic() < deadline, "Lost-reply operation never settled")
        time.sleep(0.02)
    public.device("left_arm").write(
        duration_s=len([goal]) / 50, request_id=identity, actions=[goal]
    )
    public.device("world").wait(0.02)
    require(private.request("observe")["budget"]["used"] == 2, "Uncertain recovery repeated motion")
    left, right = target(public, "left_arm", 0.03), target(public, "right_arm", 0.05)
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [
            pool.submit(public.device(name).write, actions=[goal], duration_s=0.02)
            for name, goal in (("left_arm", left), ("right_arm", right))
        ]
        for future in calls:
            future.result()
    public.device("world").wait(0.02)
    require(
        private.request("observe")["budget"]["used"] == 3,
        "Concurrent writes share one world interval",
    )
    public.device("left_arm").stop()
    public.device("right_arm").stop()
    # Sensor products are retrievable; owner files and another episode are not.
    frame = public.device(camera).read()["data"]["file"]
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    try:
        prefix = f"/artifacts/{public.session}/{public.generation}/"
        connection.request("GET", prefix + Path(frame).name)
        response = connection.getresponse()
        require(
            response.status == 200 and response.read().startswith(b"\x89PNG"),
            "Sensor artifact is not a PNG",
        )
        for path in (prefix + "run.json", "/artifacts/another-episode/0/" + Path(frame).name):
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            require(response.status != 200, "Sensor endpoint exposed an owner/foreign artifact")
    finally:
        connection.close()
    return {
        "passed": True,
        "initial_state": baseline["initial_state"],
        "implementation": baseline["implementation"],
        "camera": camera,
        "budget": private.request("observe")["budget"],
        "checks": {
            key: True
            for key in (
                "owner_denial",
                "owner_cache_denial",
                "read_invariance",
                "concurrent_dedup",
                "lost_reply_recovery",
                "independent_concurrency",
                "sensor_artifacts",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--camera", help="Native RGB/RGBD device; default: discover from inventory")
    parser.add_argument("--seeds", type=int, nargs=2, default=(0, 1), metavar=("FIRST", "SECOND"))
    parser.add_argument(
        "--instances",
        type=int,
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help="Required distinct BEHAVIOR activity instance IDs",
    )
    args = parser.parse_args()
    task = resolve_task(args.task, project=args.project).task
    cases = select_cases(task, args.seeds, args.instances)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "roboenv/native-transport/1",
        "passed": False,
        "task": args.task,
        "scope": "public HTTP; no adversarial filesystem isolation",
        "cases": [{"seed": case.seed, "backend_kwargs": case.backend_kwargs} for case in cases],
    }
    owners = []
    try:
        for index, selected in enumerate(cases):
            path = out / f"task-{index}.yaml"
            path.write_text(yaml.safe_dump(dataclasses.asdict(selected)))
            owners.append(
                roboenv.launch(
                    path,
                    project=args.project,
                    interpreter=args.python,
                    timeout=args.timeout,
                    execution_mode="stepped",
                )
            )
        report["episodes"] = [copy.deepcopy(owner.record) for owner in owners]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(partial(exercise, camera=args.camera), owners))
        require(
            results[0]["initial_state"] != results[1]["initial_state"],
            "Concurrent episodes loaded identical initial states",
        )
        report["worlds"] = [
            {key: value for key, value in result.items() if key != "initial_state"}
            for result in results
        ]
        stale, old_endpoint = owners[0].client, owners[0].endpoint
        owners[0].close()
        fresh = roboenv.launch(
            out / "task-0.yaml",
            project=args.project,
            interpreter=args.python,
            timeout=args.timeout,
            port=urlsplit(old_endpoint).port,
        )
        owners.append(fresh)
        reject(lambda: stale.device("left_arm").read(), "session_mismatch")
        wrong = Client(
            fresh.endpoint,
            session=fresh.record["session"],
            generation=fresh.record["generation"] + 1,
        )
        reject(lambda: wrong.devices(), "generation_mismatch")
        fresh.device("left_arm").read()
        require(
            owner_client(fresh).request("observe")["budget"]["used"] == 0,
            "Rebinding advanced a new episode",
        )
        report.update(
            passed=True,
            endpoint_reuse_fenced=True,
            generation_fenced=True,
            concurrent_worlds=2,
            reset_by_fresh_episode=True,
        )
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        for owner in owners:
            owner.close()
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}))


if __name__ == "__main__":
    main()
