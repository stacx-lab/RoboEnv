"""Serve an openpi checkpoint, registering the configs our vendored openpi lacks.

Runs inside the openpi virtualenv (``RoboDojo/XPolicyLab/policy/Pi_05/openpi/.venv``)
and imports nothing from roboenv, so it needs no install there.  The vendored
openpi is the XPolicyLab/RoboDojo fork, which dropped the upstream LIBERO
configs and never had RoboCasa's; the entries below are copied from upstream
openpi (``pi05_libero``) and from ``robocasa-benchmark/openpi``
(``pi05_pretrain_human300``), so the public checkpoints load with exactly the
transforms and normalisation they were trained with.

    serve_openpi.py --config pi05_libero --dir <checkpoint> --port 8807
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import threading


def _register_missing(name: str) -> None:
    from openpi.training import config as _config

    if name in _config._CONFIGS_DICT:
        return
    import openpi.models.pi0_config as pi0_config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders

    if name == "pi05_libero":
        # upstream openpi main (2026-08): src/openpi/training/config.py
        cfg = _config.TrainConfig(
            name="pi05_libero",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
            data=_config.LeRobotLiberoDataConfig(
                repo_id="physical-intelligence/libero",
                base_config=_config.DataConfig(prompt_from_task=True),
                extra_delta_transform=False,
            ),
            batch_size=256,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            num_train_steps=30_000,
        )
    elif name == "pi05_pretrain_human300":
        here = pathlib.Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        from openpi_ext.robocasa_config import make_robocasa_config

        cfg = make_robocasa_config(name)
    else:
        raise SystemExit(f"config {name!r} is neither vendored nor known to serve_openpi.py")
    _config._CONFIGS_DICT[name] = cfg
    _config._CONFIGS.append(cfg)


class _ResettablePolicy:
    """``policy`` plus a per-connection sampling key, so replays are reproducible.

    openpi's :class:`Policy` keeps one JAX key and *splits* it on every inference
    (``self._rng, sample_rng = jax.random.split(self._rng)``), so the actions a
    checkpoint predicts depend on how many inferences that server process has already
    served: the same solution replayed against a warm server samples differently from a
    fresh one, and a certificate earned through ``vla_run`` would not reproduce.  An
    observation carrying ``__roboenv_reset_rng__`` (an integer seed, or ``True`` for 0)
    restores the key first, so every episode starts from the same point.  The flag is
    removed before the policy sees the observation.

    The key is kept **per connection**, not per process.  One server serves several
    clients — a campaign runs an agent per GPU and a sweep opens a socket per episode —
    and with a single shared chain two concurrent episodes interleave their draws and
    corrupt each other's rollouts (measured 2026-08-28: parallel RoboCasa screening runs
    were not evidence).  A connection gets its own chain the first time it resets, and
    the chain is dropped when the connection closes, so concurrent episodes are
    independent and each one still replays.

    Identity: the vendored openpi server is **asyncio and single-threaded**, so a thread
    id is the same for every connection and would defeat the whole point (measured
    2026-08-28: concurrent episodes still corrupted each other with a thread-keyed map).
    Each connection is handled by its own asyncio task, so the task is the key, with the
    thread as a fallback for a caller that is not inside a loop.
    """

    RESET_KEY = "__roboenv_reset_rng__"

    def __init__(self, policy):
        self._policy = policy
        self._keys: dict[object, object] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _connection() -> object:
        """What identifies one client: its asyncio task, else its thread."""
        try:
            import asyncio

            task = asyncio.current_task()
        except (RuntimeError, ImportError):
            task = None
        return id(task) if task is not None else threading.get_ident()

    @property
    def metadata(self):
        """The wrapped policy's metadata, plus the flag telling clients resets work."""
        meta = dict(getattr(self._policy, "metadata", {}) or {})
        meta["roboenv_rng_reset"] = True
        return meta

    def infer(self, obs, **kwargs):
        """Serve one inference on this connection's own sampling chain."""
        import jax

        connection = self._connection()
        seed = obs.pop(self.RESET_KEY, None) if isinstance(obs, dict) else None
        if seed is not None:
            with self._lock:
                self._keys[connection] = jax.random.key(0 if seed is True else int(seed))
        with self._lock:
            key = self._keys.get(connection)
        if key is None:  # a client that never resets shares the process chain, as before
            return self._policy.infer(obs, **kwargs)
        saved = self._policy._rng
        self._policy._rng = key
        try:
            result = self._policy.infer(obs, **kwargs)
        finally:
            with self._lock:
                self._keys[connection] = self._policy._rng
            self._policy._rng = saved
        return result

    def forget(self, connection: object | None = None) -> None:
        """Drop a finished connection's chain (called when its handler ends)."""
        with self._lock:
            self._keys.pop(connection if connection is not None else self._connection(), None)

    def __getattr__(self, name):
        return getattr(self._policy, name)


def main() -> None:
    """Parse arguments, register the config if needed, load the checkpoint, serve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dir", required=True, help="checkpoint directory (params/ + assets/)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--default-prompt", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    _register_missing(args.config)
    extra = {}
    stats_dir = pathlib.Path(args.dir) / "assets"
    if (stats_dir / "norm_stats.json").exists():
        # checkpoints whose stats sit directly under assets/ (RoboCasa365) instead of
        # assets/<asset_id>/ — pass them explicitly rather than let the config guess
        from openpi.shared import normalize as _normalize

        extra["norm_stats"] = _normalize.load(stats_dir)
        logging.info("norm stats from %s", stats_dir)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.dir, default_prompt=args.default_prompt, **extra
    )
    logging.info("serving %s from %s on %s:%d", args.config, args.dir, args.host, args.port)
    policy = _ResettablePolicy(policy)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
