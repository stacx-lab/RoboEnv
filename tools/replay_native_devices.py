"""Replay source-native device commands, optionally rendering native HD video."""

from roboenv_runtime.adapters.native_replay import main, replay  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
