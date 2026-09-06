"""Shared public-controller classification; no worker or device dependencies."""

READ_ONLY_PREFIXES = ("get_", "list_", "check_", "is_", "observe", "render", "peek")


def is_read_only(name):
    return name.startswith(READ_ONLY_PREFIXES) or name == "tcp"
