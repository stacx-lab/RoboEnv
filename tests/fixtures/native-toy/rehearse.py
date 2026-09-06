# ruff: noqa: F821
token = checkpoint()
push(0.3)
restore(token)
assert get_object_pose("puck")[0][0] == 0.0
push(0.3)
