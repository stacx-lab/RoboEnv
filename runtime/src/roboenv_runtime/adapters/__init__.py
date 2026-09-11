"""Simulator adapters.

Importing this package imports **no simulator**.  Each adapter module imports
its simulator lazily when the adapter is instantiated, so
``roboenv_runtime.core.make_adapter`` can name every adapter in every interpreter and
fail with a clear message only when the wrong one is chosen.

Built-in adapters (see :data:`roboenv_runtime.core.adapter.BUILTIN_ADAPTERS`):

======================  =====================================  ============
name                    simulator                              embodiment
======================  =====================================  ============
``fakeshell``           in-memory toy world (tests, demos)     general
``robodojo``            RoboDojo on Isaac Sim                  arx_x5 / yam
``robotwin``            RoboTwin 2.0 on SAPIEN                 piper / yam
``libero``              Official LIBERO on MuJoCo/robosuite         panda
``robocasa``            RoboCasa365 on MuJoCo/robosuite        panda_omron
``behavior``            BEHAVIOR-1K on OmniGibson/Isaac Sim    r1pro
======================  =====================================  ============
"""
