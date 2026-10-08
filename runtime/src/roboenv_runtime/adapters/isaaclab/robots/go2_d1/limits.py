"""D1-550 arm actuator facts and the simulation velocity policy.

Unitree numbers the six arm motors J0--J5; the URDF uses d1_Joint1--6.
The published torque table does not distinguish continuous, peak or stall torque.
The gripper motor torque cannot be used as a prismatic finger force.
"""

D1_SPECIFICATION_URL = "https://support.unitree.com/home/en/developer/D1Arm_services"
D1_SPECIFICATION_CONTENT_URL = "https://doc-cdn.unitree.com/6/109/en/6_109_en"
ARM_EFFORT_LIMITS = {
    "d1_Joint1": 3.3,
    "d1_Joint2": 3.3,
    "d1_Joint3": 1.7,
    "d1_Joint4": 1.7,
    "d1_Joint5": 1.7,
    "d1_Joint6": 1.7,
}
# None in Isaac Lab inherits the USD limit, including old practice bundles' 1.5.
# Infinity explicitly disables the arm joint speed cap, even for cached assets.
ARM_VELOCITY_LIMIT_SIM = float("inf")
