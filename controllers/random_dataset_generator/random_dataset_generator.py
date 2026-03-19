"""
3DOF Robot Dataset Generator (Forward Kinematics)

This script runs inside a Webots Supervisor controller and generates a dataset
for training an inverse kinematics (IK) model.

For each random joint configuration:
1. Random joint angles are sampled within joint limits
2. The robot moves to the configuration
3. The physics engine settles
4. The end-effector position is recorded
5. The sample is saved if it is collision-free

Output dataset format:
[x, y, z, q1, q2, q3]
"""

# ========================= IMPORTS =========================

from controller import Supervisor
import random
import csv
import math
import sys

# ========================= CONFIGURATION =========================

TARGET_SAMPLES = 5000
SETTLE_STEPS = 35

OUTPUT_FILE = "3DOF_ik_dataset_5k.csv"

# Joint limits (radians)
JOINT_LIMITS = [
    (-3.0, 3.0),    # q1 — Base rotation
    (-1.8, -0.7),   # q2 — Shoulder
    (0.8, 2.4)      # q3 — Elbow
]

# ========================= INITIALISE WEBOTS =========================

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

robot_node = supervisor.getSelf()

# ========================= ROBOT JOINT SETUP =========================

joint_names = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint'
]

motors = []

for name in joint_names:

    device = supervisor.getDevice(name)

    if device is None:
        print(f"Error: motor {name} not found")
        sys.exit(1)

    motors.append(device)

# ========================= END EFFECTOR NODE =========================

end_effector_node = supervisor.getFromDef("GPS_TP")

if end_effector_node is None:
    print("Error: Could not find end effector node.")
    sys.exit(1)

# ========================= HELPER FUNCTIONS =========================

def get_uniform_joint_sample():
    """
    Generate a random joint configuration within joint limits.
    """
    return [random.uniform(limit[0], limit[1]) for limit in JOINT_LIMITS]


def is_collision():
    """
    Check if robot is colliding with anything.
    """
    return len(robot_node.getContactPoints(True)) > 0


def is_elbow_up(q3):
    """
    Enforce elbow-up configuration.
    """
    return math.sin(q3) > 0


# ========================= DATASET GENERATION =========================

collected = 0

with open(OUTPUT_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["x", "y", "z", "q1", "q2", "q3"])

    while collected < TARGET_SAMPLES:

        # 1. Random configuration
        angles = get_uniform_joint_sample()

        if not is_elbow_up(angles[2]):
            continue

        # 2. Apply joint angles
        for i, motor in enumerate(motors):
            motor.setPosition(angles[i])

        # 3. Allow physics to settle
        for _ in range(SETTLE_STEPS):
            supervisor.step(timestep)

        # 4. Reject collisions
        if is_collision():
            continue

        # 5. Read end-effector position
        pos = end_effector_node.getPosition()

        x, y, z = pos[0], pos[1], pos[2]

        # 6. Save sample
        writer.writerow([x, y, z] + angles)

        collected += 1

        if collected % 100 == 0:
            remaining = TARGET_SAMPLES - collected
            print(f"Collected: {collected} | Remaining: {remaining}")

# ================================================================

print(f"Dataset generation completed. Saved to {OUTPUT_FILE}")