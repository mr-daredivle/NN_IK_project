# ===============================================================
# WEBOTS IMPORTS
# ===============================================================
from controller import Supervisor

# ===============================================================
# PYTHON / ML IMPORTS
# ===============================================================
import torch
import torch.nn as nn
import numpy as np
import sys

# ===============================================================
# 1. MODEL DEFINITION (Must match Trainer Architecture)
# ===============================================================
class IKNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.model(x)


# ===============================================================
# 2. CONFIGURATION
# ===============================================================

# IMPORTANT: Must match the values printed during training
X_mean = torch.tensor([0.06502049, 0.00085565, 0.43471676])
X_std  = torch.tensor([0.30071208, 0.34254619, 0.20711817])

MODEL_PATH = "ik_model_3dof.pth"

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint'
]


# ===============================================================
# 3. INITIALIZE WEBOTS
# ===============================================================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

device = torch.device("cpu")


# ===============================================================
# 4. LOAD NEURAL NETWORK
# ===============================================================
model = IKNetwork().to(device)

try:
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    print(f"[SUCCESS] 3-DOF IK Model '{MODEL_PATH}' loaded.")

except Exception as e:
    print("\n[FATAL ERROR] Model Architecture Mismatch!")
    print("Ensure the .pth file matches the 3-DOF network.")
    print(e)
    sys.exit(1)


# ===============================================================
# 5. MOTOR SETUP
# ===============================================================
motors = []

for name in JOINT_NAMES:
    motor = robot.getDevice(name)

    if motor:
        motors.append(motor)
    else:
        print(f"[WARNING] Motor '{name}' not found!")

joint_min = [m.getMinPosition() for m in motors]
joint_max = [m.getMaxPosition() for m in motors]


# ===============================================================
# 6. NODE REFERENCES
# ===============================================================
target_node = robot.getFromDef("TARGET_BALL")
robot_node  = robot.getSelf()
ee_node     = robot.getFromDef("END_EFFECTOR_TIP")

if not target_node:
    print("WARNING: TARGET_BALL DEF not found!")

if not ee_node:
    print("WARNING: END_EFFECTOR_TIP DEF not found!")

print("\n--- 3-DOF Neural IK Controller Started ---")


# ===============================================================
# 7. MAIN CONTROL LOOP
# ===============================================================
step_count = 0

while robot.step(timestep) != -1:

    step_count += 1

    if target_node is None:
        continue

    # -----------------------------------------------------------
    # 1. Get target position relative to robot base
    # -----------------------------------------------------------
    pose = target_node.getPose(robot_node)

    target_xyz = torch.tensor(
        [pose[3], pose[7], pose[11]],
        dtype=torch.float32
    )

    # -----------------------------------------------------------
    # 2. Normalize input
    # -----------------------------------------------------------
    input_norm = (target_xyz - X_mean) / X_std

    # -----------------------------------------------------------
    # 3. Neural Network Prediction
    # -----------------------------------------------------------
    with torch.no_grad():

        # add batch dimension
        predicted_norm = model(input_norm.unsqueeze(0))[0]

    # Convert normalized output back to radians
    predicted_angles = predicted_norm * np.pi


    # -----------------------------------------------------------
    # 4. Apply Motor Commands with Safety Clamping
    # -----------------------------------------------------------
    applied_angles = []

    for i, motor in enumerate(motors):

        raw_angle = predicted_angles[i].item()

        safe_angle = np.clip(
            raw_angle,
            joint_min[i],
            joint_max[i]
        )

        motor.setPosition(safe_angle)
        applied_angles.append(safe_angle)


    # -----------------------------------------------------------
    # 5. DEBUG OUTPUT (Optional)
    # -----------------------------------------------------------
    full_debug = False

    if full_debug and step_count % 100 == 0:

        print("\n" + "="*50)
        print(f"DEBUG STEP: {step_count}")

        print(
            f"Target (Local m): "
            f"X:{target_xyz[0]:.3f} "
            f"Y:{target_xyz[1]:.3f} "
            f"Z:{target_xyz[2]:.3f}"
        )

        print("\nJOINT BREAKDOWN")

        for i in range(len(JOINT_NAMES)):

            status = "OK"

            if abs(predicted_angles[i] - applied_angles[i]) > 0.1:
                status = "CLIPPED"

            print(
                f"{JOINT_NAMES[i]:18} | "
                f"AI:{predicted_angles[i]:6.2f} | "
                f"APPLIED:{applied_angles[i]:6.2f} | "
                f"LIM[{joint_min[i]:.2f},{joint_max[i]:.2f}] "
                f"{status}"
            )