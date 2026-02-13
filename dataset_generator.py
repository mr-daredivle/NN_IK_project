from controller import Supervisor
import random
import csv
import math
from tqdm import tqdm

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# UR5e Joint Names
joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]
motors = [supervisor.getDevice(name) for name in joint_names]
gps = supervisor.getDevice('gps') 
gps.enable(timestep)

# Number of valid samples we want to collect
target_samples = 20000
collected_samples = 0

with open('ik_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'])

    progress_bar = tqdm(total=target_samples)
    
    while collected_samples < target_samples:
        # 1. Generate angles optimized for the upper hemisphere
        angles = [
            random.uniform(-math.pi, math.pi),     # q1: Base rotation (360 degrees)
            random.uniform(-math.pi, 0),            # q2: Shoulder (0 is upright, negative tilts forward)
            random.uniform(-math.pi, math.pi),     # q3: Elbow
            random.uniform(-math.pi, math.pi),     # q4: Wrist 1
            random.uniform(-math.pi, math.pi),     # q5: Wrist 2
            random.uniform(-math.pi, math.pi)      # q6: Wrist 3
        ]

        # 2. Set the motors
        for i, motor in enumerate(motors):
            motor.setPosition(angles[i])
        
        # 3. Wait for robot to move and physics to settle
        for _ in range(15): 
            supervisor.step(timestep)
        
        # 4. Get Position
        pos = gps.getValues() # [x, y, z]

        # 5. Only save if the end-effector is above the floor
        if pos[2] > 0.02:
            writer.writerow(pos + angles)
            collected_samples += 1
            progress_bar.update(1)
        else:
            # If it hits the floor, we don't save and try again
            continue

progress_bar.close()
print(f"Data collection complete. Saved {target_samples} valid samples.")