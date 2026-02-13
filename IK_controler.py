from controller import Supervisor
import torch
import torch.nn as nn

# 1. Define the same NN Architecture
class IKNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )
    def forward(self, x):
        return self.model(x)

# 2. Initialize Robot as Supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# 3. Load the Trained Model
model = IKNetwork()
model.load_state_dict(torch.load("ur5_ik_model.pth"))
model.eval()

# 4. Get Robot Motors
joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
motors = [robot.getDevice(name) for name in joint_names]

# 5. Get Reference to the Target Ball
target_node = robot.getFromDef("TARGET_BALL")

# Main Control Loop
while robot.step(timestep) != -1:
    # Get the current (x, y, z) of the red ball
    target_xyz = target_node.getField("translation").getSFVec3f()
    
    # Convert to Tensor for the Neural Network
    input_tensor = torch.FloatTensor(target_xyz)
    
    # Predict Joint Angles
    with torch.no_grad():
        predicted_angles = model(input_tensor).tolist()
    
    # Apply Angles to Motors
    for i, motor in enumerate(motors):
        motor.setPosition(predicted_angles[i])