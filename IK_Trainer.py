import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def dh_transform(a, d, alpha, theta):
    device = theta.device
    batch = theta.shape[0]

    # convert scalars → tensors
    a = torch.full((batch,), a, device=device)
    d = torch.full((batch,), d, device=device)
    alpha = torch.full((batch,), alpha, device=device)

    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    T = torch.stack([
        torch.stack([ct, -st*ca,  st*sa, a*ct], dim=-1),
        torch.stack([st,  ct*ca, -ct*sa, a*st], dim=-1),
        torch.stack([torch.zeros_like(ct), sa, ca, d], dim=-1),
        torch.stack([torch.zeros_like(ct), torch.zeros_like(ct), torch.zeros_like(ct), torch.ones_like(ct)], dim=-1)
    ], dim=-2)

    return T


def forward_kinematics(q):
    # q shape: (batch, 6)

    T = torch.eye(4).to(q.device).repeat(q.size(0), 1, 1)

    # example dummy params (replace with UR5 real ones)
    a = [0, 0.5, 0.3, 0, 0, 0]
    d = [0.1, 0, 0, 0.2, 0, 0.1]
    alpha = [0, 0, 0, 0, 0, 0]

    for i in range(6):
        Ti = dh_transform(a[i], d[i], alpha[i], q[:, i])
        T = T @ Ti

    pos = T[:, :3, 3]  # xyz only
    return pos


# 1. Load Data
data = pd.read_csv('200k_ik_data.csv')
X = data[['x', 'y', 'z']].values
y = data[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Define MLP Model
class IKNetwork(nn.Module):
    def __init__(self):
        super(IKNetwork, self).__init__()
        self.model  =nn.Sequential(
        nn.Linear(3, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 6)
)

    def forward(self, x):
        return self.model(x)

model = IKNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. Training Loop
for epoch in tqdm(range(20)):
    inputs = torch.FloatTensor(X_train)
    targets = torch.FloatTensor(y_train)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    
    pred_joints = outputs
    
    pred_xyz = forward_kinematics(pred_joints)

    joint_loss = criterion(pred_joints, targets)
    fk_loss = criterion(pred_xyz, inputs)

    loss = joint_loss + 0.5 * fk_loss
    
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0: print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "ur5_ik_model.pth")