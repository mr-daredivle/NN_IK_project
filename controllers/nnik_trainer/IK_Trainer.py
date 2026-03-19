import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ===============================
# 1. ROBUST MODEL (3-DOF)
# ===============================
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

            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.model(x)


# ===============================
# 2. DATA LOADING & STATS
# ===============================
data = pd.read_csv('3DOF_ik_dataset_50k.csv')

X = data[['x', 'y', 'z']].values.astype(np.float32)
y = data[['q1', 'q2', 'q3']].values.astype(np.float32)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8

print("\n" + "="*60)
print("COPY AND PASTE THESE INTO YOUR ROBOT CONTROLLER SCRIPT:")
print(f"X_mean = torch.tensor([{X_mean[0]:.8f}, {X_mean[1]:.8f}, {X_mean[2]:.8f}])")
print(f"X_std  = torch.tensor([{X_std[0]:.8f}, {X_std[1]:.8f}, {X_std[2]:.8f}])")
print("="*60 + "\n")

# Normalize inputs
X_norm = (X - X_mean) / X_std

# Normalize outputs
y_norm = y / np.pi

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_norm,
    test_size=0.1,
    shuffle=True
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=256,
    shuffle=True
)

# ===============================
# 3. TRAINING SETUP
# ===============================
device = torch.device("cpu")

model = IKNetwork().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-4
)

criterion = nn.SmoothL1Loss()

# ===============================
# TRAINING TIMER START
# ===============================
start_time = time.time()

print("Starting training (3-DOF)...")

for epoch in range(101):

    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        output = model(batch_x)

        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 20 == 0:

        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.6f}")

# ===============================
# TRAINING TIMER END
# ===============================
end_time = time.time()
training_time = end_time - start_time


# ===============================
# FINAL TEST
# ===============================
model.eval()

with torch.no_grad():

    test_outputs = model(torch.tensor(X_test))

    test_loss = criterion(test_outputs, torch.tensor(y_test))

    print(f"\nFinal Test Loss: {test_loss.item():.6f}")


# ===============================
# SAVE MODEL
# ===============================
torch.save(model.state_dict(), "ik_model_3dof.pth")

print("Model saved as ik_model_3dof.pth")

# ===============================
# FINAL TRAINING TIME OUTPUT
# ===============================
minutes = training_time / 60

print(f"\nTraining completed in {training_time:.2f} seconds ({minutes:.2f} minutes)")