#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-asymptotic confidence intervals for neural network predictions (synthetic experiment)

This script implements the synthetic experiment from the paper:
"Non-asymptotic confidence intervals for neural network predictions via
sequential parameter estimation and conformal prediction" by Timofeev A.V.

It trains a neural network on a 1D sinusoidal regression problem, estimates
the theoretical constants (L0, sigma2, q, g) from the SGD trajectory, applies
a stopping rule based on prediction width, and finally constructs conformal
prediction intervals with guaranteed coverage.

Author: Timofeev A.V.
Date: 2026-09-03
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv

# ---------- 1. Data generation ----------
def generate_data(n, noise_std=0.1):
    x = np.random.uniform(0, 1, n).astype(np.float32)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise_std, n).astype(np.float32)
    return torch.tensor(x.reshape(-1, 1), dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

n_train, n_val, n_test = 1500, 400, 600
noise_std = 0.1
x_train, y_train = generate_data(n_train, noise_std)
x_val, y_val = generate_data(n_val, noise_std)
x_test, y_test = generate_data(n_test, noise_std)

# ---------- 2. Model definition ----------
class Net(nn.Module):
    def __init__(self, hidden=100):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = Net(hidden=100)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- 3. Training with trajectory storage ----------
epochs = 500
trajectory = []  # (state_dict, loss_val, grad_norm, grad_vec)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    grad_vec = torch.cat([p.grad.view(-1).detach().clone() for p in model.parameters()])
    grad_norm = torch.norm(grad_vec).item()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(x_val)
        loss_val = criterion(pred_val, y_val).item()
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    trajectory.append((state_dict, loss_val, grad_norm, grad_vec))

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss_val = {loss_val:.4f}, grad_norm = {grad_norm:.4f}")

# ---------- 4. Estimation of constants ----------
losses = [traj[1] for traj in trajectory]
R_min = np.min(losses)
sigma2 = R_min
L0 = max([traj[2] for traj in trajectory])

# Flatten parameters for diameter computation
theta_vectors = []
for state_dict, _, _, _ in trajectory:
    vec = torch.cat([v.view(-1) for v in state_dict.values()])
    theta_vectors.append(vec)
theta_vectors = torch.stack(theta_vectors)

# q: diameter of the parameter space (last 50% epochs to avoid initial transient)
start_epoch = epochs // 2
theta_stable = theta_vectors[start_epoch:]
q = 0.0
for i in range(len(theta_stable)):
    for j in range(i+1, len(theta_stable)):
        dist = torch.norm(theta_stable[i] - theta_stable[j]).item()
        if dist > q:
            q = dist

# g: estimate of strong convexity via gradient differences
g_estimates = []
for t in range(1, len(trajectory)):
    grad_t = trajectory[t][3]
    grad_prev = trajectory[t-1][3]
    diff_grad = torch.norm(grad_t - grad_prev).item()
    theta_t = theta_vectors[t]
    theta_prev = theta_vectors[t-1]
    diff_theta = torch.norm(theta_t - theta_prev).item()
    if diff_theta > 1e-8:
        g_estimates.append(diff_grad / diff_theta)
g = np.min(g_estimates) if g_estimates else 0.001
g = max(g, 1e-5)

print(f"Estimated constants:")
print(f"  sigma2 = {sigma2:.6f}")
print(f"  L0     = {L0:.4f}")
print(f"  q      = {q:.4f}")
print(f"  g      = {g:.6f}")

# ---------- 5. Theoretical threshold gamma ----------
Pc = 0.95
r = 1.0
pi2_6 = np.pi**2 / 6
def c(n):
    return 4 * L0 * np.sqrt(sigma2) * q / np.sqrt(1 - Pc) * n**(-r/2) * np.sqrt(pi2_6)

gamma = c(n_val)
print(f"  gamma  = {gamma:.6f} (at n={n_val})")

# ---------- 6. Stopping rule based on prediction width ----------
desired_pred_error = 0.2   # target prediction accuracy (Scenario 1)
prediction_widths = []      # average width on validation set for each epoch

for n in range(1, len(trajectory)+1):
    sub_indices = [i for i in range(n) if losses[i] <= R_min + gamma]
    if len(sub_indices) < 2:
        prediction_widths.append(float('inf'))
        continue
    preds = []
    for idx in sub_indices:
        model.load_state_dict(trajectory[idx][0])
        model.eval()
        with torch.no_grad():
            pred = model(x_val).numpy().flatten()
            preds.append(pred)
    preds = np.array(preds)
    width_per_point = np.max(preds, axis=0) - np.min(preds, axis=0)
    avg_width = np.mean(width_per_point)
    prediction_widths.append(avg_width)

# Find stopping time tau
tau = None
for n, w in enumerate(prediction_widths, start=1):
    if w <= desired_pred_error:
        tau = n
        break
if tau is None:
    tau = len(trajectory)
    print(f"Stopping time not reached in {epochs} epochs. Using final model.")
else:
    print(f"Stopping time τ = {tau} (avg width = {prediction_widths[tau-1]:.6f} ≤ {desired_pred_error})")

final_idx = tau - 1
model.load_state_dict(trajectory[final_idx][0])

# ---------- 7. Conformal prediction ----------
model.eval()
with torch.no_grad():
    pred_val = model(x_val).numpy().flatten()
    y_val_np = y_val.numpy().flatten()
    residuals = np.abs(y_val_np - pred_val)

alpha = 1 - Pc
n_cal = len(residuals)
q_hat = np.quantile(residuals, (1 - alpha) * (1 + 1/n_cal), interpolation='higher')
print(f"Conformal quantile q_hat = {q_hat:.6f}")

with torch.no_grad():
    pred_test = model(x_test).numpy().flatten()
lower = pred_test - q_hat
upper = pred_test + q_hat

y_test_np = y_test.numpy().flatten()
coverage = np.mean((y_test_np >= lower) & (y_test_np <= upper))
print(f"Test coverage: {coverage:.3f} (expected ≥ {Pc:.2f})")

# ---------- 8. Save results to CSV ----------
csv_filename = "experiment_results.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "prediction_width"])
    for n, w in enumerate(prediction_widths, start=1):
        writer.writerow([n, w])
    writer.writerow([])
    writer.writerow(["metric", "value"])
    writer.writerow(["stopping_time_tau", tau])
    writer.writerow(["coverage_test", coverage])
    writer.writerow(["conformal_quantile_q_hat", q_hat])
    writer.writerow(["R_min", R_min])
    writer.writerow(["sigma2_est", sigma2])
    writer.writerow(["L0_est", L0])
    writer.writerow(["q_est", q])
    writer.writerow(["g_est", g])
    writer.writerow(["gamma", gamma])
print(f"Results saved to {csv_filename}")

# ---------- 9. Visualization ----------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(1, len(prediction_widths)+1), prediction_widths, label='Average prediction width')
plt.axhline(y=desired_pred_error, color='r', linestyle='--', label=f'Threshold δ = {desired_pred_error}')
if tau is not None:
    plt.axvline(x=tau, color='g', linestyle='--', label=f'Stopping time τ = {tau}')
plt.xlabel('Epoch n')
plt.ylabel('Average prediction width on validation set')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
x_test_np = x_test.numpy().flatten()
plt.plot(x_test_np, np.sin(2*np.pi*x_test_np), 'k--', label='True function')
plt.fill_between(x_test_np, lower, upper, alpha=0.3, color='blue', label='Conformal prediction interval (95%)')
plt.scatter(x_val.numpy().flatten(), y_val_np, s=10, alpha=0.5, label='Validation data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title(f'Coverage = {coverage:.3f}, τ = {tau}')
plt.tight_layout()
plt.savefig('experiment_plot.png', dpi=300, bbox_inches='tight')
plt.show()