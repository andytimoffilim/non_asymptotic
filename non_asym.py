#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-asymptotic confidence intervals for neural network predictions (synthetic experiment)
Final working version: raw data, no smoothing, honest visualization.
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

# Split validation set: first half for stopping, second half for calibration
n_stop = n_val // 2
x_stop, y_stop = x_val[:n_stop], y_val[:n_stop]
x_cal, y_cal = x_val[n_stop:], y_val[n_stop:]

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
        pred_stop = model(x_stop)
        loss_stop = criterion(pred_stop, y_stop).item()
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    trajectory.append((state_dict, loss_stop, grad_norm, grad_vec))

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss_stop = {loss_stop:.4f}, grad_norm = {grad_norm:.4f}")

# ---------- 4. Estimation of constants ----------
losses = [traj[1] for traj in trajectory]
R_min = np.min(losses)
sigma2 = R_min

KAPPA = 1.2
L0_raw = max([traj[2] for traj in trajectory])
L0 = KAPPA * L0_raw

theta_vectors = []
for state_dict, _, _, _ in trajectory:
    vec = torch.cat([v.view(-1) for v in state_dict.values()])
    theta_vectors.append(vec)
theta_vectors = torch.stack(theta_vectors)

start_epoch = epochs // 2
theta_stable = theta_vectors[start_epoch:]
q = 0.0
for i in range(len(theta_stable)):
    for j in range(i+1, len(theta_stable)):
        dist = torch.norm(theta_stable[i] - theta_stable[j]).item()
        if dist > q:
            q = dist

print(f"Estimated constants:")
print(f"  sigma2 = {sigma2:.6f}")
print(f"  L0     = {L0:.4f} (raw = {L0_raw:.4f}, kappa = {KAPPA})")
print(f"  q      = {q:.4f}")

# ---------- 5. Theoretical threshold c(n) ----------
Pc = 0.95
r = 1.0
pi2_6 = np.pi**2 / 6

def c(n_eff):
    return 4 * L0 * np.sqrt(sigma2) * q / np.sqrt(1 - Pc) * n_eff**(-r/2) * np.sqrt(pi2_6)

# ---------- 6. Stopping rule based on prediction width ----------
desired_pred_error = 0.2
prediction_widths = []      # raw widths (no smoothing)
gammas = []
min_models = 10

for epoch in range(1, len(trajectory)+1):
    n_eff = epoch * n_train
    gamma_n = c(n_eff)
    gammas.append(gamma_n)
    current_loss = losses[epoch-1]
    sub_indices = [i for i in range(epoch) if losses[i] <= current_loss + gamma_n]
    if len(sub_indices) < min_models:
        prediction_widths.append(None)
        continue
    preds = []
    for idx in sub_indices:
        model.load_state_dict(trajectory[idx][0])
        model.eval()
        with torch.no_grad():
            pred = model(x_stop).numpy().flatten()
            preds.append(pred)
    preds = np.array(preds)
    width_per_point = np.max(preds, axis=0) - np.min(preds, axis=0)
    avg_width = np.mean(width_per_point)
    prediction_widths.append(avg_width)

# Find stopping time tau
tau = None
for n, w in enumerate(prediction_widths, start=1):
    if w is not None and w <= desired_pred_error:
        tau = n
        break
if tau is None:
    tau = len(trajectory)
    print(f"Stopping time not reached in {epochs} epochs. Using final model.")
else:
    print(f"Stopping time τ = {tau} (avg width = {prediction_widths[tau-1]:.6f} ≤ {desired_pred_error})")
    print(f"  gamma at τ = {gammas[tau-1]:.6f}")

final_idx = tau - 1
model.load_state_dict(trajectory[final_idx][0])

# ---------- 7. Conformal prediction (using calibration subset) ----------
model.eval()
with torch.no_grad():
    pred_cal = model(x_cal).numpy().flatten()
    y_cal_np = y_cal.numpy().flatten()
    residuals = np.abs(y_cal_np - pred_cal)

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
    writer.writerow(["epoch", "prediction_width", "gamma"])
    for n, (w, g) in enumerate(zip(prediction_widths, gammas), start=1):
        writer.writerow([n, w, g])
    writer.writerow([])
    writer.writerow(["metric", "value"])
    writer.writerow(["stopping_time_tau", tau])
    writer.writerow(["coverage_test", coverage])
    writer.writerow(["conformal_quantile_q_hat", q_hat])
    writer.writerow(["R_min", R_min])
    writer.writerow(["sigma2_est", sigma2])
    writer.writerow(["L0_est", L0])
    writer.writerow(["q_est", q])
    writer.writerow(["gamma_at_tau", gammas[tau-1] if tau else None])
print(f"Results saved to {csv_filename}")

# ---------- 9. Visualization ----------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
# Plot raw widths (only finite values)
valid_epochs = [i+1 for i, w in enumerate(prediction_widths) if w is not None]
valid_widths = [w for w in prediction_widths if w is not None]
plt.plot(valid_epochs, valid_widths, 'b-', linewidth=2, label='Prediction width (raw)')
plt.axhline(y=desired_pred_error, color='r', linestyle='--', label=f'Threshold δ = {desired_pred_error}')
if tau is not None:
    plt.axvline(x=tau, color='g', linestyle='--', label=f'Stopping time τ = {tau}')
plt.xlabel('Epoch n')
plt.ylabel('Average prediction width on stop set')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
x_test_np = x_test.numpy().flatten()
plt.plot(x_test_np, np.sin(2*np.pi*x_test_np), 'k--', label='True function')
plt.fill_between(x_test_np, lower, upper, alpha=0.3, color='blue', label='Conformal prediction interval (95%)')
plt.scatter(x_stop.numpy().flatten(), y_stop.numpy().flatten(), s=10, alpha=0.5, label='Stop data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title(f'Coverage = {coverage:.3f}, τ = {tau}')
plt.tight_layout()
plt.savefig('experiment_plot.png', dpi=300, bbox_inches='tight')
plt.show()