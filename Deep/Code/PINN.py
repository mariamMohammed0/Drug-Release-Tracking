import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Define Architecture ---
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.net(inputs)
        u = out[:, 0:1]
        v = out[:, 1:2]
        sigma = out[:, 2:3]
        return u, v, sigma

# --- 2. Corrected Loss Function ---
def compute_loss(model, x_col, t_col, t_bc, params):
    D, E, alpha, beta, gamma, ub, vb, kr, ua, xl, xu = params
    
    # --- 1. PDE Loss (Interior points) ---
    x_col.requires_grad_(True)
    t_col.requires_grad_(True)
    
    u, v, sigma = model(x_col, t_col)
    
    u_t = torch.autograd.grad(u, t_col, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_col, torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t_col, torch.ones_like(v), create_graph=True)[0]
    sigma_t = torch.autograd.grad(sigma, t_col, torch.ones_like(sigma), create_graph=True)[0]
    sigma_x = torch.autograd.grad(sigma, x_col, torch.ones_like(sigma), create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x_col, torch.ones_like(u_x), create_graph=True)[0]
    sigma_xx = torch.autograd.grad(sigma_x, x_col, torch.ones_like(sigma_x), create_graph=True)[0]
    
    g_uv = u * (ub - u) - v * (vb - v)
    f_uv = -g_uv
    
    res_u = u_t - (D * u_xx + E * sigma_xx + f_uv)
    res_v = v_t - g_uv
    res_sigma = sigma_t + beta * sigma - (alpha * u + gamma * u_t)
    
    loss_pde = torch.mean(res_u**2) + torch.mean(res_v**2) + torch.mean(res_sigma**2)
    
    # --- 2. Initial Conditions Loss (t = 0) ---
    t0 = torch.zeros_like(x_col)
    u_0, v_0, sigma_0 = model(x_col, t0)
    loss_ic = torch.mean((u_0 - 0.75)**2) + torch.mean((v_0 - 0.25)**2) + torch.mean(sigma_0**2)
    
    # --- 3. Boundary Conditions Loss (x = xl, x = xu) ---
    x_l = torch.full_like(t_bc, xl).requires_grad_(True)
    x_r = torch.full_like(t_bc, xu).requires_grad_(True)
    
    u_l, _, sigma_l = model(x_l, t_bc)
    u_r, _, sigma_r = model(x_r, t_bc)
    
    u_xl = torch.autograd.grad(u_l, x_l, torch.ones_like(u_l), create_graph=True)[0]
    u_xr = torch.autograd.grad(u_r, x_r, torch.ones_like(u_r), create_graph=True)[0]
    sigma_xl = torch.autograd.grad(sigma_l, x_l, torch.ones_like(sigma_l), create_graph=True)[0]
    sigma_xr = torch.autograd.grad(sigma_r, x_r, torch.ones_like(sigma_r), create_graph=True)[0]
    
    loss_bc = torch.mean((D * u_xl + kr * (ua - u_l))**2) + \
              torch.mean((D * u_xr - kr * (ua - u_r))**2) + \
              torch.mean(sigma_xl**2) + torch.mean(sigma_xr**2)
              
    return loss_pde + loss_ic + loss_bc

# --- 3. Main Training Execution ---
if __name__ == "__main__":
    # Hyperparameters & Constants
    xl, xu = -0.5, 0.5
    t0, tf = 0.0, 2.0
    params = [0.6, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, xl, xu] # D, E, alpha, beta, gamma, ub, vb, kr, ua, xl, xu
    
    # Seed for reproducibility
    torch.manual_seed(42)
    
    # Generate Training Collocation Points (Domain sampling)
    num_collocation = 2000
    num_boundary = 500
    
    # Uniformly sample space and time
    x_train = torch.distributions.Uniform(xl, xu).sample((num_collocation, 1))
    t_train = torch.distributions.Uniform(t0, tf).sample((num_collocation, 1))
    t_boundary = torch.distributions.Uniform(t0, tf).sample((num_boundary, 1))
    
    # Initialize PINN and Optimizer
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting PINN Optimization Loop...")
    print("-" * 40)
    
    # Optimization Loop
    epochs = 3000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Compute joint residual loss
        loss = compute_loss(model, x_train, t_train, t_boundary, params)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.6f}")
            
    print("-" * 40)
    print("Training Complete!")
    
    # Inference Check: Evaluate at a specific test coordinate (center point at final time)
    test_x = torch.tensor([[0.0]])
    test_t = torch.tensor([[2.0]])
    u_pred, v_pred, s_pred = model(test_x, test_t)
    print(f"\nInference evaluation at x=0, t=2:")
    print(f"Predicted Unbound Drug (u): {u_pred.item():.4f}")
    print(f"Predicted Bound Drug (v):   {v_pred.item():.4f}")
    print(f"Predicted Stress (sigma):   {s_pred.item():.4f}")

import matplotlib.pyplot as plt
import torch
import numpy as np

# Time snapshots to evaluate (matching the Neural ODE t_space)
t_snapshots = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]

# Spatial grid
x_grid = torch.linspace(xl, xu, 100).view(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('PINN Solution: Drug Patch Dynamics Over Time', fontsize=13)

u_all, v_all, s_all = [], [], []

for t_val in t_snapshots:
    t_grid = torch.full_like(x_grid, t_val)
    with torch.no_grad():
        u_pred, v_pred, s_pred = model(x_grid, t_grid)
    u_all.append(u_pred.numpy())
    v_all.append(v_pred.numpy())
    s_all.append(s_pred.numpy())

x_np = x_grid.numpy()

# --- Unbound drug u ---
ax = axes[0]
for i, t_val in enumerate(t_snapshots):
    ax.plot(x_np, u_all[i], label=f't = {t_val:.1f}', linewidth=2)
ax.set_title('Free drug u(x,t)', fontsize=11)
ax.set_xlabel('Position x')
ax.set_ylabel('Concentration')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=8)

# --- Bound drug v ---
ax = axes[1]
for i, t_val in enumerate(t_snapshots):
    ax.plot(x_np, v_all[i], label=f't = {t_val:.1f}', linewidth=2)
ax.set_title('Bound drug v(x,t)', fontsize=11)
ax.set_xlabel('Position x')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=8)

# --- Polymer stress sigma ---
ax = axes[2]
for i, t_val in enumerate(t_snapshots):
    ax.plot(x_np, s_all[i], label=f't = {t_val:.1f}', linewidth=2)
ax.set_title('Polymer stress σ(x,t)', fontsize=11)
ax.set_xlabel('Position x')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()