import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

# --- 1. Define the Physics-Informed Neural ODE Framework ---
class DrugDeliveryNeuralODE(nn.Module):
    def __init__(self, nx, dx, params):
        super(DrugDeliveryNeuralODE, self).__init__()
        self.nx = nx
        self.dx = dx
        # Unpack model constants
        self.D, self.E, self.alpha, self.beta, self.gamma, self.ub, self.vb, self.kr, self.ua = params

    def forward(self, t, U):
        """
        U is a flat vector of length (3 * nx):
        U[:nx]       -> Unbound drug (u)
        U[nx:2*nx]   -> Bound drug (v)
        U[2*nx:]     -> Polymer stress (sigma)
        """
        u = U[:self.nx]
        v = U[self.nx:2*self.nx]
        sigma = U[2*self.nx:]
        
        # Initialize derivative vectors
        du_dt = torch.zeros_like(u)
        dv_dt = torch.zeros_like(v)
        
        # --- 1. Spatial Discretization (Finite Differences for Interior Points) ---
        # Central difference approximation for second derivatives: (U_{i+1} - 2*U_i + U_{i-1}) / dx^2
        u_xx = (u[2:] - 2*u[1:-1] + u[:-2]) / (self.dx ** 2)
        sigma_xx = (sigma[2:] - 2*sigma[1:-1] + sigma[:-2]) / (self.dx ** 2)
        
        # Reaction Rate Functions for interior grid points
        g_uv = u[1:-1] * (self.ub - u[1:-1]) - v[1:-1] * (self.vb - v[1:-1])
        f_uv = -g_uv
        
        # Assign rates to interior nodes
        du_dt[1:-1] = self.D * u_xx + self.E * sigma_xx + f_uv
        dv_dt[1:-1] = g_uv
        
        # --- 2. Enforce Boundary Conditions explicitly on Boundaries ---
        # Robin Boundary Conditions for u at left (0) and right (-1) edges
        u_x_left = (-self.kr / self.D) * (self.ua - u[0])
        u_x_right = (self.kr / self.D) * (self.ua - u[-1])
        
        # Assign boundary approximations using ghost-cell/one-sided formulas
        du_dt[0] = self.D * (2 * (u[1] - u[0] - self.dx * u_x_left) / (self.dx**2)) + (-u[0]*(self.ub-u[0]) + v[0]*(self.vb-v[0]))
        du_dt[-1] = self.D * (2 * (u[-2] - u[-1] + self.dx * u_x_right) / (self.dx**2)) + (-u[-1]*(self.ub-u[-1]) + v[-1]*(self.vb-v[-1]))
        
        # Boundary reactions for bound drug
        dv_dt[0] = u[0] * (self.ub - u[0]) - v[0] * (self.vb - v[0])
        dv_dt[-1] = u[-1] * (self.ub - u[-1]) - v[-1] * (self.vb - v[-1])
        
        # --- 3. Compute Stress Evolution (Coupled Equation) ---
        # Equation 7.1c: dsigma/dt = alpha*u - beta*sigma + gamma*du/dt
        dsigma_dt = self.alpha * u - self.beta * sigma + self.gamma * du_dt
        
        # Enforce Neumann boundary conditions for stress (slopes are 0 at boundaries)
        dsigma_dt[0] = 0.0
        dsigma_dt[-1] = 0.0
        
        # Re-pack into a flat 1D trajectory derivative vector
        return torch.cat([du_dt, dv_dt, dsigma_dt])

# --- 2. Simulation Execution Block ---
if __name__ == "__main__":
    print("Initializing Neural ODE System...")
    
    # Grid Configurations (Matching Chapter 7 Main Script)
    nx = 26
    xl, xu = -0.5, 0.5
    dx = (xu - xl) / (nx - 1)
    
    # Model Physical Constants
    # D, E, alpha, beta, gamma, ub, vb, kr, ua
    params = [0.6, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    
    # Set Initial Conditions (u0 = 0.75, v0 = 0.25, sigma0 = 0.0)
    u0 = torch.full((nx,), 0.75, dtype=torch.float64)
    v0 = torch.full((nx,), 0.25, dtype=torch.float64)
    sigma0 = torch.zeros((nx,), dtype=torch.float64)
    U0 = torch.cat([u0, v0, sigma0]) # Initial state vector of length 78
    
    # Setup Time Grid vector (From t=0 to t=2, reporting at 6 intervals)
    t_space = torch.linspace(0.0, 2.0, 6, dtype=torch.float64)
    
    # Instantiate the model
    neural_ode_model = DrugDeliveryNeuralODE(nx, dx, params)
    
    print("Integrating system trajectory using Runge-Kutta 4 (rk4)...")
    print("-" * 50)
    
    # Change the method to 'dopri5' and remove the rigid step_size option
    with torch.no_grad():
        solution = odeint(neural_ode_model, U0, t_space, method='dopri5')
         
    print("Integration Complete!")
    print(f"Output Trajectory Tensor Shape: {solution.shape} (Time steps, Flat state variables)")
    print("-" * 50)
    
    # --- 3. Extract and Display Final Results at Center Point (Index 12/13) ---
    final_time_step_index = -1
    center_spatial_index = 12 # Midpoint of 26 grid steps
    
    u_final = solution[final_time_step_index, center_spatial_index].item()
    v_final = solution[final_time_step_index, center_spatial_index + nx].item()
    s_final = solution[final_time_step_index, center_spatial_index + 2*nx].item()
    
    print(f"Neural ODE evaluation at Center (x=0.0) at Final Time (t=2.0):")
    print(f"Predicted Unbound Drug (u): {u_final:.4f}")
    print(f"Predicted Bound Drug (v):   {v_final:.4f}")
    print(f"Predicted Stress (sigma):   {s_final:.4f}")

import matplotlib.pyplot as plt

# Re-create spatial coordinate array for the x-axis
x_grid = np.linspace(xl, xu, nx)

plt.figure(figsize=(10, 6))

# Loop through the 6 time snapshots stored in the solution tensor
for idx, t_val in enumerate(t_space.numpy()):
    # Extract just the 'u' vector (first 26 elements) for this specific time step
    u_snapshot = solution[idx, :nx].numpy()
    plt.plot(x_grid, u_snapshot, label=f'Time t = {t_val:.1f}', linewidth=2)

plt.title('Neural ODE Solution: Unbound Drug Depletion Over Time', fontsize=12)
plt.xlabel('Position across Patch (x)', fontsize=10)
plt.ylabel('Unbound Drug Concentration (u)', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()