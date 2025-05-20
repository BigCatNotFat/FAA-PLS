# File: verify_derivatives.py

import numpy as np
import math
import sys
import os

# Adjust the path to import your modules if they are in a different directory structure
# Example: if your main script and 'models', 'config', 'optimizers', 'logs' are in a parent directory 'my_project'
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, os.pardir)) # Or specific path
# sys.path.append(project_root)

from config.config import Config
from models.faa_rot_model import faa_rot_model
from optimizers.opt import Optimizer # Assuming your Optimizer class is in optimizers.opt
# No logger needed for this verification script, or you can initialize it if you want debug prints from modules

def verify_dh_dpsi():
    print("\n--- Verifying get_channel_and_derivative (dh_dpsi) ---")
    config = Config()
    model = faa_rot_model(config)

    # Generate original antenna positions (copied from Faa_pls for self-containment)
    antenna_positions_original = []
    nh_indices = np.arange(1, config.N_h + 1)
    nv_indices = np.arange(1, config.N_v + 1)
    y_coords = ((2 * nh_indices - config.N_h - 1) / 2) * config.d
    z_coords = ((2 * nv_indices - config.N_v - 1) / 2) * config.d
    y_grid, z_grid = np.meshgrid(y_coords, z_coords)
    x_grid = np.zeros_like(y_grid)
    antenna_positions_original = np.column_stack((
        x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
    ))

    psi_degrees = 30.0  # Angle in degrees
    h_diff_rad = 1e-7   # Small perturbation for finite difference in RADIANS
                        # Convert to degrees if your model.transform_antenna_positions expects degrees for perturbation
    h_diff_deg = math.degrees(h_diff_rad)


    # --- Analytical Derivative ---
    # Positions at psi
    ant_pos_rot_psi = model.transform_antenna_positions(antenna_positions_original, psi_degrees)
    h_B_psi, dh_B_dpsi_analytical = model.get_channel_and_derivative(
        ant_pos_rot_psi, config.beta_paths_bob, config.theta_paths_bob, config.phi_paths_bob, psi_degrees
    )
    h_E_psi, dh_E_dpsi_analytical = model.get_channel_and_derivative(
        ant_pos_rot_psi, config.beta_paths_eve, config.theta_paths_eve, config.phi_paths_eve, psi_degrees
    )

    # --- Numerical Derivative for Bob ---
    # h_B(psi) is already h_B_psi
    # Calculate h_B(psi + h_diff_rad)
    psi_plus_h_deg = psi_degrees + h_diff_deg
    ant_pos_rot_psi_plus_h = model.transform_antenna_positions(antenna_positions_original, psi_plus_h_deg)
    
    h_B_psi_plus_h, _ = model.get_channel_and_derivative(
        ant_pos_rot_psi_plus_h, config.beta_paths_bob, config.theta_paths_bob, config.phi_paths_bob, psi_plus_h_deg
    )
    dh_B_dpsi_numerical = (h_B_psi_plus_h - h_B_psi) / h_diff_rad

    # --- Numerical Derivative for Eve ---
    # h_E(psi) is already h_E_psi
    # Calculate h_E(psi + h_diff_rad)
    h_E_psi_plus_h, _ = model.get_channel_and_derivative(
        ant_pos_rot_psi_plus_h, config.beta_paths_eve, config.theta_paths_eve, config.phi_paths_eve, psi_plus_h_deg
    )
    dh_E_dpsi_numerical = (h_E_psi_plus_h - h_E_psi) / h_diff_rad
    

    print(f"Psi (degrees): {psi_degrees}")
    print(f"h_diff (radians): {h_diff_rad}")

    print("\nFor Bob's channel (dh_B/dpsi):")
    # print("Analytical:\n", dh_B_dpsi_analytical.flatten())
    # print("Numerical:\n", dh_B_dpsi_numerical.flatten())
    abs_error_bob = np.abs(dh_B_dpsi_analytical - dh_B_dpsi_numerical)
    rel_error_bob = np.abs(dh_B_dpsi_analytical - dh_B_dpsi_numerical) / (np.abs(dh_B_dpsi_analytical) + 1e-12) # Add epsilon to avoid div by zero
    print(f"Max absolute error for dh_B/dpsi: {np.max(abs_error_bob)}")
    print(f"Mean absolute error for dh_B/dpsi: {np.mean(abs_error_bob)}")
    print(f"Max relative error for dh_B/dpsi: {np.max(rel_error_bob)}")
    print(f"Mean relative error for dh_B/dpsi: {np.mean(rel_error_bob)}")


    print("\nFor Eve's channel (dh_E/dpsi):")
    # print("Analytical:\n", dh_E_dpsi_analytical.flatten())
    # print("Numerical:\n", dh_E_dpsi_numerical.flatten())
    abs_error_eve = np.abs(dh_E_dpsi_analytical - dh_E_dpsi_numerical)
    rel_error_eve = np.abs(dh_E_dpsi_analytical - dh_E_dpsi_numerical) / (np.abs(dh_E_dpsi_analytical) + 1e-12)
    print(f"Max absolute error for dh_E/dpsi: {np.max(abs_error_eve)}")
    print(f"Mean absolute error for dh_E/dpsi: {np.mean(abs_error_eve)}")
    print(f"Max relative error for dh_E/dpsi: {np.max(rel_error_eve)}")
    print(f"Mean relative error for dh_E/dpsi: {np.mean(rel_error_eve)}")

    if np.allclose(dh_B_dpsi_analytical, dh_B_dpsi_numerical, atol=1e-5, rtol=1e-3) and \
       np.allclose(dh_E_dpsi_analytical, dh_E_dpsi_numerical, atol=1e-5, rtol=1e-3):
        print("\nSUCCESS: dh/dpsi analytical and numerical results are close.")
    else:
        print("\nWARNING: dh/dpsi analytical and numerical results differ significantly. Check tolerances or h_diff.")
    return dh_B_dpsi_analytical, dh_E_dpsi_analytical # Return these for the next verification

def verify_total_gradient_F_psi(dh_B_dpsi_analytical_at_psi, dh_E_dpsi_analytical_at_psi):
    print("\n--- Verifying model.psi_derivative (gradient of F(psi)) ---")
    config = Config()
    model = faa_rot_model(config)
    optimizer = Optimizer()

    antenna_positions_original = []
    nh_indices = np.arange(1, config.N_h + 1)
    nv_indices = np.arange(1, config.N_v + 1)
    y_coords = ((2 * nh_indices - config.N_h - 1) / 2) * config.d
    z_coords = ((2 * nv_indices - config.N_v - 1) / 2) * config.d
    y_grid, z_grid = np.meshgrid(y_coords, z_coords)
    x_grid = np.zeros_like(y_grid)
    antenna_positions_original = np.column_stack((
        x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
    ))

    psi_degrees = 30.0
    h_diff_rad = 1e-7  # Small perturbation in RADIANS
    h_diff_deg = math.degrees(h_diff_rad)

    # Helper function to calculate F(psi) = (sigma^2 + S_B(psi)) / (sigma^2 + S_E(psi))
    # w_fixed is crucial here - it's calculated based on h_B(psi_degrees) and h_E(psi_degrees)
    # and then *kept fixed* for F(psi_degrees) and F(psi_degrees + h_diff)
    def calculate_F(psi_val_deg, w_beamformer):
        # Antenna positions at psi_val_deg
        current_ant_pos_rot = model.transform_antenna_positions(antenna_positions_original, psi_val_deg)
        
        # Channel at psi_val_deg
        h_B_current, _ = model.get_channel_and_derivative(
            current_ant_pos_rot, config.beta_paths_bob, config.theta_paths_bob, config.phi_paths_bob, psi_val_deg
        )
        h_E_current, _ = model.get_channel_and_derivative(
            current_ant_pos_rot, config.beta_paths_eve, config.theta_paths_eve, config.phi_paths_eve, psi_val_deg
        )
        
        # S_B and S_E at psi_val_deg with the FIXED w_beamformer
        S_B_val = np.abs(np.conjugate(h_B_current.T) @ w_beamformer)**2
        S_E_val = np.abs(np.conjugate(h_E_current.T) @ w_beamformer)**2
        
        numerator = config.sigma_sq + S_B_val
        denominator = config.sigma_sq + S_E_val
        if denominator == 0:
            return np.nan # Should not happen if sigma_sq > 0
        return numerator / denominator

    # 1. Calculate h_B(psi), h_E(psi) at the chosen psi_degrees
    ant_pos_rot_at_psi = model.transform_antenna_positions(antenna_positions_original, psi_degrees)
    h_B_at_psi, _ = model.get_channel_and_derivative( # We already have dh_B_dpsi_analytical_at_psi
        ant_pos_rot_at_psi, config.beta_paths_bob, config.theta_paths_bob, config.phi_paths_bob, psi_degrees
    )
    h_E_at_psi, _ = model.get_channel_and_derivative( # We already have dh_E_dpsi_analytical_at_psi
        ant_pos_rot_at_psi, config.beta_paths_eve, config.theta_paths_eve, config.phi_paths_eve, psi_degrees
    )

    # 2. Calculate the fixed beamformer w_fixed using these channels
    w_fixed = optimizer.optimize_beamformer_fixed_psi(
        h_B_at_psi, h_E_at_psi, config.P_max, config.sigma_sq
    )
    w_fixed = np.array(w_fixed).reshape(-1, 1) # Ensure it's a column vector

    # --- Analytical Gradient dF/dpsi ---
    # Note: model.psi_derivative expects dh_B_dpsi and dh_E_dpsi calculated at psi_degrees
    grad_F_psi_analytical = model.psi_derivative(
        h_B_at_psi, h_E_at_psi,
        dh_B_dpsi_analytical_at_psi, dh_E_dpsi_analytical_at_psi, # From previous verification
        w_fixed
    )

    # --- Numerical Gradient dF/dpsi ---
    # F(psi)
    F_at_psi = calculate_F(psi_degrees, w_fixed)
    
    # F(psi + h_diff)
    psi_plus_h_deg = psi_degrees + h_diff_deg
    F_at_psi_plus_h = calculate_F(psi_plus_h_deg, w_fixed)
    
    grad_F_psi_numerical = (F_at_psi_plus_h - F_at_psi) / h_diff_rad


    print(f"\nPsi (degrees): {psi_degrees}")
    print(f"h_diff (radians): {h_diff_rad}")
    print(f"Analytical dF/dpsi: {grad_F_psi_analytical.item()}") # .item() if it's a 1x1 array
    print(f"Numerical dF/dpsi:  {grad_F_psi_numerical.item()}")

    abs_error = np.abs(grad_F_psi_analytical - grad_F_psi_numerical)
    rel_error = abs_error / (np.abs(grad_F_psi_analytical) + 1e-12) # Add epsilon for stability
    print(f"Absolute error for dF/dpsi: {abs_error.item()}")
    print(f"Relative error for dF/dpsi: {rel_error.item() if grad_F_psi_analytical !=0 else 'N/A (analytical is zero)'}")

    if np.allclose(grad_F_psi_analytical, grad_F_psi_numerical, atol=1e-5, rtol=1e-3): # Adjust tolerances as needed
        print("\nSUCCESS: dF/dpsi analytical and numerical results are close.")
    else:
        print("\nWARNING: dF/dpsi analytical and numerical results differ significantly. Check tolerances or h_diff.")


if __name__ == "__main__":
    print("开始验证梯度")
    # Make sure random seeds are the same as in your main execution if you want identical channels
    # The Config class already sets np.random.seed.

    # First, verify the channel derivatives, as they are input to the F(psi) gradient
    dh_B_dpsi_analyt, dh_E_dpsi_analyt = verify_dh_dpsi()

    # Then, verify the gradient of F(psi) using the analytically computed channel derivatives
    verify_total_gradient_F_psi(dh_B_dpsi_analyt, dh_E_dpsi_analyt)

    print("\nVerification finished.")
    print("If 'SUCCESS' messages are shown, your analytical derivatives are likely correct.")
    print("If 'WARNING' messages persist, check:")
    print("  1. The value of h_diff (too large or too small).")
    print("  2. Consistency of units (degrees vs. radians) in calculations.")
    print("  3. The mathematical derivation of the analytical gradient.")
    print("  4. Ensure w_fixed is truly fixed when evaluating F(psi) and F(psi+h_diff).")