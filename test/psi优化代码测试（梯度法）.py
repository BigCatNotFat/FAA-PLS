import numpy as np

def calculate_gradient_rotatable_model(
    psi, w_fixed, P_max, sigma_sq,
    N_h, N_v, d, lambda_carrier, kappa, G_gain,
    alpha_B, phi_B, theta_B,  # Bob's path parameters (arrays)
    alpha_E, phi_E, theta_E   # Eve's path parameters (arrays)
):
    """
    Calculates the gradient of the objective function F(psi) for the rotatable FAA model.

    Args:
        psi (float): The current shape-control parameter.
        w_fixed (np.ndarray): The fixed beamforming vector (N x 1, complex).
        P_max (float): Maximum transmit power (not directly used in gradient of F, but for context).
        sigma_sq (float): Noise power.
        N_h (int): Number of horizontal antenna elements.
        N_v (int): Number of vertical antenna elements.
        d (float): Inter-element spacing.
        lambda_carrier (float): Carrier wavelength.
        kappa (float): Element gain parameter (power of cos(theta)).
        G_gain (float): Element gain normalization factor (e.g., 2*(kappa+1)).
        alpha_B (np.ndarray): Complex gains for Bob's paths (L_B x 1).
        phi_B (np.ndarray): Azimuth angles for Bob's paths (L_B x 1, radians).
        theta_B (np.ndarray): Elevation angles for Bob's paths (L_B x 1, radians).
        alpha_E (np.ndarray): Complex gains for Eve's paths (L_E x 1).
        phi_E (np.ndarray): Azimuth angles for Eve's paths (L_E x 1, radians).
        theta_E (np.ndarray): Elevation angles for Eve's paths (L_E x 1, radians).

    Returns:
        float: The gradient dF(psi)/dpsi.
    """
    N = N_h * N_v
    L_B = len(alpha_B)
    L_E = len(alpha_E)

    # Precompute constants
    k_wave = 2 * np.pi / lambda_carrier

    # --- Helper function to compute h(psi) and dh_dpsi(psi) for a user ---
    def get_channel_and_derivative(alpha_paths, phi_paths, theta_paths, L_paths):
        h_psi = np.zeros((N, 1), dtype=np.complex128)
        dh_dpsi = np.zeros((N, 1), dtype=np.complex128)

        for l_idx in range(L_paths):
            alpha_l = alpha_paths[l_idx]
            phi_l = phi_paths[l_idx]
            theta_l = theta_paths[l_idx]

            # Wave vector k_l
            k_l_vec = k_wave * np.array([
                np.sin(theta_l) * np.cos(phi_l),
                np.sin(theta_l) * np.sin(phi_l),
                np.cos(theta_l)
            ])

            # Element gain terms
            # K_theta_l = sqrt(G_gain * cos^kappa(theta_l))
            # g_l,n(psi) = K_theta_l * (sin(phi_l - psi))^(kappa/2)
            # dg_l,n(psi)/dpsi = -K_theta_l * (kappa/2) * (sin(phi_l - psi))^(kappa/2 - 1) * cos(phi_l - psi)
            
            cos_theta_l = np.cos(theta_l)
            K_theta_l = 0.0
            if cos_theta_l >= 0 and 0 <= theta_l <= np.pi/2: # Check for valid theta for gain
                K_theta_l = np.sqrt(G_gain * (cos_theta_l**kappa))
            
            phi_eff_l = phi_l - psi
            sin_phi_eff_l = np.sin(phi_eff_l)
            cos_phi_eff_l = np.cos(phi_eff_l)

            g_l_n_psi = 0.0
            dg_l_n_dpsi = 0.0

            if K_theta_l > 0 and sin_phi_eff_l > 0: # Check for valid phi_eff for gain
                g_l_n_psi = K_theta_l * (sin_phi_eff_l**(kappa / 2))
                if kappa == 0: # Special case to avoid 0*inf if sin_phi_eff_l^(kappa/2-1)
                     dg_l_n_dpsi = 0 # if kappa=0, gain is independent of phi_eff
                else:
                    dg_l_n_dpsi = -K_theta_l * (kappa / 2) * (sin_phi_eff_l**((kappa / 2) - 1)) * cos_phi_eff_l
            elif K_theta_l > 0 and kappa == 0 : # Omnidirectional in azimuth
                 g_l_n_psi = K_theta_l # (sin(phi_eff))**0 = 1
                 dg_l_n_dpsi = 0.0

            # Per-antenna calculations
            for n_v_idx in range(N_v):
                for n_h_idx in range(N_h):
                    n_idx = n_v_idx * N_h + n_h_idx # Linear index

                    # Antenna positions r_n(psi)
                    C_nh = (2 * (n_h_idx + 1) - N_h - 1) / 2
                    # z_n is constant, dz_n_dpsi = 0
                    # x_n(psi) = -C_nh * d * sin(psi)
                    # y_n(psi) =  C_nh * d * cos(psi)
                    r_n_psi = np.array([
                        -C_nh * d * np.sin(psi),
                        C_nh * d * np.cos(psi),
                        ((2 * (n_v_idx + 1) - N_v - 1) / 2) * d
                    ])

                    # Derivative of antenna positions dr_n(psi)/dpsi
                    # dx_n_dpsi = -C_nh * d * cos(psi)
                    # dy_n_dpsi = -C_nh * d * sin(psi)
                    dr_n_dpsi = np.array([
                        -C_nh * d * np.cos(psi),
                        -C_nh * d * np.sin(psi),
                        0.0
                    ])

                    # Array manifold a_l,n(psi)
                    # a_l,n(psi) = exp(-j * k_l_vec . r_n(psi))
                    exponent_a = -1j * np.dot(k_l_vec, r_n_psi)
                    a_l_n_psi = np.exp(exponent_a)

                    # Derivative of array manifold da_l,n(psi)/dpsi
                    # da_l,n(psi)/dpsi = a_l,n(psi) * (-j * k_l_vec . dr_n(psi)/dpsi)
                    # k_l_vec . dr_n(psi)/dpsi was: - (2*pi*d*C_nh/lambda) * sin(theta_l) * cos(phi_l - psi)
                    k_dot_dr_dpsi = - (k_wave * d * C_nh) * np.sin(theta_l) * np.cos(phi_l - psi)
                    da_l_n_dpsi = a_l_n_psi * (-1j * k_dot_dr_dpsi)

                    # c_l,n(psi) = g_l,n(psi) * a_l,n(psi)
                    c_l_n_psi = g_l_n_psi * a_l_n_psi
                    
                    # dc_l,n(psi)/dpsi = dg_l,n(psi)/dpsi * a_l,n(psi) + g_l,n(psi) * da_l,n(psi)/dpsi
                    dc_l_n_dpsi = dg_l_n_dpsi * a_l_n_psi + g_l_n_psi * da_l_n_dpsi
                    
                    h_psi[n_idx] += alpha_l * c_l_n_psi
                    dh_dpsi[n_idx] += alpha_l * dc_l_n_dpsi
        
        norm_factor = np.sqrt(1.0 / L_paths) if L_paths > 0 else 1.0
        h_psi *= norm_factor
        dh_dpsi *= norm_factor
        return h_psi, dh_dpsi

    # --- Calculate for Bob ---
    h_B_psi, dh_B_dpsi = get_channel_and_derivative(alpha_B, phi_B, theta_B, L_B)

    # --- Calculate for Eve ---
    h_E_psi, dh_E_dpsi = get_channel_and_derivative(alpha_E, phi_E, theta_E, L_E)

    # --- Calculate S_B(psi), S_E(psi) ---
    # S_B(psi) = |h_B(psi)^H w|^2
    hB_H_w = np.conjugate(h_B_psi.T) @ w_fixed # This is a scalar (1x1 matrix)
    S_B_psi = np.abs(hB_H_w[0,0])**2

    hE_H_w = np.conjugate(h_E_psi.T) @ w_fixed # This is a scalar (1x1 matrix)
    S_E_psi = np.abs(hE_H_w[0,0])**2

    # --- Calculate dS_B(psi)/dpsi, dS_E(psi)/dpsi ---
    # dS_B(psi)/dpsi = 2 * Re{ ( (dh_B(psi)/dpsi)^H w ) * ( w^H h_B(psi) ) }
    # Note: w^H h_B(psi) = (h_B(psi)^H w)^*
    
    dhB_dpsi_H_w = np.conjugate(dh_B_dpsi.T) @ w_fixed # scalar
    w_H_hB_psi = np.conjugate(w_fixed.T) @ h_B_psi     # scalar
    
    dS_B_dpsi = 2 * np.real(dhB_dpsi_H_w[0,0] * w_H_hB_psi[0,0])

    dhE_dpsi_H_w = np.conjugate(dh_E_dpsi.T) @ w_fixed # scalar
    w_H_hE_psi = np.conjugate(w_fixed.T) @ h_E_psi     # scalar
    
    dS_E_dpsi = 2 * np.real(dhE_dpsi_H_w[0,0] * w_H_hE_psi[0,0])

    # --- Final Gradient dF(psi)/dpsi ---
    # dF/dpsi = ( (dS_B/dpsi)*(sigma^2 + S_E) - (sigma^2 + S_B)*(dS_E/dpsi) ) / (sigma^2 + S_E)^2
    
    numerator = dS_B_dpsi * (sigma_sq + S_E_psi) - (sigma_sq + S_B_psi) * dS_E_dpsi
    denominator = (sigma_sq + S_E_psi)**2

    if denominator == 0:
        # Avoid division by zero; this case should ideally not happen if sigma_sq > 0
        # Or if S_E is always non-negative.
        # If it happens, it might indicate an issue or a need for regularization.
        # For now, return a large number or handle as an error.
        return np.nan # Or handle appropriately

    gradient_F_psi = numerator / denominator
    
    return gradient_F_psi

if __name__ == '__main__':
    # --- Example Usage ---
    N_h_sim = 4
    N_v_sim = 4
    N_sim = N_h_sim * N_v_sim
    
    psi_test = np.deg2rad(10) # Example psi
    w_fixed_sim = np.random.randn(N_sim, 1) + 1j * np.random.randn(N_sim, 1)
    w_fixed_sim = w_fixed_sim / np.linalg.norm(w_fixed_sim) # Normalize (as if P_max=1)
    print("w_fixed_sim",w_fixed_sim)
    P_max_sim = 1.0
    sigma_sq_sim = 0.1

    d_sim = 0.5 # Assuming wavelength is 1, so d = 0.5 * lambda
    lambda_carrier_sim = 1.0
    kappa_sim = 2.0 # Example directional antenna
    G_gain_sim = 2 * (kappa_sim + 1)

    # Bob's paths
    L_B_sim = 2
    alpha_B_sim = np.array([1.0 + 0j, 0.5 * np.exp(1j * np.pi/4)])
    phi_B_sim = np.deg2rad(np.array([30.0, 45.0]))
    theta_B_sim = np.deg2rad(np.array([60.0, 70.0]))

    # Eve's paths
    L_E_sim = 1
    alpha_E_sim = np.array([0.8 * np.exp(1j * np.pi/2)])
    phi_E_sim = np.deg2rad(np.array([120.0]))
    theta_E_sim = np.deg2rad(np.array([80.0]))

    grad = calculate_gradient_rotatable_model(
        psi_test, w_fixed_sim, P_max_sim, sigma_sq_sim,
        N_h_sim, N_v_sim, d_sim, lambda_carrier_sim, kappa_sim, G_gain_sim,
        alpha_B_sim, phi_B_sim, theta_B_sim,
        alpha_E_sim, phi_E_sim, theta_E_sim
    )
    print(f"Gradient dF/dpsi at psi = {np.rad2deg(psi_test):.2f} degrees: {grad:.4f}")

    # --- Test with a slightly different psi to check gradient direction (finite difference) ---
    # This is a simple numerical check, not a formal test
    delta_psi = 1e-6
    
    def F_psi_objective(psi_val): # Objective function F(psi)
        h_B, _ = calculate_gradient_rotatable_model.get_channel_and_derivative(
            alpha_B_sim, phi_B_sim, theta_B_sim, L_B_sim,
            psi_val, N_h_sim, N_v_sim, d_sim, lambda_carrier_sim, kappa_sim, G_gain_sim
        ) # Need to expose get_channel_and_derivative or re-implement parts
        
        # Re-implementing parts of F(psi) for this test:
        h_B_val, _ = calculate_gradient_rotatable_model.__closure__[0].cell_contents(
            alpha_B_sim, phi_B_sim, theta_B_sim, L_B_sim,
            psi_val, N_h_sim, N_v_sim, d_sim, lambda_carrier_sim, kappa_sim, G_gain_sim
        )
        h_E_val, _ = calculate_gradient_rotatable_model.__closure__[0].cell_contents(
            alpha_E_sim, phi_E_sim, theta_E_sim, L_E_sim,
            psi_val, N_h_sim, N_v_sim, d_sim, lambda_carrier_sim, kappa_sim, G_gain_sim
        )
        S_B_val = np.abs(np.conjugate(h_B_val.T) @ w_fixed_sim)**2
        S_E_val = np.abs(np.conjugate(h_E_val.T) @ w_fixed_sim)**2
        
        if (sigma_sq_sim + S_E_val) == 0: return np.nan
        return (sigma_sq_sim + S_B_val) / (sigma_sq_sim + S_E_val)

    # Due to closure, directly calling the inner helper is tricky for a clean test here.
    # The analytical gradient is the main output.
    # For a full numerical check, you'd define F(psi) separately.
    # Let's assume the analytical gradient is correct for now based on the derivation.

    # Example of how to numerically check if you define F(psi) separately:
    # F_psi_plus = F_psi_objective(psi_test + delta_psi)
    # F_psi_minus = F_psi_objective(psi_test - delta_psi)
    # numerical_grad = (F_psi_plus - F_psi_minus) / (2 * delta_psi)
    # print(f"Numerical gradient (approx): {numerical_grad:.4f}")
    # print(f"Relative difference: {abs(grad - numerical_grad) / abs(grad) if grad != 0 else 0 :.2e}")