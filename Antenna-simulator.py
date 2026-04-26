import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --- Physical Constants ---
C = 299792458.0
MU_0 = 4 * np.pi * 1e-7
EPS_0 = 8.854187817e-12

def calculate_cavity_model(freq, L, W, h, er, nx, ny):
    """Analytical Cavity Model for comparison."""
    k0 = 2 * np.pi * freq / C
    
    # Effective parameters (Hammerstad & Jensen)
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / W)**-0.5
    delta_L = 0.412 * h * ((eeff + 0.3) * (W / h + 0.264)) / ((eeff - 0.258) * (W / h + 0.8))
    L_eff = L + 2 * delta_L
    
    # Create the same coordinate grid as FDTD FFT (u, v space)
    u = np.linspace(-1, 1, nx)
    v = np.linspace(-1, 1, ny)
    U, V = np.meshgrid(v, u) # V maps to X, U maps to Y in our FDTD setup
    
    # Mask visible space
    visible = U**2 + V**2 <= 1
    THETA = np.arcsin(np.sqrt(np.clip(U**2 + V**2, 0, 1)))
    PHI = np.arctan2(V, U + 1e-10)
    
    # Cavity Model Equations
    X = k0 * h / 2 * np.sin(THETA) * np.cos(PHI) + 1e-10
    f_theta = (np.sin(X) / X) * np.cos(k0 * L_eff / 2 * np.sin(THETA) * np.cos(PHI)) * np.cos(PHI)
    f_phi = (np.sin(X) / X) * np.cos(k0 * L_eff / 2 * np.sin(THETA) * np.cos(PHI)) * np.cos(THETA) * np.sin(PHI)
    
    E_total = np.sqrt(np.abs(f_theta)**2 + np.abs(f_phi)**2)
    E_total = np.where(visible, E_total, 0)
    
    return E_total, U, V

def calculate_directivity_dBi(Pattern_Linear, U, V):
    """Integrates the linear pattern (E-field) over the hemisphere to find absolute Gain (Directivity)."""
    # Radiation intensity U_rad is proportional to |E|^2
    U_rad = np.abs(Pattern_Linear)**2
    
    # Area element in u,v space: du * dv / cos(theta)
    du = U[0,1] - U[0,0]
    dv = V[1,0] - V[0,0]
    
    cos_theta = np.sqrt(np.clip(1 - U**2 - V**2, 1e-5, 1))
    visible = U**2 + V**2 <= 1
    
    # Total radiated power (integrating over the visible hemisphere in u,v coordinates)
    P_rad = np.sum(U_rad[visible] / cos_theta[visible]) * du * dv
    
    # Maximum intensity
    U_max = np.max(U_rad)
    
    # Directivity = 4 * pi * U_max / (Total Power) 
    # Since we only integrate over a hemisphere, we multiply by 2*pi for the hemisphere power equivalent
    D = (2 * np.pi * U_max) / P_rad
    
    return 10 * np.log10(D)

def run_fdtd(f_target, L_mm, W_mm, h_mm, er, padding, steps):
    """Core FDTD engine returning actual linear far-field pattern."""
    lambda_min = C / (f_target * np.sqrt(er))
    dx = dy = dz = min(1e-3, lambda_min / 12) 
    
    L_cells = int(np.ceil((L_mm * 1e-3) / dx))
    W_cells = int(np.ceil((W_mm * 1e-3) / dy))
    h_cells = int(np.ceil((h_mm * 1e-3) / dz))
    
    nx = L_cells + 2 * padding
    ny = W_cells + 2 * padding
    nz = h_cells + 15 
    
    if nx * ny * nz > 3e6:
        st.error("Grid too large. Reduce dimensions or frequency.")
        return None, None, None, None, None
        
    dt = 1.0 / (C * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) * 0.99
    
    eps = np.ones((nx, ny, nz)) * EPS_0
    mu = np.ones((nx, ny, nz)) * MU_0
    z_ground = 5
    z_patch = z_ground + h_cells
    eps[:, :, z_ground:z_patch] = EPS_0 * er
    
    C_e = dt / eps
    C_h = dt / mu
    
    Ex, Ey, Ez = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
    Hx, Hy, Hz = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
    Ez_phasor = np.zeros((nx, ny), dtype=complex)
    
    px_start, px_end = padding, padding + L_cells
    py_start, py_end = padding, padding + W_cells
    feed_x, feed_y = px_start + L_cells // 4, py_start + W_cells // 2
    
    progress_bar = st.progress(0)
    
    for t in range(steps):
        Hx[:, :-1, :-1] -= C_h[:, :-1, :-1] / dy * (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) - C_h[:, :-1, :-1] / dz * (Ey[:, :-1, 1:] - Ey[:, :-1, :-1])
        Hy[:-1, :, :-1] += C_h[:-1, :, :-1] / dx * (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) - C_h[:-1, :, :-1] / dz * (Ex[:-1, :, 1:] - Ex[:-1, :, :-1])
        Hz[:-1, :-1, :] -= C_h[:-1, :-1, :] / dx * (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) - C_h[:-1, :-1, :] / dy * (Ex[:-1, 1:, :] - Ex[:-1, :-1, :])

        Ex[:, 1:, 1:] += C_e[:, 1:, 1:] / dy * (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) - C_e[:, 1:, 1:] / dz * (Hy[:, 1:, 1:] - Hy[:, 1:, :-1])
        Ey[1:, :, 1:] -= C_e[1:, :, 1:] / dx * (Hz[1:, :, 1:] - Hz[:-1, :, 1:]) - C_e[1:, :, 1:] / dz * (Hx[1:, :, 1:] - Hx[1:, :, :-1])
        Ez[1:, 1:, :] += C_e[1:, 1:, :] / dx * (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) - C_e[1:, 1:, :] / dy * (Hx[1:, 1:, :] - Hx[:-1, 1:, :])

        pulse = np.exp(-0.5 * ((t - 40) / 15.0)**2)
        Ez[feed_x, feed_y, z_ground:z_patch] += pulse

        Ex[:, :, z_ground], Ey[:, :, z_ground] = 0, 0
        Ex[px_start:px_end, py_start:py_end, z_patch], Ey[px_start:px_end, py_start:py_end, z_patch] = 0, 0
        
        omega = 2 * np.pi * f_target
        Ez_phasor += Ez[:, :, z_patch] * np.exp(-1j * omega * t * dt)

        if t % max(1, steps // 20) == 0:
            progress_bar.progress((t + 1) / steps)
            
    progress_bar.empty()
    E_far_linear = np.abs(np.fft.fftshift(np.fft.fft2(Ez_phasor)))
    return E_far_linear, nx, ny

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("⚡ Antenna Gain: FDTD vs. Cavity Model")

with st.sidebar:
    st.header("Antenna Parameters")
    f_ghz = st.slider("Frequency (GHz)", 1.0, 30.0, 2.4, step=0.1)
    L_mm = st.number_input("Patch Length (mm)", 1.0, 100.0, 29.0)
    W_mm = st.number_input("Patch Width (mm)", 1.0, 100.0, 38.0)
    h_mm = st.number_input("Substrate Height (mm)", 0.1, 10.0, 1.5)
    er = st.number_input("Relative Permittivity (εr)", 1.0, 10.0, 4.4)
    
    st.header("FDTD Setup")
    steps = st.slider("Time Steps", 100, 1000, 300)
    padding = st.slider("Grid Padding", 10, 50, 20)
    run_sim = st.button("Run Comparison", type="primary")

if run_sim:
    f_target = f_ghz * 1e9
    
    with st.spinner("Running FDTD and Analytical Models..."):
        fdtd_linear, nx, ny = run_fdtd(f_target, L_mm, W_mm, h_mm, er, padding, steps)
        
    if fdtd_linear is not None:
        # Calculate Cavity Model
        cavity_linear, U, V = calculate_cavity_model(f_target, L_mm*1e-3, W_mm*1e-3, h_mm*1e-3, er, nx, ny)
        
        # Calculate Absolute Gain (dBi) for both
        gain_fdtd = calculate_directivity_dBi(fdtd_linear, U, V)
        gain_cavity = calculate_directivity_dBi(cavity_linear, U, V)
        
        st.success("Simulation Complete!")
        
        # --- Metrics Display ---
        col1, col2, col3 = st.columns(3)
        col1.metric("FDTD Computed Gain", f"{gain_fdtd:.2f} dBi")
        col2.metric("Cavity Model Gain", f"{gain_cavity:.2f} dBi")
        col3.metric("Difference", f"{abs(gain_fdtd - gain_cavity):.2f} dB")
        
        # Normalize patterns to their respective actual gains for plotting
        FDTD_dB = 20 * np.log10(fdtd_linear / np.max(fdtd_linear) + 1e-10) + gain_fdtd
        Cavity_dB = 20 * np.log10(cavity_linear / np.max(cavity_linear) + 1e-10) + gain_cavity
        
        FDTD_dB[FDTD_dB < -40] = -40
        Cavity_dB[Cavity_dB < -40] = -40
        
        # --- 2D Comparisons ---
        st.subheader("2D Planar Cuts: FDTD vs Cavity Model")
        mid_x, mid_y = nx // 2, ny // 2
        theta_axis = np.arcsin(np.linspace(-1, 1, nx)) * (180/np.pi)
        mask = ~np.isnan(theta_axis)
        
        fig2d_V = go.Figure()
        fig2d_V.add_trace(go.Scatter(x=theta_axis[mask], y=FDTD_dB[mid_x, mask], name="FDTD (Vertical / E-Plane)", line=dict(color='blue')))
        fig2d_V.add_trace(go.Scatter(x=theta_axis[mask], y=Cavity_dB[mid_x, mask], name="Cavity (Vertical / E-Plane)", line=dict(color='blue', dash='dash')))
        
        fig2d_H = go.Figure()
        fig2d_H.add_trace(go.Scatter(x=theta_axis[mask], y=FDTD_dB[mask, mid_y], name="FDTD (Horizontal / H-Plane)", line=dict(color='red')))
        fig2d_H.add_trace(go.Scatter(x=theta_axis[mask], y=Cavity_dB[mask, mid_y], name="Cavity (Horizontal / H-Plane)", line=dict(color='red', dash='dash')))
        
        fig2d_V.update_layout(title="Vertical Cut (E-Plane)", xaxis_title="Theta (Degrees)", yaxis_title="Actual Gain (dBi)", height=400)
        fig2d_H.update_layout(title="Horizontal Cut (H-Plane)", xaxis_title="Theta (Degrees)", yaxis_title="Actual Gain (dBi)", height=400)
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig2d_V, use_container_width=True)
        c2.plotly_chart(fig2d_H, use_container_width=True)
