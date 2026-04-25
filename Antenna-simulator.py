import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# Physical Constants
C = 299792458.0
MU_0 = 4 * np.pi * 1e-7
EPS_0 = 8.854187817e-12

def run_fdtd_with_ntff(f_target, L_mm, W_mm, h_mm, er, padding, steps):
    """3D FDTD Engine with dynamic meshing and 2D planar cut extraction."""
    
    # 1. Dynamic Mesh Resolution (Lambda / 12 rule)
    lambda_min = C / (f_target * np.sqrt(er))
    dx = dy = dz = min(1e-3, lambda_min / 12) # Cap at 1mm, shrink for high freq
    
    # Convert physical dimensions to cell counts
    L_cells = int(np.ceil((L_mm * 1e-3) / dx))
    W_cells = int(np.ceil((W_mm * 1e-3) / dy))
    h_cells = int(np.ceil((h_mm * 1e-3) / dz))
    
    nx = L_cells + 2 * padding
    ny = W_cells + 2 * padding
    nz = h_cells + 15 
    
    dt = 1.0 / (C * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) * 0.99
    
    # Check memory bounds to prevent crashing
    total_cells = nx * ny * nz
    if total_cells > 3e6:
        st.error(f"Grid too large ({total_cells:,} cells). Lower frequency or dimensions.")
        return None, None, None, None, None
        
    # 2. Material Setup
    eps = np.ones((nx, ny, nz)) * EPS_0
    mu = np.ones((nx, ny, nz)) * MU_0
    z_ground = 5
    z_patch = z_ground + h_cells
    eps[:, :, z_ground:z_patch] = EPS_0 * er
    
    C_e = dt / eps
    C_h = dt / mu
    
    # 3. Field Initialization
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))
    
    Ez_phasor = np.zeros((nx, ny), dtype=complex)
    
    px_start = padding
    px_end = padding + L_cells
    py_start = padding
    py_end = padding + W_cells
    
    feed_x = px_start + L_cells // 4
    feed_y = py_start + W_cells // 2
    
    progress_bar = st.progress(0)
    
    # 4. Main Loop
    for t in range(steps):
        Hx[:, :-1, :-1] -= C_h[:, :-1, :-1] / dy * (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) - C_h[:, :-1, :-1] / dz * (Ey[:, :-1, 1:] - Ey[:, :-1, :-1])
        Hy[:-1, :, :-1] += C_h[:-1, :, :-1] / dx * (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) - C_h[:-1, :, :-1] / dz * (Ex[:-1, :, 1:] - Ex[:-1, :, :-1])
        Hz[:-1, :-1, :] -= C_h[:-1, :-1, :] / dx * (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) - C_h[:-1, :-1, :] / dy * (Ex[:-1, 1:, :] - Ex[:-1, :-1, :])

        Ex[:, 1:, 1:] += C_e[:, 1:, 1:] / dy * (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) - C_e[:, 1:, 1:] / dz * (Hy[:, 1:, 1:] - Hy[:, 1:, :-1])
        Ey[1:, :, 1:] -= C_e[1:, :, 1:] / dx * (Hz[1:, :, 1:] - Hz[:-1, :, 1:]) - C_e[1:, :, 1:] / dz * (Hx[1:, :, 1:] - Hx[1:, :, :-1])
        Ez[1:, 1:, :] += C_e[1:, 1:, :] / dx * (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) - C_e[1:, 1:, :] / dy * (Hx[1:, 1:, :] - Hx[:-1, 1:, :])

        pulse = np.exp(-0.5 * ((t - 40) / 15.0)**2)
        Ez[feed_x, feed_y, z_ground:z_patch] += pulse

        Ex[:, :, z_ground] = 0
        Ey[:, :, z_ground] = 0
        Ex[px_start:px_end, py_start:py_end, z_patch] = 0
        Ey[px_start:px_end, py_start:py_end, z_patch] = 0
        
        omega = 2 * np.pi * f_target
        Ez_phasor += Ez[:, :, z_patch] * np.exp(-1j * omega * t * dt)

        if t % max(1, steps // 20) == 0:
            progress_bar.progress((t + 1) / steps)
            
    progress_bar.empty()
    
    # 5. Near-to-Far-Field
    E_far = np.fft.fftshift(np.fft.fft2(Ez_phasor))
    Pattern = np.abs(E_far)
    Pattern_dB = 20 * np.log10(Pattern / np.max(Pattern) + 1e-10)
    Pattern_dB[Pattern_dB < -40] = -40 
    
    return Ez_phasor, Pattern_dB, nx, ny, dx

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("⚡ FDTD Patch Solver: mmWave to 100 GHz")

with st.sidebar:
    st.header("1. Target Frequency")
    f_ghz = st.slider("Frequency (GHz)", 1.0, 100.0, 28.0, step=0.5)
    f_target = f_ghz * 1e9
    
    st.header("2. Physical Dimensions")
    # Defaults set smaller for mmWave frequencies
    L_mm = st.number_input("Patch Length (mm)", 0.1, 100.0, 4.0)
    W_mm = st.number_input("Patch Width (mm)", 0.1, 100.0, 5.0)
    h_mm = st.number_input("Substrate Height (mm)", 0.1, 10.0, 0.5)
    er = st.number_input("Relative Permittivity (εr)", 1.0, 10.0, 2.2)
    
    st.header("3. FDTD Settings")
    steps = st.slider("Time Steps", 100, 2000, 400)
    padding = st.slider("Grid Padding (cells)", 10, 50, 20)
    
    run_sim = st.button("Solve FDTD Mesh", type="primary")

if run_sim:
    start = time.time()
    with st.spinner("Calculating Sub-Millimeter Mesh & Running FDTD..."):
        Ez_phasor, Pattern_dB, nx, ny, dx = run_fdtd_with_ntff(f_target, L_mm, W_mm, h_mm, er, padding, steps)
    
    if Ez_phasor is not None:
        st.success(f"Simulation solved in {time.time() - start:.2f} seconds! Grid Resolution: {dx*1000:.3f} mm")
        
        # --- 2D Planar Cuts (Vertical and Horizontal) ---
        st.subheader("2D Radiation Pattern Cuts")
        
        # In spatial frequency u,v coords:
        # Center row = Vertical axis (Phi=0, E-plane for standard excitation)
        # Center column = Horizontal axis (Phi=90, H-plane)
        mid_x = nx // 2
        mid_y = ny // 2
        
        theta_axis = np.arcsin(np.linspace(-1, 1, nx)) * (180/np.pi)
        theta_axis_y = np.arcsin(np.linspace(-1, 1, ny)) * (180/np.pi)
        
        # Mask out imaginary angles outside the visible hemisphere
        mask_x = ~np.isnan(theta_axis)
        mask_y = ~np.isnan(theta_axis_y)
        
        e_plane = Pattern_dB[mid_x, :]  # Slice across Y
        h_plane = Pattern_dB[:, mid_y]  # Slice across X
        
        fig2d = go.Figure()
        fig2d.add_trace(go.Scatter(x=theta_axis_y[mask_y], y=e_plane[mask_y], mode='lines', name="Vertical Cut (E-Plane / Φ=0°)"))
        fig2d.add_trace(go.Scatter(x=theta_axis[mask_x], y=h_plane[mask_x], mode='lines', name="Horizontal Cut (H-Plane / Φ=90°)"))
        
        fig2d.update_layout(
            xaxis_title="Theta (Degrees)", 
            yaxis_title="Normalized Gain (dB)", 
            height=400,
            hovermode="x unified",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig2d, use_container_width=True)

        # --- 3D Pattern ---
        st.subheader("3D Radiation Pattern")
        u = np.linspace(-1, 1, ny)
        v = np.linspace(-1, 1, nx)
        U, V = np.meshgrid(u, v)
        
        visible_space = U**2 + V**2 <= 1
        Pattern_dB_masked = np.where(visible_space, Pattern_dB, -40)
        
        R = Pattern_dB_masked - np.min(Pattern_dB_masked)
        Z = R * np.sqrt(np.clip(1 - U**2 - V**2, 0, 1))
        X = R * U
        Y = R * V
        
        fig3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=Pattern_dB_masked, colorscale='Jet')])
        fig3d.update_layout(scene=dict(xaxis_title='U', yaxis_title='V', zaxis_title='Gain (dB)'), height=500)
        st.plotly_chart(fig3d, use_container_width=True)
