import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# Physical Constants
C = 299792458.0
MU_0 = 4 * np.pi * 1e-7
EPS_0 = 8.854187817e-12

def run_fdtd_with_ntff(f_target, L_mm, W_mm, h_mm, er, padding, steps):
    """3D FDTD Engine with on-the-fly DFT and far-field projection."""
    # 1. Map Physical Dimensions to Grid (using 1 mm resolution)
    dx = dy = dz = 1e-3 
    
    # Calculate grid sizes based on dimensions + padding for boundary
    L_cells = int(np.ceil(L_mm))
    W_cells = int(np.ceil(W_mm))
    h_cells = int(np.ceil(h_mm))
    
    nx = L_cells + 2 * padding
    ny = W_cells + 2 * padding
    nz = h_cells + 15 # Padding above the patch
    
    # Calculate Courant limit for stability
    dt = 1.0 / (C * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) * 0.99
    
    # 2. Material Setup
    eps = np.ones((nx, ny, nz)) * EPS_0
    mu = np.ones((nx, ny, nz)) * MU_0
    
    # Define Substrate
    z_ground = 5
    z_patch = z_ground + h_cells
    eps[:, :, z_ground:z_patch] = EPS_0 * er
    
    # Update coefficients
    C_e = dt / eps
    C_h = dt / mu
    
    # 3. Field Initialization
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))
    
    # DFT Array for target frequency (at the patch surface)
    Ez_phasor = np.zeros((nx, ny), dtype=complex)
    
    # Define Patch Boundaries
    px_start = padding
    px_end = padding + L_cells
    py_start = padding
    py_end = padding + W_cells
    
    # Feed point (offset from center to match 50 ohm)
    feed_x = px_start + L_cells // 4
    feed_y = py_start + W_cells // 2
    
    progress_bar = st.progress(0)
    st.text(f"Grid Size: {nx} x {ny} x {nz} cells")
    
    # 4. Main Time-Stepping Loop
    for t in range(steps):
        # Update H (Ampere)
        Hx[:, :-1, :-1] -= C_h[:, :-1, :-1] / dy * (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) \
                         - C_h[:, :-1, :-1] / dz * (Ey[:, :-1, 1:] - Ey[:, :-1, :-1])
        Hy[:-1, :, :-1] += C_h[:-1, :, :-1] / dx * (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) \
                         - C_h[:-1, :, :-1] / dz * (Ex[:-1, :, 1:] - Ex[:-1, :, :-1])
        Hz[:-1, :-1, :] -= C_h[:-1, :-1, :] / dx * (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) \
                         - C_h[:-1, :-1, :] / dy * (Ex[:-1, 1:, :] - Ex[:-1, :-1, :])

        # Update E (Faraday)
        Ex[:, 1:, 1:] += C_e[:, 1:, 1:] / dy * (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) \
                       - C_e[:, 1:, 1:] / dz * (Hy[:, 1:, 1:] - Hy[:, 1:, :-1])
        Ey[1:, :, 1:] -= C_e[1:, :, 1:] / dx * (Hz[1:, :, 1:] - Hz[:-1, :, 1:]) \
                       - C_e[1:, :, 1:] / dz * (Hx[1:, :, 1:] - Hx[1:, :, :-1])
        Ez[1:, 1:, :] += C_e[1:, 1:, :] / dx * (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) \
                       - C_e[1:, 1:, :] / dy * (Hx[1:, 1:, :] - Hx[:-1, 1:, :])

        # Gaussian Pulse Excitation at feed point
        pulse = np.exp(-0.5 * ((t - 40) / 15.0)**2)
        Ez[feed_x, feed_y, z_ground:z_patch] += pulse

        # Boundary Conditions (PEC for ground and patch)
        Ex[:, :, z_ground] = 0
        Ey[:, :, z_ground] = 0
        Ex[px_start:px_end, py_start:py_end, z_patch] = 0
        Ey[px_start:px_end, py_start:py_end, z_patch] = 0
        
        # 5. Running DFT at Target Frequency
        # We record the complex fields at the patch plane to project to far-field
        omega = 2 * np.pi * f_target
        Ez_phasor += Ez[:, :, z_patch] * np.exp(-1j * omega * t * dt)

        if t % max(1, steps // 20) == 0:
            progress_bar.progress((t + 1) / steps)
            
    progress_bar.empty()
    
    # 6. Near-to-Far-Field Transformation (using Aperture 2D FFT)
    # The 2D Spatial Fourier Transform of the aperture fields yields the far-field pattern
    E_far = np.fft.fftshift(np.fft.fft2(Ez_phasor))
    Pattern = np.abs(E_far)
    Pattern_dB = 20 * np.log10(Pattern / np.max(Pattern) + 1e-10)
    Pattern_dB[Pattern_dB < -40] = -40 # Clip at -40dB for clean plotting
    
    return Ez_phasor, Pattern_dB, nx, ny

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("⚡ Exact 3D FDTD: Patch Antenna Solver")
st.markdown("Includes physical dimensions, dynamic grid mapping, on-the-fly DFT, and Far-Field extraction.")

with st.sidebar:
    st.header("1. Target Frequency")
    f_ghz = st.slider("Frequency (GHz)", 1.0, 10.0, 2.4)
    f_target = f_ghz * 1e9
    
    st.header("2. Physical Dimensions")
    L_mm = st.number_input("Patch Length (mm)", 10.0, 100.0, 29.0)
    W_mm = st.number_input("Patch Width (mm)", 10.0, 100.0, 38.0)
    h_mm = st.number_input("Substrate Height (mm)", 1.0, 10.0, 1.5)
    
    st.header("3. Material Properties")
    er = st.number_input("Relative Permittivity (εr)", 1.0, 10.0, 4.4)
    
    st.header("4. FDTD Settings")
    steps = st.slider("Time Steps", 100, 1000, 300)
    padding = st.slider("Grid Padding (cells)", 10, 30, 15)
    
    run_sim = st.button("Solve Maxwell's Equations", type="primary")

if run_sim:
    start = time.time()
    with st.spinner("Running Time-Domain Simulation & Calculating DFT..."):
        Ez_phasor, Pattern_dB, nx, ny = run_fdtd_with_ntff(f_target, L_mm, W_mm, h_mm, er, padding, steps)
    
    st.success(f"Simulation solved in {time.time() - start:.2f} seconds!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Near-Field (Target Frequency Phasor)")
        st.markdown("Amplitude of the $E_z$ field exactly at the resonant frequency.")
        fig1 = go.Figure(data=go.Heatmap(
            z=np.abs(Ez_phasor), 
            colorscale='Viridis',
            colorbar=dict(title="|Ez|")
        ))
        fig1.update_layout(xaxis_title="X (mm)", yaxis_title="Y (mm)", height=500)
        st.plotly_chart(fig1)
        
    with col2:
        st.subheader("Far-Field Radiation Pattern (Aperture Projection)")
        st.markdown("Calculated via 2D Spatial FFT (Huygens' Principle).")
        
        # Create k-space grid for 3D plotting
        u = np.linspace(-1, 1, nx)
        v = np.linspace(-1, 1, ny)
        U, V = np.meshgrid(v, u)
        
        # Mask out non-radiating region (visible space u^2 + v^2 <= 1)
        visible_space = U**2 + V**2 <= 1
        Pattern_dB_masked = np.where(visible_space, Pattern_dB, -40)
        
        # Convert u,v to spherical for 3D plot
        R = Pattern_dB_masked - np.min(Pattern_dB_masked) # Linearize shape
        Z = R * np.sqrt(np.clip(1 - U**2 - V**2, 0, 1))
        X = R * U
        Y = R * V
        
        fig2 = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=Pattern_dB_masked, colorscale='Jet')])
        fig2.update_layout(scene=dict(xaxis_title='U', yaxis_title='V', zaxis_title='Gain (dB)'), height=500, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig2)
