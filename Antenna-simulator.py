import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --- Constants ---
C = 299792458.0
MU_0 = 4 * np.pi * 1e-7
EPS_0 = 8.854187817e-12

def run_fdtd_3d(nx, ny, nz, dx, dy, dz, steps):
    """
    Core 3D FDTD Engine using Vectorized NumPy Slicing.
    """
    # Calculate time step (Courant Condition)
    dt = 1.0 / (C * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)) * 0.99 
    
    # Material grids (Free space by default)
    eps = np.ones((nx, ny, nz)) * EPS_0
    mu = np.ones((nx, ny, nz)) * MU_0
    
    # Define a simple dielectric substrate for the patch
    # Substrate from z=5 to z=10
    eps[:, :, 5:10] = EPS_0 * 4.4 
    
    # Update coefficients
    C_e = dt / eps
    C_h = dt / mu
    
    # Initialize Field Arrays (Yee Grid)
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))

    # Define Patch Antenna (Perfect Electric Conductor - PEC)
    # We enforce PEC by forcing tangential E-fields to 0
    patch_x_start, patch_x_end = nx//3, 2*nx//3
    patch_y_start, patch_y_end = ny//3, 2*ny//3
    patch_z = 10 # Top of substrate
    
    # Define Ground Plane (PEC)
    ground_z = 5
    
    # Source location (Coax feed model)
    feed_x, feed_y = nx//2, ny//3 + 2
    
    # Progress bar for Streamlit
    progress_bar = st.progress(0)
    
    # --- Main Time-Stepping Loop ---
    for t in range(steps):
        # 1. Update H fields (Ampere's Law)
        # Using numpy slicing: Hx[i,j,k] depends on Ey[i,j,k+1]-Ey[i,j,k] and Ez[i,j+1,k]-Ez[i,j,k]
        Hx[:, :-1, :-1] -= C_h[:, :-1, :-1] / dy * (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) \
                         - C_h[:, :-1, :-1] / dz * (Ey[:, :-1, 1:] - Ey[:, :-1, :-1])
                         
        Hy[:-1, :, :-1] += C_h[:-1, :, :-1] / dx * (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) \
                         - C_h[:-1, :, :-1] / dz * (Ex[:-1, :, 1:] - Ex[:-1, :, :-1])
                         
        Hz[:-1, :-1, :] -= C_h[:-1, :-1, :] / dx * (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) \
                         - C_h[:-1, :-1, :] / dy * (Ex[:-1, 1:, :] - Ex[:-1, :-1, :])

        # 2. Update E fields (Faraday's Law)
        Ex[:, 1:, 1:] += C_e[:, 1:, 1:] / dy * (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) \
                       - C_e[:, 1:, 1:] / dz * (Hy[:, 1:, 1:] - Hy[:, 1:, :-1])
                       
        Ey[1:, :, 1:] -= C_e[1:, :, 1:] / dx * (Hz[1:, :, 1:] - Hz[:-1, :, 1:]) \
                       - C_e[1:, :, 1:] / dz * (Hx[1:, :, 1:] - Hx[1:, :, :-1])
                       
        Ez[1:, 1:, :] += C_e[1:, 1:, :] / dx * (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) \
                       - C_e[1:, 1:, :] / dy * (Hx[1:, 1:, :] - Hx[:-1, 1:, :])

        # 3. Apply Source (Gaussian Pulse)
        # Injecting a soft Ez field at the feed point
        pulse = np.exp(-0.5 * ((t - 30) / 10.0)**2)
        Ez[feed_x, feed_y, ground_z:patch_z] += pulse
        
        # 4. Enforce Boundary Conditions (PEC for Patch and Ground)
        # Ground plane
        Ex[:, :, ground_z] = 0
        Ey[:, :, ground_z] = 0
        # Patch
        Ex[patch_x_start:patch_x_end, patch_y_start:patch_y_end, patch_z] = 0
        Ey[patch_x_start:patch_x_end, patch_y_start:patch_y_end, patch_z] = 0

        # Update UI
        if t % max(1, (steps // 20)) == 0:
            progress_bar.progress((t + 1) / steps)
            
    progress_bar.empty()
    return Ez # Return Z-directed E-field for visualization

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("⚡ 3D FDTD Maxwell Solver")

with st.sidebar:
    st.header("Mesh Parameters")
    st.write("Keep grid size small (< 60) for Python execution speed.")
    grid_size = st.slider("Grid Resolution (Cells)", 20, 80, 40)
    nx = ny = nz = grid_size
    dx = dy = dz = 1e-3 # 1 mm resolution
    
    steps = st.slider("Time Steps", 50, 500, 200)
    
    run_sim = st.button("Run FDTD Simulation", type="primary")

if run_sim:
    start_time = time.time()
    with st.spinner("Solving Maxwell's Equations..."):
        Ez_final = run_fdtd_3d(nx, ny, nz, dx, dy, dz, steps)
    
    st.success(f"Simulation completed in {time.time() - start_time:.2f} seconds!")
    
    # Visualize a 2D slice of the 3D field (just below the patch)
    slice_z = 9
    field_slice = Ez_final[:, :, slice_z]
    
    fig = go.Figure(data=go.Heatmap(
        z=field_slice,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Ez Amplitude")
    ))
    
    fig.update_layout(
        title=f"E_z Field Distribution at Substrate Interface (Time Step: {steps})",
        xaxis_title="X (cells)",
        yaxis_title="Y (cells)",
        width=700, height=600
    )
    
    st.plotly_chart(fig)
    
    st.markdown("""
    ### What you are looking at:
    This is a 2D slice of the 3D grid right underneath the patch antenna. 
    The ripples you see are the exact electromagnetic waves propagating outwards from the feed point, bouncing off the edges of the perfect electric conductor (PEC) patch, and establishing the resonant mode.
    """)
