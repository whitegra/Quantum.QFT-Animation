# Quantum Wavefunction Visualization with PennyLane  
This project simulates and visualizes the evolution of a quantum wavefunction using a parameterized quantum circuit built with PennyLane. It features a dual-view animation: a 2D heatmap and a 3D surface plot of the evolving probability distribution, rendered frame by frame with Matplotlib.

## Key Features  
**Quantum Simulation**: Utilizes Hadamard gates, phase shifts, and the Quantum Fourier Transform (QFT) to simulate a wavefunction evolution.  
**GPU Acceleration (Optional)**: Automatically uses JAX for GPU-accelerated NumPy operations if available.  
**Dual Visualization**: Displays both a 2D heatmap and a 3D surface plot of the quantum state's probability distribution.  
**Smooth Animation**: Creates a dynamic animation of the wavefunction's evolution over time with rotating 3D perspective.  
**Export-Ready**: Saves output as a high-quality .gif file for presentations or educational materials.

## Requirements  
Install the following Python packages:
bash
pip install pennylane matplotlib jax jaxlib pillow
JAX is optional but recommended for better performance (especially on GPU).

## How It Works  
**Quantum Circuit Design**  
A parameterized quantum circuit is defined using `qml.qnode`. It:
- Places qubits in superposition using Hadamard gates.  
- Applies phase shifts (RZ gates) based on a time-evolving angle θ.  
- Applies a Quantum Fourier Transform to simulate interference.

**Simulation Grid**  
The grid resolution (Nx x Ny, default 16x16) determines the number of qubits (log2(Nx * Ny)). Each quantum state maps to a point on the grid.

**Animation**  
Matplotlib’s `FuncAnimation` updates the visual output every frame:
- **2D Heatmap**: Shows the real-time probability distribution.  
- **3D Surface Plot**: Offers a spatial view of the quantum wave.

**Output**  
The animation is saved as a .gif and also viewable in a Jupyter Notebook via `to_jshtml()`.

## Customization Tips  
- **Grid Size**: Change Nx, Ny to increase or decrease resolution (e.g., 8×8 for faster simulations).  
- **Circuit Logic**: Modify the quantum circuit (e.g., add entanglement) to explore different dynamics.  
- **Color Maps**: Adjust `cmap` in `imshow` and `plot_surface` for different visual styles.

##  Learning Objectives  
This project aims to help:
- Understand quantum state representation and evolution.  
- Visualize the abstract concept of quantum probabilities.  
- Explore basic quantum gates and transformations.  
- Practice using PennyLane, JAX, and Matplotlib for quantum computing applications.


