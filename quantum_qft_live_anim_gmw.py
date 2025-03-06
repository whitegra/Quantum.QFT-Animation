# -*- coding: utf-8 -*-
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# optional (BUT RECOMMENDED) use JAX for GPU acceleration
try:
    import jax.numpy as jnp
    from jax import random
    np = jnp  # replace NumPy with JAX for speed
    print("JAX successfuly enabled for GPU acceleration.")
    #or not
except ImportError:
    print("JAX not found, youre stuck with NumPy. Good luck.")

# define the dimensions of the quantum grid
Nx, Ny = 16, 16  # number of points in the x and y directions for visualization, this will dtermine your qbit usage so be mindful. 16 x 16 is a good res
n_qubits = int(np.log2(Nx * Ny))  # determine the number of qubits required for the grid (these are limited if local)

# initialize a quantum device using pennylane, this simulates a quantum computer that we can run our circuit on
dev = qml.device("default.qubit", wires=n_qubits) # you can replace this with any quantum device, or run locally with pennylane.

@qml.qnode(dev)
def quantum_wavefunction(theta):
    """
    quantum circuit simulating wavefunction evolution.
    applies superposition, phase shifts, and the quantum fourier transform (qft).
    returns the probability distribution of the quantum state.
    """

    # 1. apply hadamard gates to all qubits to create superposition:
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    # 2 apply phase shift gates (rz) to introduce phase evolution over time:
    for i in range(n_qubits):
        qml.RZ(theta, wires=i)

    # 2. apply quantum fourier transform to simulate interference patterns:
    qml.QFT(wires=range(n_qubits))

    # 3. return the probability distribution of the final quantum state:
    return qml.probs(wires=range(n_qubits))

# now to create the figures with two subplots: a 2d heatmap and a 3d plot
fig, axs = plt.subplots(1, 2, figsize=(14, 7), facecolor="black")
fig.patch.set_facecolor("black")  # set background color to black for aesthetics
fig.tight_layout()  # adjust layout to prevent overlap

# define the 2d subplot for heatmap visualization, and the 3D one for the probability wave.
ax2d = axs[0]
ax3d = fig.add_subplot(122, projection="3d")

# MAKE SURE the subplot backgrounds are black for consistency, set them both individually. 
ax2d.set_facecolor("black")
ax3d.set_facecolor("black")

# create a mesh grid for the x and y axes
# this is used to plot the 2d and 3d probability distributions (2D)
X, Y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))

def update(frame):
    """
    update function for animation.
    recalculates the probability distribution at each frame to visualize quantum evolution.
    """

    theta = frame * np.pi / 20  # phase evolution over time (theta)
    probs = quantum_wavefunction(theta).reshape((Nx, Ny))  # probs reshape probability distribution to match grid over time per frame

    # update left plot (2d probability heatmap)
    ax2d.clear()
    ax2d.imshow(probs, cmap="plasma", extent=[-1, 1, -1, 1], interpolation="bilinear")
    ax2d.set_title("2d quantum probability grid", color="cyan", fontsize=12, fontweight="bold")
    ax2d.set_xlabel("x [a.u.]", color="cyan")
    ax2d.set_ylabel("y [a.u.]", color="cyan")
    ax2d.set_xticks([])
    ax2d.set_yticks([])

    # update right plot (3d quantum probability wave)
    ax3d.clear()
    cmap = plt.get_cmap("plasma") 
    colors = cmap(probs / np.max(probs))  # normalize colors for better contrast

    # plot the 3d probability surface
    ax3d.plot_surface(X, Y, probs, facecolors=colors, edgecolor="none", shade=False, alpha=0.8)
    # add a wireframe overlay for depth perception (THIS LOOKS BETTER, but optional. )
    ax3d.plot_wireframe(X, Y, probs, color="white", alpha=0.15)

    ax3d.set_title("3d quantum probability wave", color="cyan", fontsize=12, fontweight="bold")
    ax3d.set_xlabel("x [a.u.]", color="cyan")
    ax3d.set_ylabel("y [a.u.]", color="cyan")
    ax3d.set_zlabel("probability", color="cyan")

    # remove tick labels for cleaner visualization, again, optional but looks the best. 
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])

    # dynamically adjust z-axis limits per frame
    ax3d.set_zlim(0, np.max(probs) + 0.1)

    # apply smooth camera movement for animation effect because it looks cooler
    ax3d.view_init(elev=30 + np.sin(frame / 30) * 15, azim=frame * 2)

# create animation with 100 frames and 200ms interval between frames (faster interval = shorter frames)
ani = animation.FuncAnimation(fig, update, frames=100, interval=200)

# save animation as gif  (lower fps for smoother playback) (optional)
ani.save("quantum_dual_visualization2.gif", writer="pillow", fps=15, dpi=100)

# to display the animation in jupyter notebook:
from IPython.display import HTML
display(HTML(ani.to_jshtml()))
