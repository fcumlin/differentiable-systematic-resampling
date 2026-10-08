"""Visualizes handwritten characters from motion data."""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch


def integrate_to_position(traj):
    x = np.cumsum(traj[:, 0])
    y = np.cumsum(traj[:, 1])
    z = traj[:, 2]
    return x, y, z


def plot_character_stroke(trajectory, title="", ax=None, cmap="viridis"):
    """
    Plot a handwritten character by integrating motion data.
    Color encodes time, line width encodes third channel.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()

    x = np.cumsum(trajectory[:, 0])
    y = np.cumsum(trajectory[:, 1])
    z = trajectory[:, 2]

    t = np.linspace(0, 1, len(x))
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)

    for i in range(len(x) - 1):
        ax.plot(
            x[i:i+2],
            y[i:i+2],
            color=plt.cm.get_cmap(cmap)(t[i]),
            linewidth=0.6 + 2.5 * z_norm[i],
            alpha=0.9,
        )

    ax.scatter(x[0], y[0], s=40, c="black", marker="o", zorder=3)
    ax.text(x[0], y[0], " start", fontsize=9, va="bottom")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    return ax


def add_noise(traj, smnr_db):
    signal_power = np.var(traj)#np.mean(traj**2)
    noise_power = signal_power / (10**(smnr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*traj.shape)
    return traj + noise



# ===== Load and plot =====
mat_data = scipy.io.loadmat(
    "/mimer/NOBACKUP/groups/naiss2025-22-438/characters/mixoutALL_shifted.mat"
)
def get_character_label(mat_data=mat_data, trajectory_idx=0):
    """Extract character label from the 'key' field in consts."""
    consts = mat_data['consts'][0][0]
    keys = consts['key'][0]
    
    # The dataset has multiple samples per character. Figure out which character
    # class this trajectory belongs to.
    total_trajectories = len(mat_data['mixout'][0])
    num_chars = len(keys)
    samples_per_char = total_trajectories // num_chars
    
    char_idx = trajectory_idx // samples_per_char
    if char_idx >= len(keys):
        char_idx = len(keys) - 1
    
    char_label = keys[char_idx][0]
    return char_label
print(get_character_label())
trajectory = mat_data["mixout"][0][0].T[:130]
print(trajectory.shape)

char = trajectory

char_clean = char
char_noisy = add_noise(char, smnr_db=10)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

plot_character_stroke(
    char_clean,
    title="(a)",
    ax=axes[0],
)

plot_character_stroke(
    char_noisy,
    title="(b)",
    ax=axes[1],
)

plt.tight_layout()
plt.savefig("character_clean_vs_noisy.png", dpi=300, bbox_inches="tight")
