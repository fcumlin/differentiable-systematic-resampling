"""Visualizes the Lorenz system and the camera observation model."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches
import mpl_toolkits.mplot3d  # noqa: F401


def lorenz_dynamics(x, sigma=10.0, rho=28.0, beta=8/3, dt=0.02):
    x1, x2, x3 = x
    dx1 = sigma * (x2 - x1)
    dx2 = x1 * (rho - x3) - x2
    dx3 = x1 * x2 - beta * x3
    return np.array([x1 + dt * dx1, x2 + dt * dx2, x3 + dt * dx3])


def generate_lorenz_trajectory(T=2500, dt=0.02):
    traj = np.zeros((T, 3))
    traj[0] = np.array([1.0, 1.0, 1.0])
    for t in range(1, T):
        traj[t] = lorenz_dynamics(traj[t - 1], dt=dt)
    return traj


def camera_measurement(x, grid):
    x1, x2, x3 = x
    pos = np.array([x1, x2])
    spread = 7.0 + x3
    d2 = np.sum((grid - pos) ** 2, axis=1)
    return 10.0 * np.exp(-d2 / (2.0 * spread))


def create_pixel_grid():
    xs = np.linspace(-30, 30, 8)
    ys = np.linspace(-40, 40, 8)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def add_noise(y, smnr_db):
    sig = np.var(y)
    noise = sig / (10 ** (smnr_db / 10))
    return y + np.sqrt(noise) * np.random.randn(*y.shape)


def plot_lorenz_camera_system():
    np.random.seed(1)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11
    })

    traj = generate_lorenz_trajectory()
    grid = create_pixel_grid()

    sample_times = [600, 1200, 1800]
    colors = ["#D32F2F", "#F57C00", "#1976D2"]

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 8, figure=fig, wspace=0.45)

    # (a) Lorenz attractor
    axA = fig.add_subplot(gs[0, :3], projection="3d")
    axA.plot(traj[:, 0], traj[:, 1], traj[:, 2],
             color="#1565C0", lw=1.2, alpha=1.0)
    for t, c in zip(sample_times, colors):
        axA.scatter(*traj[t], s=90, color=c,
                    edgecolor="white", linewidth=1.8, zorder=10)

    axA.set_xlabel(r"$x_1$")
    axA.set_ylabel(r"$x_2$")
    axA.set_zlabel(r"$x_3$")
    #axA.view_init(elev=22, azim=55)
    axA.grid(alpha=0.15)
    for axis in (axA.xaxis, axA.yaxis, axA.zaxis):
        axis.pane.fill = False

    # (b) Camera model
    axB = fig.add_subplot(gs[0, 3:5])
    axB.axis("off")

    origin = np.array([0.0, 55.0])
    axB.arrow(*origin, 22, 0, width=0.8, color="#C62828")
    axB.arrow(*origin, 0, 18, width=0.8, color="#2E7D32")
    axB.arrow(*origin, -12, -12, width=0.8, color="#1565C0")

    axB.text(24, 55, r"$x_1$", color="#C62828")
    axB.text(0, 75, r"$x_2$", color="#2E7D32")
    axB.text(-20, 40, r"$x_3$", color="#1565C0")

    xt = origin + np.array([10, 8])
    axB.scatter(*xt, s=120, color="#6A1B9A")
    axB.text(xt[0] + 2, xt[1] + 4, r"$\mathbf{x}_t$")

    proj = np.array([10, -15])
    axB.annotate("", xy=proj, xytext=xt,
                 arrowprops=dict(arrowstyle="->", lw=2.5,
                                 linestyle="--", alpha=0.6,
                                 color="#6A1B9A"))

    size = 36
    axB.add_patch(matplotlib.patches.Rectangle((-size/2, -15 - size/2),
                            size, size,
                            linewidth=2, edgecolor="#37474F",
                            facecolor="#ECEFF1", alpha=0.35))

    for k in np.linspace(-size/2, size/2, 9):
        axB.plot([k, k], [-15 - size/2, -15 + size/2], lw=0.6, alpha=0.3)
        axB.plot([-size/2, size/2], [-15 + k, -15 + k], lw=0.6, alpha=0.3)

    for r, a in [(9, 0.08), (6, 0.16), (3, 0.3)]:
        axB.add_patch(matplotlib.patches.Circle(proj, r, color="#6A1B9A", alpha=a))

    #axB.text(0, -40,
    #         r"$y_{t,i}=10\exp\!\left(-\frac{2\|\mathbf{g}_i-[x_1,x_2]^\top\|^2}{7+x_3}\right)$",
    #         ha="center", fontsize=10)

    # (c) Observations
    gsC = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 5:], wspace=0.25)
    obs = []
    for t in sample_times:
        y = camera_measurement(traj[t], grid)
        obs.append(add_noise(y, 0).reshape(8, 8))
    vmax = max(o.max() for o in obs)

    for i, (o, c, t) in enumerate(zip(obs, colors, sample_times)):
        ax = fig.add_subplot(gsC[0, i])
        im = ax.imshow(o, cmap="plasma", vmin=0, vmax=vmax)
        ax.set_title(fr"$t={t}$", color=c, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(c)
            s.set_linewidth(2.5)

    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pixel intensity")

    y_label = 0.93
    fig.text(0.25, y_label, "(a)", fontsize=14) # fontweight='bold'
    fig.text(0.50, y_label, "(b)", fontsize=14)
    fig.text(0.75, y_label, "(c)", fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    return fig


if __name__ == "__main__":
    fig = plot_lorenz_camera_system()
    fig.savefig("lorenz63_camera.png", dpi=300, bbox_inches="tight")
    plt.show()
