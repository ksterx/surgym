import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "Helvetica"


class Viewer:
    def __init__(self, xlim, ylim, figsize, dt, fmax):
        self.xlim = xlim
        self.ylim = ylim
        self.dt = dt
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.ims = []

        # color map
        self.cmap = plt.get_cmap("coolwarm")
        self.cnorm = Normalize(vmin=0, vmax=fmax)
        self.scalar_map = ScalarMappable(norm=self.cnorm, cmap=self.cmap)
        self.fig.colorbar(self.scalar_map)

    def show(self, pos_flatten, links: list, step: int):
        self.ax.cla()
        self.ax.set_xlim(-self.xlim, self.xlim)
        self.ax.set_ylim(0, self.ylim)
        self.ax.set_xlabel("$x$ [m]")
        self.ax.set_ylabel("$y$ [m]")
        self.ax.set_title(f" Step #{step}   dt = {self.dt} [s]")
        self.ax.set_aspect("equal", "box")
        for link in links:
            self.ax.plot(
                [link.node1.pos[0], link.node2.pos[0]],
                [link.node1.pos[1], link.node2.pos[1]],
                color=self.cmap(self.cnorm(np.linalg.norm(link.force))),
                zorder=1,
            )
        self.ax.scatter(
            pos_flatten[:, 0], pos_flatten[:, 1], s=20, c="k", marker="o", zorder=2
        )
        im = [self.ax]
        self.ims.append(im)

        # TODO: add arrows of external force

        plt.pause(self.dt)

    def save_gif(self, filename):
        print("Saving gif...")
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=1)
        ani.save(f"output/{filename}.gif", writer="imagemagick")
