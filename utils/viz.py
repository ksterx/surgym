import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
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

        # color map
        self.cmap = plt.get_cmap("coolwarm")
        self.cnorm = Normalize(vmin=0, vmax=fmax)
        self.scalar_map = ScalarMappable(norm=self.cnorm, cmap=self.cmap)
        # divider = mpl_toolkits.axes_grid1.make_axes_locatable(self.ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
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

        # TODO: add arrows of external force

        plt.pause(self.dt)
