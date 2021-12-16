import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


class Viewer:
    def __init__(self, xlim, ylim, figsize, dt, fmax):
        self.xlim = xlim
        self.ylim = ylim
        self.dt = dt
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)

        # color map
        self.cmap = plt.get_cmap("jet")
        self.cnorm = Normalize(vmin=0, vmax=fmax)
        self.scalar_map = ScalarMappable(norm=self.cnorm, cmap=self.cmap)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.fig.colorbar(self.scalar_map)

    def show(self, pos_flatten, links: list):
        self.ax.cla()
        self.ax.set_xlim(0, self.xlim)
        self.ax.set_ylim(0, self.ylim)
        self.ax.set_aspect("equal", "box")
        for link in links:
            self.ax.plot(
                [link.node1.pos[0], link.node2.pos[0]],
                [link.node1.pos[1], link.node2.pos[1]],
                color=self.cmap(self.cnorm(np.linalg.norm(link.force))),
                zorder=1,
            )
        self.ax.scatter(
            pos_flatten[:, 0], pos_flatten[:, 1], s=40, c="k", marker="o", zorder=2
        )

        plt.pause(self.dt)
