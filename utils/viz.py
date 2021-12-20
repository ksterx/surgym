import io
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

from .geometry import Object

# Global settings
plt.rcParams["font.family"] = "Helvetica"


class Viewer:
    def __init__(self, args, figsize):
        self.args = args
        self.xlim = args.length * (args.nx - 1) / 2 + args.space
        self.ylim = args.length * (args.ny - 1) + args.space
        self.fig = plt.figure(figsize=figsize)
        grid = ImageGrid(
            self.fig,
            111,
            nrows_ncols=(1, 3),
            axes_pad=0.15,
            share_all=True,
            label_mode="L",
            cbar_location="right",
            cbar_mode="single",
        )
        self.ax1, self.ax2, self.ax3 = grid
        self.imgs = []

        # color map
        self.cmap = plt.get_cmap("coolwarm")
        self.cnorm = Normalize(vmin=0, vmax=args.fmax)
        scalar_map = ScalarMappable(norm=self.cnorm, cmap=self.cmap)
        cbar = grid.cbar_axes[0].colorbar(scalar_map)
        cbar.set_label("Force [N]")

    def show(self, obj: Object, step: int):
        self.step = step
        self.obj = obj
        self.ax_dict = {self.ax1: "force", self.ax2: "f_spring", self.ax3: "f_damper"}

        for ax, title in self.ax_dict.items():
            self.draw_ax(ax, title)
        self.ax1.set_ylabel("$y$ [m]")
        self.fig.suptitle(
            f"Step #{self.step},  $k = {self.args.k}$ [N/m],  $c = {self.args.c}$ [Nm/s],  $dt = {self.args.dt}$ [s]"
        )

        self.imgs.append(fig_to_img(self.fig))
        plt.pause(self.args.dt)

    def save_gif(self, filename):
        print("Saving gif...")
        self.imgs[0].save(
            filename, format="GIF", append_images=self.imgs[1:], save_all=True, loop=0
        )
        fn = Path(filename)
        print(f"Saved successfully to {fn.resolve()}")

    def draw_ax(self, ax, title):
        def _draw_arrow(node):
            x, y = node.pos
            dx, dy = node.f_ext / 10
            ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3, fc="k", ec="k")

        ax.cla()
        ax.set_xlim(-self.xlim, self.xlim)
        ax.set_ylim(0, self.ylim)
        ax.set_xlabel("$x$ [m]")
        ax.set_title(f"{title}", va="top", y=0.92)
        ax.set_aspect("equal", "box")

        # Plot links
        for link in self.obj.links:
            ax.plot(
                [link.node1.pos[0], link.node2.pos[0]],
                [link.node1.pos[1], link.node2.pos[1]],
                color=self.cmap(self.cnorm(np.linalg.norm(eval(f"link.{title}")))),
                zorder=1,
            )

        # Plot nodes
        ax.scatter(
            self.obj.pos_flatten[:, 0],
            self.obj.pos_flatten[:, 1],
            s=20,
            c="k",
            marker="o",
            zorder=2,
        )

        # Plot external force
        topleft_node = self.obj.find_topleft_node()
        topright_node = self.obj.find_topright_node()
        _draw_arrow(topleft_node)
        _draw_arrow(topright_node)


def fig_to_img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)
