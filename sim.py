# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from numpy.lib.function_base import gradient
from numpy.random.mtrand import seed


# %%
class Viewer:
    def __init__(self):
        self.height = 14
        self.width = 14
        self.shape = (self.height, self.width)
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)

    def show(self, pos):
        self.ax.cla()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        # pos = np.reshape(pos, (10, 10, 2))
        self.ax.scatter(pos[:, 0], pos[:, 1], s=30, c='r', marker='o')
        self.ax.set_aspect("equal", "box")
        plt.pause(0.001)


# %%
class MSNetwork:
    """
    Mass-Spring Network

    Indexing example
        6 - 7 - 8       (0, 2) - (1, 2) - (2, 2)
        | x | x |         |    x   |    x   |
        3 - 4 - 5   =   (0, 1) - (1, 1) - (2, 1)
        | x | x |         |    x   |    x   |
        0 - 1 - 2       (0, 0) - (1, 0) - (2, 0)
    """

    def __init__(self, k, c, l0, m):  # TODO: heterogeneous parameters
        self.obj_height = 10
        self.obj_width = 10
        self.obj_shape = (self.obj_height, self.obj_width)
        self.nx = 11
        self.ny = 11
        self.n_nodes = self.ny * self.nx
        self.links, self.link_mat = self._get_links()
        self.k = self.link_mat * k  # spring coefficients
        self.c = self.link_mat * c  # damping coefficients
        self.l0 = self.link_mat * l0  # rest lengths
        self.l = self.link_mat
        self.masses = np.ones(self.n_nodes) * m  # masses
        self.g = np.array([0, 0])

        self.viewer = Viewer()
        assert (
            self.viewer.height > self.obj_height
            and self.viewer.width > self.obj_width
        )

        self.pos, self.bottom_pos = self._initialize_position()
        self.vel = np.zeros(self.pos.shape)

    def _subidx_to_idx(self, subidx):
        """Convert subindex to linear index"""
        idx = subidx[0] * self.nx + subidx[1]
        return idx

    def _idx_to_subidx(self, idx):
        """Convert linear index to subindex"""
        i = idx // self.nx
        j = idx % self.nx
        return i, j

    def _get_neighbors(self, subidx=None, idx=None):
        """Get neighbor subindexes

        Args:
            subidx (tuple): subindex of node
            idx (int): linear index of node
        """
        if subidx is None:
            subidx = self._idx_to_subidx(idx)
        if idx is None:
            idx = self._subidx_to_idx(subidx)

        idx_neighbors = []
        subidx_neighbors = []


        def register(subidx_neighbors):
            for subidx_neighbor in subidx_neighbors:
                idx_neighbor = self._subidx_to_idx(subidx_neighbor)
                idx_neighbors.append(idx_neighbor)
                self.links.add((min(idx, idx_neighbor), max(idx, idx_neighbor)))

        xmax = self.nx - 1
        ymax = self.ny - 1

        def upper_left(i, j):
            return [(i - 1, j), (i - 1, j + 1), (i, j + 1)]

        def upper_right(i, j):
            return [(i, j + 1), (i + 1, j + 1), (i + 1, j)]

        def lower_left(i, j):
            return [(i - 1, j - 1), (i - 1, j), (i, j - 1)]

        def lower_right(i, j):
            return [(i + 1, j - 1), (i + 1, j), (i, j - 1)]

        match subidx:
            # Corners
            case (0, 0):
                subidx_neighbors = upper_right(0, 0)
                register(subidx_neighbors)
            case (0, ymax):
                subidx_neighbors = lower_right(0, ymax)
                register(subidx_neighbors)
            case (xmax, 0):
                subidx_neighbors = upper_left(xmax, 0)
                register(subidx_neighbors)
            case (xmax, ymax):
                subidx_neighbors = [(self.nx - 2, self.ny - 2), (self.nx - 1, self.ny - 2), (self.nx - 2, self.ny - 1)]
                register(subidx_neighbors)

            # Edges
            case (0, j):
                subidx_neighbors = list(set(upper_right(0, j) + lower_right(0, j)))
                register(subidx_neighbors)
            case (xmax, j):
                subidx_neighbors = list(set(upper_left(xmax, j) + lower_left(xmax, j)))
                register(subidx_neighbors)
            case (i, 0):
                subidx_neighbors = list(set(upper_right(i, 0) + upper_left(i, 0)))
                register(subidx_neighbors)
            case (i, ymax):
                subidx_neighbors = list(set(lower_right(i, ymax) + lower_left(i, ymax)))
                register(subidx_neighbors)
            case (i, j):
                subidx_neighbors = list(set(upper_right(i, j) + upper_left(i, j) + lower_right(i, j) + lower_left(i, j)))
                register(subidx_neighbors)

        return idx_neighbors, subidx_neighbors, self.links

    def _get_links(self):
        link_mat = np.zeros((self.n_nodes, self.n_nodes))
        self.links = set()  # Upper triangular matrix
        for idx in range(self.n_nodes):
            _, _, links = self._get_neighbors(idx=idx)

        for link in links:
            link_mat[link[0], link[1]] = 1

        return links, link_mat

    # set params between nodes
    def _set_param_between_nodes(self, subidx1: tuple, subidx2: tuple, param_array, param_value: float):
        idx1 = self._subidx_to_idx(subidx1)
        idx2 = self._subidx_to_idx(subidx2)
        param_array[idx1, idx2] = param_value

    def _initialize_position(self):
        """Initialize position of objects randomly"""
        space = (self.viewer.width - self.obj_width) / 2
        xlist = np.linspace(space, space + self.obj_width, self.nx)
        ylist = np.linspace(0, self.obj_height, self.ny)
        x, y = np.meshgrid(xlist, ylist)
        x = x.flatten()
        y = y.flatten()
        pos = np.stack((x, y), axis=1)
        bottom_pos = pos[: self.nx]

        return pos, bottom_pos

    def _fix_bottom(self):
        """Fix the bottom nodes to ground"""
        self.pos[: self.nx] = self.bottom_pos

    def update_euler(self, dt):
        """Update physics using Euler method"""
        self.acc = self.calc_acc()
        self.vel += self.acc * dt
        self.pos += self.vel * dt

        return self.pos

    def calc_length(self):
        for link in self.links:
            idx1, idx2 = link
            self.l[idx1, idx2] = np.linalg.norm(self.pos[idx1] - self.pos[idx2])

        return self.l

    def calc_acc(self):
        self.acc = np.zeros(self.pos.shape)
        l = self.calc_length()

        for idx in range(self.n_nodes):
            idx_neighbors, _, _ = self._get_neighbors(idx=idx)
            f_spring = np.zeros((1, 2))
            f_damper = np.zeros((1, 2))
            f_ext = np.zeros((1, 2))

            for neighbor in idx_neighbors:
                unit_vec = (self.pos[neighbor] - self.pos[idx]) / l[idx, neighbor]
                link = min(idx, neighbor), max(idx, neighbor)
                f_spring += - self.k[link] * (l[link] - self.l0[link]) * unit_vec
                f_damper += - self.c[link] * (self.vel[neighbor] - self.vel[idx]) * unit_vec
                # TODO: add external force

            self.acc[idx, :] = (f_spring + f_damper) / self.masses[idx] + self.g
        return self.acc

def simulate():
    net = MSNetwork(k=0, c=0.1, l0=1, m=5)
    viewer = Viewer()
    n_steps = 400
    for _ in tqdm.trange(n_steps):
        viewer.show(net.pos)
        net.update_euler(0.01)


if __name__ == "__main__":
    simulate()

# %%
