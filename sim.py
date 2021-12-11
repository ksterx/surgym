# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from numpy.random.mtrand import seed


# %%
class Viewer:
    def __init__(self):
        self.height = 12
        self.width = 12
        self.shape = (self.height, self.width)
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)

    def show(self, pos, vel, acc):
        self.ax.cla()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.scatter(pos[:, 0], pos[:, 1], s=1)
        self.fig.canvas.draw()
        plt.pause(0.001)


# %%
class MSDNetwork:
    """
    Indexing example
        6 - 7 - 8
        | x | x |
        3 - 4 - 5
        | x | x |
        0 - 1 - 2
    """
    def __init__(self):
        self.obj_height = 10
        self.obj_width = 10
        self.obj_shape = (self.obj_height, self.obj_width)
        self.n_height_nodes = 10
        self.n_width_nodes = 10
        self.n_nodes = self.n_height_nodes * self.n_width_nodes

        self.viewer = Viewer()
        assert (
            self.viewer.window_height > self.obj_height and self.viewer.window_width > self.obj_width
        )


    def _subidx_to_idx(self, subidx):
        """Convert subindex to linear index"""
        idx = subidx[0] * self.n_width_nodes + subidx[1]
        return idx

    def _idx_to_subidx(self, idx):
        """Convert linear index to subindex"""
        i = idx // self.n_width_nodes
        j = idx % self.n_width_nodes
        return i, j

    def _initialize_position(self):
        """Initialize position of objects randomly"""
        space = (self.window_width - self.obj_width) / 2
        xlist = np.linspace(space, space + self.obj_width, self.n_width_nodes)
        ylist = np.linspace(0, self.window_height, self.n_height_nodes)
        x, y = np.meshgrid(xlist, ylist)
        x = x.flatten()
        y = y.flatten()
        self.pos = np.stack((x, y), axis=1)
        self.bottom_pos = self.pos[:self.n_width_nodes]
        self.vel = np.zeros(self.pos.shape)
        self.acc = np.zeros((self.pos.shape)

    def _fix_bottom(self):
        """Fix the bottom nodes to ground"""
        self.pos[:self.n_width_nodes] = self.bottom_pos

    def update_physics(self, dt):
        self.vel += self.acc * dt
        self.pos += self.vel * dt

    def calc_acc(self):
        raise NotImplementedError


# calculate acceleration on nodes using hooke's law + gravity
def get_acc(pos, vel, ci, cj, spring_length, spring_coefficient, gravity):
    # initialize
    acc = np.zeros(pos.shape)

    # Hooke's law: F = - k * displacement along spring
    sep_vec = pos[ci, :] - pos[cj, :]
    sep = np.linalg.norm(sep_vec, axis=1)
    dL = sep - spring_length
    ax = -spring_coefficient * dL * sep_vec[:, 0] / sep
    ay = -spring_coefficient * dL * sep_vec[:, 1] / sep
    np.add.at(acc[:, 0], ci, ax)
    np.add.at(acc[:, 1], ci, ay)
    np.add.at(acc[:, 0], cj, -ax)
    np.add.at(acc[:, 1], cj, -ay)

    # gravity
    acc[:, 1] += gravity

    return acc


def apply_bcondition(pos, vel, boxsize, shape):

    # apply box boundary conditions, reverse velocity if outside box
    for ax in range(2):
        is_out = np.where(pos[:, ax] < 0)
        pos[is_out, ax] *= -1
        vel[is_out, ax] *= -1

        is_out = np.where(pos[:, ax] > boxsize)
        pos[is_out, ax] = boxsize - (pos[is_out, ax] - boxsize)
        vel[is_out, ax] *= -1

    # Fix the position of bottom nodes
    for k in range(len(pos)):
        i, j = idx_to_sub(shape, k)
        if i == 0:
            pos[sub_to_idx(shape, i, j), 1] = 0

    return (pos, vel)


# simulation main loop
def main():
    """N-body simulation"""

    # Simulation parameters
    N = 5  # Number of nodes per linear dimension
    t = 0  # current time of the simulation
    dt = 0.1  # timestep
    Nt = 400  # number of timesteps
    spring_coeff = 20  # Hooke's law spring coefficient
    gravity = -0.1  # strength of gravity
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # construct spring nodes / initial conditions
    boxsize = 3
    xlin = np.linspace(1, 2, N)

    x, y = np.meshgrid(xlin, xlin)
    x = x.flatten()
    y = y.flatten() - 1

    pos = np.vstack((x, y)).T
    vel = np.zeros(pos.shape)
    acc = np.zeros(pos.shape)

    # add a bit of random noise
    np.random.seed(17)  # set the random number generator seed
    vel += 0.01 * np.random.randn(N ** 2, 2)

    # construct spring network connections
    ci = []
    cj = []
    #  o--o
    for r in range(0, N):
        for c in range(0, N - 1):
            idx_i = sub_to_idx([N, N], r, c)
            idx_j = sub_to_idx([N, N], r, c + 1)
            ci.append(idx_i)
            cj.append(idx_j)
    # o
    # |
    # o
    for r in range(0, N - 1):
        for c in range(0, N):
            idx_i = sub_to_idx([N, N], r, c)
            idx_j = sub_to_idx([N, N], r + 1, c)
            ci.append(idx_i)
            cj.append(idx_j)
    # o
    #   \
    #     o
    for r in range(0, N - 1):
        for c in range(0, N - 1):
            idx_i = sub_to_idx([N, N], r, c)
            idx_j = sub_to_idx([N, N], r + 1, c + 1)
            ci.append(idx_i)
            cj.append(idx_j)
    #     o
    #   /
    # o
    for r in range(0, N - 1):
        for c in range(0, N - 1):
            idx_i = sub_to_idx([N, N], r + 1, c)
            idx_j = sub_to_idx([N, N], r, c + 1)
            ci.append(idx_i)
            cj.append(idx_j)

    # calculate spring rest-lengths
    spring_length = np.linalg.norm(pos[ci, :] - pos[cj, :], axis=1)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = fig.add_subplot(111)

    # Simulation Main Loop
    for i in tqdm.trange(Nt):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # apply boundary conditions
        pos, vel = apply_bcondition(pos, vel, boxsize, [N, N])

        # update accelerations
        acc = get_acc(pos, vel, ci, cj, spring_length, spring_coeff, gravity)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # plot in real time
        if plotRealTime or (i == Nt - 1):
            plt.cla()
            plt.plot(pos[[ci, cj], 0], pos[[ci, cj], 1], color="blue")
            plt.scatter(pos[:, 0], pos[:, 1], s=10, color="blue")
            ax.set(xlim=(0, boxsize), ylim=(0, boxsize))
            ax.set_aspect("equal", "box")
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 1, 2, 3])
            plt.pause(0.001)

    # Save figure
    # plt.savefig('springnetwork.png', dpi=240)
    plt.show()


if __name__ == "__main__":
    main()

# %%
