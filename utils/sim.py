import math

import numpy as np
import tqdm

from .viz import Viewer


class Sim:
    def __init__(self, cfg, obj):
        self.dt = cfg.sim.dt
        self.n_steps = cfg.sim.n_steps

        if obj == "rect2d":
            self.obj = Rectangle2D(id=1, cfg=cfg)

        else:
            raise NotImplementedError

    def update_euler(self, dt):
        self.obj.update

    def update_rk4(self, dt):
        k1p = self.update_euler(dt)
        k1v = self.update_euler(dt)
        k2p = self.update_euler(dt)

    def run(self):

        for step in tqdm.trange(self.n_steps):
            pass


def calc_f_link(node_a, node_b):
    """Calculate the force between two nodes"""


class Node:
    def __init__(
        self, mass: float, id: int, gravity_on: bool, pos, vel, acc, dim: int = 2
    ):
        self.mass = mass  # mass (kg)
        self.id = id  # unique id
        self.dim = dim  # dimension
        self.pos = pos  # position (m)
        self.vel = vel  # velocity (m/s)
        self.acc = acc  # acceleration (m/s^2)
        self.neighbors = set()  # set of neighbor nodes
        self.connected_links = set()  # set of connected links
        if gravity_on:
            self.g = 9.81  # gravitational acceleration (m/s^2)
        else:
            self.g = 0
        self.f_ext = np.zeros(self.dim)  # external force (N)

    def is_neighbor(self, node, dist_threshold):
        """Check if a node is within a certain distance from this node at initial position"""
        return np.linalg.norm(abs(self.pos - node.pos)) <= dist_threshold

    def update_physics(self, dt, f_ext=np.zeros(2), fix_pos=False, method="euler"):
        """Update physics using Euler method"""
        if fix_pos:
            self.vel = np.zeros(self.dim)
            self.acc = np.zeros(self.dim)
            self.force = np.zeros(self.dim)
        else:
            self.f_ext = f_ext
            if method == "euler":
                self.force = self._apply_force()
                self.acc = self.force / self.mass
                self.vel += self.acc * dt
                self.pos += self.vel * dt
            elif method == "rk4":
                self.force = self._apply_force()
                self.acc = self.force / self.mass
                k1v = self.acc = self.force / self.mass
                k1p =

                raise NotImplementedError
            else:
                raise ValueError("Unknown method")

    def _apply_force(self):
        self.f_link = np.zeros(self.dim)
        self.f_g = self.mass * np.array([0, -self.g])  # TODO: 2d or 3d
        for link in self.connected_links:
            if link.node1.id == self.id:
                self.f_link -= link.calc_force()
            elif link.node2.id == self.id:
                self.f_link += link.calc_force()
            else:
                raise ValueError("Node ID not found in link")

        return self.f_link + self.f_g + self.f_ext

    def _calcurate_acc(self, vel, pos):
        self.f_link = np.zeros(self.dim)
        self.f_g = self.mass * np.array([0, -self.g])  # TODO: 2d or 3d
        # TODO: f_link formula (move node1 and node2 virtuallly??) / Use copy.copy() ??


class Link:
    def __init__(
        self, node1: Node, node2: Node, k: float, c: float, rest_length: float, id: int
    ):
        # link between node1 and node2 (i.e. node1.id < node2.id)
        if node1.id > node2.id:
            node1, node2 = node2, node1
        elif node1.id == node2.id:
            raise ValueError("Link cannot be between the same node")
        else:
            pass
        self.node1 = node1
        self.node2 = node2
        self.k = k  # spring coefficient
        self.c = c  # damping coefficient
        self.rest_length = rest_length  # rest length
        self.id = id  # unique id
        self.calc_force()
        self.is_connected = True

    def update(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def calc_force(self):
        self.length = np.linalg.norm(self.node2.pos - self.node1.pos)
        unit_vec = (self.node2.pos - self.node1.pos) / self.length
        self.f_spring = -self.k * (self.length - self.rest_length) * unit_vec
        self.f_damper = -self.c * (self.node2.vel - self.node1.vel) * unit_vec
        self.force = self.f_spring + self.f_damper
        return self.force

    def break_link(self):
        self.k = 0
        self.c = 0
        self.is_connected = False


class Object:
    def __init__(
        self,
        id: int,
        cfg,
        init_pos_flatten,
        dist_threshold,
        dim: int = 2,
    ):
        def register_nodes():
            """Register nodes in the object from the flattened position array"""
            for id in range(self.n_nodes):
                node = Node(
                    pos=self.pos_flatten[id],
                    vel=np.zeros(self.dim),
                    acc=np.zeros(self.dim),
                    mass=self.masses[id],
                    gravity_on=cfg.sim.gravity_on,
                    id=id,
                )
                self.nodes.append(node)

        def register_links():
            """Register links between nodes that are within a certain distance at initial position"""
            id = 0
            for node1 in self.nodes:
                for node2 in self.nodes:
                    if node1.is_neighbor(node2, self.dist_threshold):
                        if node1 == node2:
                            continue
                        link = Link(
                            node1,
                            node2,
                            k=self.k,
                            c=self.c,
                            rest_length=np.linalg.norm(node1.pos - node2.pos),
                            id=id,
                        )  # TODO: determine k and c for each link
                        self.links.add(link)
                        node1.neighbors.add(node2)  # node2 is a neighbor of node1
                        node1.connected_links.add(link)  # node1 is connected to link
                        id += 1
            self.links = list(self.links)  # convert set to list

        self.id = id
        self.dim = dim
        self.k = cfg.obj.k
        self.c = cfg.obj.c
        self.dist_threshold = dist_threshold
        self.nodes = []
        self.links = set()
        self.pos_flatten = init_pos_flatten
        assert self.pos_flatten.shape[1] == dim  # 2d or 3d
        self.n_nodes = self.pos_flatten.shape[0]
        self.masses = [1 for _ in range(self.n_nodes)]  # TODO: mass

        register_nodes()
        register_links()

    @classmethod
    def update_physics(self, dt):
        self.pos_flatten = np.zeros(self.pos_flatten.shape)
        for node in self.nodes:
            node.update_physics(dt=dt)
            self.pos_flatten[node.id] = node.pos


class Rectangle2D(Object):
    def __init__(self, id, cfg):
        self.length = cfg.obj.length
        self.nx = cfg.obj.nx
        self.ny = cfg.obj.ny
        self.n_nodes = self.nx * self.ny
        init_pos_flatten, self.bottom_pos = self._set_initial_position()
        dist_threshold = self._get_dist_threshold()
        super().__init__(
            id=id,
            cfg=cfg,
            init_pos_flatten=init_pos_flatten,
            dist_threshold=dist_threshold,
            dim=2,
        )

    def _set_initial_position(self):
        xlist = np.linspace(
            -self.length * (self.nx - 1) / 2, self.length * (self.nx - 1) / 2, self.nx
        )
        ylist = np.linspace(0, self.length * (self.ny - 1), self.ny)
        x, y = np.meshgrid(xlist, ylist)
        x = x.flatten()
        y = y.flatten()
        pos_flatten = np.vstack((x, y)).T
        assert pos_flatten.shape == (self.nx * self.ny, 2)
        bottom_pos = pos_flatten[: self.nx]

        return pos_flatten, bottom_pos

    def _get_dist_threshold(self):
        return math.ceil(math.sqrt(2) * self.length * 1000) / 1000

    def update_physics(self, dt):
        self.pos_flatten = np.zeros(self.pos_flatten.shape)
        for node in self.nodes:
            # If node is at the bottom, fix it
            if node.id < self.nx:
                node.update_physics(dt=dt, fix_pos=True)
            # If node is at the top left corner, apply external force
            elif node.id == self.nx * (self.ny - 1):
                f_ext = np.array([-10, 10])
                node.update_physics(dt=dt, f_ext=f_ext)
            elif node.id == self.n_nodes - 1:
                f_ext = np.array([10, 10])
                node.update_physics(dt=dt, f_ext=f_ext)
            else:
                node.update_physics(dt=dt)
            self.pos_flatten[node.id] = node.pos

    def find_top_left_node(self):
        return self.nodes[self.nx * (self.ny - 1)]

    def find_top_right_node(self):
        return self.nodes[self.n_nodes - 1]
