# %%
import argparse

import tqdm
from matplotlib.animation import ArtistAnimation

from utils.geometry import Rectangle2D
from utils.viz import Viewer


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--length", type=float, default=1, help="length of each link"
    )
    parser.add_argument(
        "-nx", "--nx", type=int, default=10, help="number of links in x direction"
    )
    parser.add_argument(
        "-ny", "--ny", type=int, default=10, help="number of links in y direction"
    )
    parser.add_argument("--space", type=float, default=3, help="space from boundary")
    parser.add_argument(
        "-n", "--n_steps", type=int, default=100, help="number of steps"
    )
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="time step")
    parser.add_argument("-k", "--k", type=float, default=40, help="spring constant")
    parser.add_argument("-c", "--c", type=float, default=0.1, help="damping constant")
    parser.add_argument("-g", "--gravity", action="store_true", help="apply gravity")
    parser.add_argument(
        "-fmax",
        "--fmax",
        type=float,
        default=10,
        help="normalized max force for visualization",
    )
    parser.add_argument("-save", "--save-gif", action="store_true", help="save gif")
    return parser.parse_args()


# %%
def main():
    args = parse_args()
    xlim = args.length * (args.nx - 1) / 2 + args.space
    ylim = args.length * (args.ny - 1) + args.space
    obj = Rectangle2D(
        id=1,
        gravity_on=args.gravity,
        length=args.length,
        nx=args.nx,
        ny=args.ny,
        k=args.k,
        c=args.c,
    )
    viewer = Viewer(xlim, ylim, figsize=(8, 5), dt=args.dt, fmax=args.fmax)

    for step in tqdm.trange(args.n_steps):
        if step == 0:
            viewer.show(obj.pos_flatten, obj.links, step)
        obj.update_physics(dt=args.dt)
        viewer.show(obj.pos_flatten, obj.links, step)

    if args.save_gif:
        viewer.save_gif("test")


# %%
if __name__ == "__main__":
    main()
