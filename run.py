# %%
import argparse

import tqdm

from utils.geometry import Rectangle2D
from utils.viz import Viewer


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=1)
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--space", type=float, default=3)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--k", type=float, default=40)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--fmax", type=float, default=10)
    parser.add_argument("--gravity", action="store_true")
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


# %%
if __name__ == "__main__":
    main()
