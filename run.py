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
    parser.add_argument("-r", "--render", action="store_true", help="render animation")
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
    print("Loading objects...")
    obj = Rectangle2D(
        id=1,
        gravity_on=args.gravity,
        length=args.length,
        nx=args.nx,
        ny=args.ny,
        k=args.k,
        c=args.c,
    )

    create_viewer = args.render or args.save_gif

    if create_viewer:
        viewer = Viewer(args, (15, 4))

    print("Simulating...")
    for step in tqdm.trange(args.n_steps):
        if create_viewer and step == 0:
            viewer.show(obj, step)
        obj.update_physics(dt=args.dt)
        if create_viewer:
            viewer.show(obj, step)

    if args.save_gif:
        viewer.save_gif("output/test.gif")


# %%
if __name__ == "__main__":
    main()
    print("Done.")
