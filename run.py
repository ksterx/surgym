# %%
import heartrate
import hydra
import tqdm
from omegaconf import DictConfig

from utils.sim import Rectangle2D, Sim
from utils.viz import Viewer

# for debugging
# heartrate.trace(browser=True)


# %%
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Loading objects...")
    obj = Rectangle2D(id=1, cfg=cfg)

    create_viewer = cfg.viewer.render or cfg.viewer.save_gif

    if create_viewer:
        viewer = Viewer(cfg, (15, 4))

    print("Simulating...")
    for step in tqdm.trange(cfg.sim.n_steps):
        if create_viewer and step == 0:
            viewer.show(obj, step)
        obj.update_physics(dt=cfg.sim.dt)
        if create_viewer:
            viewer.show(obj, step)

    if cfg.viewer.save_gif:
        viewer.save_gif("outputs/test.gif")


# %%
if __name__ == "__main__":
    main()
    print("...Done")
