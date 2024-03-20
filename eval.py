import logging
import os
import re
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from train import setup_training

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(config_path="config", version_base="1.1")
def main(cfg: DictConfig):
    log.info(f"** running from source tree at {hydra.utils.get_original_cwd()}")

    log.info(f"** configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    setup = setup_training(cfg)
    for cp_file in sorted(os.listdir(cfg.agent.hi.init_from)):
        m = re.match(r"checkpoint_(\d*)\.pt", cp_file)
        if m:
            checkpoint_samples = int(m.group(1))
            with open(os.path.join(cfg.agent.hi.init_from, cp_file), "rb") as fd:
                setup.agent.load_checkpoint(fd)

            setup.agent.eval()
            setup.eval_fn(setup, n_samples=checkpoint_samples)

    setup.close()


if __name__ == "__main__":
    main()
