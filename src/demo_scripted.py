import pyrootutils
import urllib

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple
from PIL import Image
import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
from typing import Dict

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from src import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)
    model.eval()

    

    log.info(f"Loaded Model: {model}")


    url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]


    def recognize_image(image):
        if image is None:
            return None
        image = torch.tensor(image[None, ...],dtype=torch.float32)
        image = image.permute(0,3,1,2)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        confidences = {categories[i]: float(preds[i]) for i in range(10)}

        return confidences

    im = gr.Image(shape=(32, 32), image_mode="RGB")

    demo = gr.Interface(
        fn=recognize_image,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        #live=True,
    )

    demo.launch()

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()