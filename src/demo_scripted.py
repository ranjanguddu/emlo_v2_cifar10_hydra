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

    log.info(f"Loaded Model: {model}")

    url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]


    def predict(image: Image) -> Dict[str, float]:
        if image is None:
            return None

        # transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        # img_tensor = transform(image).unsqueeze(0)
        
        # inference
        with torch.no_grad():
            
            # img_tensor1= np.array((np.divide(image, 255)))
            # img_tensor1 = torch.from_numpy(img_tensor1).float()
            # #print(f'img_tensor1:{img_tensor1}, shape:{img_tensor1.shape}, type:{type(img_tensor1)}')
            # img_tensor1 = img_tensor1.permute(2,0,1)
            # image_tensor = torch.tensor(image, dtype=torch.float32)
            # image_tensor = image_tensor.permute(2,0,1)

            convert_tensor = transforms.ToTensor()
            image_tensor = convert_tensor(image)

            preds = model.forward_jit(image_tensor)
            probabilities = preds[0].tolist()

            confidences = {categories[i]: float(probabilities[i]) for i in range(10)}

        return confidences

    # for CIFAR 
    demo = gr.Interface(
        fn=predict,
        #inputs=gr.Image(source="webcam", streaming=True), #gr.Image(type="pil"),
        inputs=gr.Image(type='pil'),
        outputs=[gr.Label(num_top_classes=10)],
        #live=True
    )

    demo.launch(share=True)

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()