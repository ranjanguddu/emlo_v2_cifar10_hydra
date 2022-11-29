import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple
import urllib
import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from typing import Dict

from src import utils

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    ckpt = torch.load(cfg.ckpt_path)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    log.info(f"Loaded Model: {model}")

    transforms = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    def recognize_digit(image):
        if image is None:
            return None
        image = transforms(image).unsqueeze(0)
        logits = model(image)
        preds = F.softmax(logits, dim=1).squeeze(0).tolist()
        return {str(i): preds[i] for i in range(10)}

    def predict(inp_img: Image) -> Dict[str, float]:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        img_tensor = transform(inp_img).unsqueeze(0)  # transform and add batch dimension

        

        # inference
        with torch.no_grad():
            out = model(img_tensor)
            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            confidences = {categories[i]: float(probabilities[i]) for i in range(10)}

        
        return confidences

    '''
    gr.Interface(
                    fn=predict, 
                    inputs=gr.Image(type="pil"), 
                    outputs=gr.Label(num_top_classes=10)
                ).launch(share=True)
                '''

    #im = gr.Image(shape=(28, 28), image_mode="L", invert_colors=True, source="canvas")

    demo = gr.Interface(
        #fn=recognize_digit,
        fn=predict,
        #inputs=[im],
        #inputs=gr.Image(source="webcam", streaming=True), #gr.Image(type="pil"),
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=10)],
        #live=True
    )

    demo.launch(share=True)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demo.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()