import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING
from PIL import ImageDraw
from copy import deepcopy

model_id = "google/paligemma-3b-mix-224"

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)

device = "cuda:0"
dtype = torch.bfloat16

model = PaliGemmaForConditionalGeneration.from_pretrained(
    "/home/dzb/pretrained/paligemma3b",
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained("/home/dzb/pretrained/paligemma3b")

image = Image.open("1.png").convert("RGB")

prompt = "caption this image"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
    
    
def draw_bounding_box(image: Image.Image, box: list[int], color: str = "red", width: int = 2) -> Image.Image:
    """
    Draw a bounding box on an image.

    Args:
        image (PIL.Image.Image): The image to draw on.
        box (list[int]): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].
            It is a normalized bounding box, where coordinates are in the range int [0, 1024].
        color (str): The color of the bounding box.
        width (int): The width of the bounding box lines.

    Returns:
        PIL.Image.Image: The image with the bounding box drawn.
    """
    draw = ImageDraw.Draw(image)
    y_min, x_min, y_max, x_max = box
    x_min = int(x_min / 1024 * image.width)
    y_min = int(y_min / 1024 * image.height)
    x_max = int(x_max / 1024 * image.width)
    y_max = int(y_max / 1024 * image.height)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    return image