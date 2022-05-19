import io
from random import random

import PIL
import torch as th
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from train.loader import pil_image_to_norm_tensor
from util import get_tokens_and_mask, get_uncond_tokens_mask
import webdataset as wds

def glide_wds_loader(
    urls,
    enable_text=True,
    enable_image=True,
    enable_metadata=False,
    image_key="jpg",
    caption_key="txt",
    metadata_key="json",
    cache_path=None,
    tokenizer=None,
    base_x=64,
    base_y=64,
    uncond_p=0.2,
):
    base_image_shape = (base_x, base_y)
    dataset = wds.WebDataset(
        urls,
        cache_dir=cache_path,
        cache_size=10**10,
        handler=wds.handlers.warn_and_continue,
    )

    def filter_dataset_laion(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and metadata_key not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset_laion)

    def preprocess_dataset(item):
        # tokens, mask, base_tensor = None, None, None
        image_data = item[image_key]
        original_pil_image = PIL.Image.open(io.BytesIO(image_data)).convert("RGB")
        original_pil_image.load()

        image_transform = T.Compose([
            T.RandomResizedCrop(base_image_shape, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation=T.InterpolationMode.LANCZOS),
        ])
        if not enable_text or random() < uncond_p: # Classifier free guidance
            tokens, mask = get_uncond_tokens_mask(tokenizer)
        else:
            caption = item[caption_key].decode("utf-8")
            description = caption.strip()
            tokens, mask = get_tokens_and_mask(tokenizer, description)
        base_pil_image = image_transform(original_pil_image)
        base_tensor = pil_image_to_norm_tensor(base_pil_image)
        return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor

    transformed_dataset = filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.warn_and_continue
    )
    return transformed_dataset
