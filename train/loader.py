import pathlib
import time
from random import choice, randint, random

import numpy as np
import PIL
import torch as th
from torch.utils.data import Dataset
from torchvision import transforms as T

from util import get_tokens_and_mask, get_uncond_tokens_mask


def random_resized_crop(image, shape, resize_ratio=1.0):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.Compose(
        [
            T.RandomResizedCrop(
                shape,
                scale=(resize_ratio, 1.0),
                ratio=(1.0, 1.0),
                interpolation=T.InterpolationMode.LANCZOS,
            )
        ]
    )
    return image_transform(image)


def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image)).float().permute(2, 0, 1) / 127.5 - 1.0


def get_image_files_dict(base_path, shard=0, num_shards=1):
    image_files = [
        *base_path.glob("**/*.png"),
        *base_path.glob("**/*.jpg"),
        *base_path.glob("**/*.jpeg"),
        *base_path.glob("**/*.bmp"),
    ]
    return {
        image_file.stem: image_file
        for image_file in image_files
        if image_file.is_file()
    }


def get_text_files_dict(base_path):
    text_files = [*base_path.glob("**/*.txt")]
    return {
        text_file.stem: text_file for text_file in text_files if text_file.is_file()
    }


def get_shared_stems(image_files_dict, text_files_dict):
    image_files_stems = set(image_files_dict.keys())
    text_files_stems = set(text_files_dict.keys())
    return list(image_files_stems & text_files_stems)


class TextImageDataset(Dataset):
    def __init__(
        self,
        base_dir="",
        side_x=64,
        side_y=64,
        resize_ratio=0.75,
        shuffle=False,
        tokenizer=None,
        text_ctx_len=128,
        uncond_p=0.0,
        use_captions=False,
        upscale_factor=4,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        assert len(base_dir) > 0, "base_dir must be a valid directory"
        base_dir = pathlib.Path(base_dir)
        self.image_files = get_image_files_dict(
            base_dir, shard=rank, num_shards=world_size
        )
        print(f"{len(self.image_files)} images found on rank {rank}.")
        if use_captions:
            self.text_files = get_text_files_dict(base_dir)
            # self.text_files = [image_file.with_suffix(".txt") for image_file in self.image_files.values()]
            self.keys = list(self.image_files.keys())
            # self.keys = get_shared_stems(self.image_files, self.text_files)
            print(f"Found {len(self.keys)} images.")
            print(f"Using {len(self.text_files)} text files.")
            print(f"But not really.")
        else:
            self.text_files = None
            self.keys = list(self.image_files.keys())
            print(f"Found {len(self.keys)} images.")
            print(f"NOT using text files. Restart with --use_captions to enable...")
            time.sleep(3)

        self.resize_ratio = resize_ratio
        self.text_ctx_len = text_ctx_len

        self.shuffle = shuffle
        self.prefix = base_dir
        self.side_x = side_x
        self.side_y = side_y
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def get_caption(self, ind):
        key = self.keys[ind]
        image_file = self.image_files[key]
        text_file = image_file.with_suffix(".txt")
        if text_file.is_file():
            descriptions = open(text_file, "r").readlines()
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            return choice(descriptions)
        else:
            return None

    def __getitem__(self, ind):
        try:
            key = self.keys[ind]
            image_file = self.image_files[key]
            original_pil_image = PIL.Image.open(image_file).convert("RGB")
            original_pil_image = random_resized_crop(
                original_pil_image,
                shape=(self.side_x, self.side_y),
                resize_ratio=self.resize_ratio,
            )
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
            # base_pil_image = original_pil_image.resize((self.side_x, self.side_y), PIL.Image.LANCZOS)
            # 20% chance to use the empty caption/unconditional
            # if self.text_files is None or self.uncond_p < random():
        if self.uncond_p < random():
            tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        else:
            try:
                description = self.get_caption(ind)
            except Exception:
                print(f"An exception occurred trying to load file {ind}.")
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)

            if description is None:
                print(f"{image_file.name} has no caption. Using uncond.")
                tokens, mask = get_uncond_tokens_mask(self.tokenizer)
            elif len(description) == 0:
                print(f"No descriptions found for {key}. Skipping.")
                return self.skip_sample(ind)
            else:
                tokens, mask = get_tokens_and_mask(
                    tokenizer=self.tokenizer,
                    prompt=description,
                    context_len=self.text_ctx_len,
                )
        base_tensor = pil_image_to_norm_tensor(original_pil_image)
        # original_tensor = pil_image_to_norm_tensor(original_pil_image)
        return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor
