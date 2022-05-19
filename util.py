## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.

import os
import sys
sys.path.append("glide-text2im")

from glob import glob
from random import choice
from typing import Tuple

import numpy as np
import PIL
import torch as th

import os

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder
from PIL import Image

MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]


def pred_to_pil(pred: th.Tensor) -> Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return Image.fromarray(reshaped.numpy())


def load_glide_model_for_eval(
    glide_path: str = "",
    activation_checkpointing: bool = False,
    model_type: str = "base",
    use_fp16: bool = False,
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True

    options["use_fp16"] = use_fp16
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    if activation_checkpointing:
        glide_model.use_checkpoint = True
    if len(glide_path) > 0:  # user provided checkpoint
        # if not using_deepspeed:
        assert os.path.exists(glide_path), "glide path does not exist"
        weights = th.load(str(glide_path), map_location="cpu")
        glide_model.load_state_dict(weights)
    else:  # use default checkpoint from openai
        glide_model.load_state_dict(
            load_checkpoint(model_type, device="cpu")
        )  # always load to cpu, saves memory
    if use_fp16:
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    glide_model.eval()
    return glide_model, glide_diffusion, options


def get_uncond_tokens_mask(tokenizer: Encoder):
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return uncond_tokens, uncond_mask


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> Tuple[th.tensor, th.tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)
    else:
        tokens = tokenizer.encode(prompt)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, context_len)
        return tokens, mask


def load_glide_model_for_training(
    glide_path: str = "",
    use_fp16: bool = False,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True

    options["use_fp16"] = use_fp16
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    if activation_checkpointing:
        glide_model.use_checkpoint = True

    glide_model.requires_grad_(True)
    if freeze_transformer:
        glide_model.transformer.requires_grad_(False)
        glide_model.transformer_proj.requires_grad_(False)
        glide_model.token_embedding.requires_grad_(False)
        glide_model.padding_embedding.requires_grad_(False)
        glide_model.positional_embedding.requires_grad_(False)
    if freeze_diffusion:
        glide_model.time_embed.requires_grad_(False)
        glide_model.input_blocks.requires_grad_(False)
        glide_model.middle_block.requires_grad_(False)
        glide_model.output_blocks.requires_grad_(False)
        glide_model.out.requires_grad_(False)
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True

    options["use_fp16"] = use_fp16
    glide_model, glide_diffusion = create_model_and_diffusion(**options)
    if activation_checkpointing:
        glide_model.use_checkpoint = True
    if len(glide_path) > 0:  # user provided checkpoint
        # if not using_deepspeed:
        assert os.path.exists(glide_path), "glide path does not exist"
        weights = th.load(str(glide_path), map_location="cpu")
        glide_model.load_state_dict(weights)
    else:  # use default checkpoint from openai
        glide_model.load_state_dict(
            load_checkpoint(model_type, device="cpu")
        )  # always load to cpu, saves memory
    if use_fp16:
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    return glide_model, glide_diffusion, options


def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert("RGB")
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


# Sample from the base model.


@th.inference_mode()
def sample(
    glide_model,
    glide_options,
    side_x,
    side_y,
    data_dir,
    prompt="",
    batch_size=1,
    guidance_scale=4,
    device="cuda",
    prediction_respacing="100",
    upsample_enabled=False,
    upsample_factor=4,
    low_res=None,
    upsample_temp=1.0,
):
    glide_model.del_cache()
    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        noise_schedule=glide_options["noise_schedule"],
        timestep_respacing=prediction_respacing,
    )
    # Create the text tokens to feed to the model.
    tokens = glide_model.tokenizer.encode(prompt)
    tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(
        tokens, glide_options["text_ctx"]
    )

    # Pack the tokens together into model kwargs.
    if upsample_enabled:
        full_batch_size = batch_size
        model_kwargs = dict(
            tokens=th.tensor([tokens] * batch_size, device=device),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )
    else:
        # Create the classifier-free guidance tokens (empty)
        uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask(
            [], glide_options["text_ctx"]
        )
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )
        full_batch_size = batch_size * 2

    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        beta = eval_diffusion.betas[
            int(
                ts.flatten()[0].item()
                / glide_options["diffusion_steps"]
                * len(eval_diffusion.betas)
            )
        ]
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        current_prediction_pil = pred_to_pil((x_t - eps * (beta**0.5))[:batch_size])
        current_prediction_pil.save("current_prediction.png")
        return th.cat([eps, rest], dim=1)

    if upsample_enabled:
        model_kwargs["low_res"] = low_res
        up_side_y = side_y * upsample_factor
        up_side_x = side_x * upsample_factor
        noise = (
            th.randn((batch_size, 3, up_side_y, up_side_x), device=device)
            * upsample_temp
        )

        samples = eval_diffusion.ddim_sample_loop(
            glide_model,
            (full_batch_size, 3, up_side_y, up_side_x),  # only thing that's changed
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        glide_model.del_cache()
        return samples, prompt
    else:
        samples = eval_diffusion.plms_sample_loop(
            cfg_model_fn,
            (full_batch_size, 3, side_y, side_x),  # only thing that's changed
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        glide_model.del_cache()
        return samples, prompt


def deepspeed_config_from_args(args):
    deepspeed_config = {
        "zero_optimization": {
            "stage": 0,
        },
        "ignore_unused_parameters": False,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.ga_steps,
        "gradient_clipping": 1.0,
        "tensorboard": {
            "enabled": True,
            "output_path": f"tensorboard_logs/{args.project_name}",
            "job_name": f"{args.project_name}",
        },
        # 'fp16': {
        #     'enabled': args.use_fp16,
        #     'initial_scale_power': 20,
        # },
        "amp": {
            "enabled": True,
            "opt_level": "O1",
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
    }
    return deepspeed_config


def save_model(
    glide_model: th.nn.Module, checkpoints_dir: str, train_idx: int, epoch: int
):
    th.save(
        glide_model.state_dict(),
        os.path.join(checkpoints_dir, f"glide-ft-{epoch}x{train_idx}.pt"),
    )
    print(f"SAVED STATE DICT {str(glide_model.state_dict())}")
    print(
        f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{epoch}x{train_idx}.pt"
    )
