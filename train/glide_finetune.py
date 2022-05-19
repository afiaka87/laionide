import os
from typing import Tuple

import torch as th
import wandb
from braceexpand import braceexpand
from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from tqdm import tqdm


def exists(val):
    return val is not None


def base_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where tokens is a tensor of shape (batch_size, seq_len), masks is a tensor of shape (batch_size, seq_len) and reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1].
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, reals = [x.detach().clone().to(device) for x in batch]
    reals.requires_grad_(True)
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = th.randn_like(reals, device=device)
    x_t = (
        glide_diffusion.q_sample(reals, timesteps, noise=noise)
        .requires_grad_(True)
        .to(device)
    )

    _, C = x_t.shape[:2]
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    ).requires_grad_(True)
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.detach().to(device))


def upsample_train_step(
    glide_model: Text2ImUNet,
    glide_diffusion: SpacedDiffusion,
    batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
    device: str,
):
    tokens, masks, low_res_image, upsampled_image = [
        x.detach().clone().to(device) for x in batch
    ]
    # upsampled_image = interpolate(low_res_image, scale_factor=4, mode='bicubic', align_corners=False).requires_grad_(True)
    timesteps = th.randint(
        0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device
    )
    noise = th.randn_like(upsampled_image, device=device)
    x_t = glide_diffusion.q_sample(upsampled_image, timesteps, noise=noise).to(device)
    _, C = x_t.shape[:2]
    # gaussian blur the input to the model (but not the upsampled image) kernel/sigma from unCLIP/DALLE-2
    # if random() < 0.1: # low_res_image = TF.gaussian_blur(low_res_image, kernel_size=3, sigma=0.6).requires_grad_(False).to(device)
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = th.split(model_output, C, dim=1)
    return th.nn.functional.mse_loss(epsilon, noise.to(device).detach())


from loader import TextImageDataset
from tqdm import trange
from wds_loader import glide_wds_loader

from util import load_model, pred_to_pil, sample, save_model, wandb_setup


def run_glide_finetune(
    data_dir="./data",
    batch_size=1,
    learning_rate=1e-5,
    side_x=64,
    side_y=64,
    resize_ratio=1.0,
    uncond_p=0.0,
    resume_ckpt="",
    checkpoints_dir="./finetune_checkpoints",
    use_fp16=False,
    device="cpu",
    freeze_transformer=False,
    freeze_diffusion=False,
    project_name="glide_finetune",
    use_captions=True,
    num_epochs=100,
    log_frequency=100,
    sample_bs=1,
    sample_gs=8.0,
    use_webdataset=False,
    image_key="jpg",
    caption_key="txt",
    enable_upsample=False,
    upsample_factor=4,
    image_to_upsample="low_res_face.png",
):
    if "~" in data_dir:
        data_dir = os.path.expanduser(data_dir)
    if "~" in checkpoints_dir:
        checkpoints_dir = os.path.expanduser(checkpoints_dir)

    distr_backend = None
    # is_root = True
    # if args.deepspeed:
    #     distr_backend = distributed_utils.set_backend_from_args(args)
    #     distr_backend.initialize()
    # is_root = (distr_backend.get_local_rank() == 0)
    is_root = True  # TODO

    # Start wandb logging
    if is_root:
        # Create the checkpoint/output directories
        os.makedirs(checkpoints_dir, exist_ok=True)
        wandb_run = wandb_setup(
            batch_size=batch_size,
            side_x=side_x,
            side_y=side_y,
            learning_rate=learning_rate,
            use_fp16=use_fp16,
            device=device,
            data_dir=data_dir,
            base_dir=checkpoints_dir,
            project_name=project_name,
        )
        print("Wandb setup.")
    else:
        wandb_run = None

    # Model setup
    glide_model, glide_diffusion, glide_options = load_model(
        glide_path=resume_ckpt,
        use_fp16=use_fp16,
        freeze_transformer=freeze_transformer,
        freeze_diffusion=freeze_diffusion,
        activation_checkpointing=False,
        model_type="base" if not enable_upsample else "upsample",
    )

    if is_root:
        number_of_params = sum(x.numel() for x in glide_model.parameters())
        print(f"Number of parameters: {number_of_params}")
        number_of_trainable_params = sum(
            x.numel() for x in glide_model.parameters() if x.requires_grad
        )
        print(f"Trainable parameters: {number_of_trainable_params}")

    if distr_backend is not None:
        rank = distr_backend.get_local_rank()
        shards = distr_backend.get_world_size()
    else:
        rank = 0
        shards = 1
    # Data setup
    print("Loading data...")
    if use_webdataset:
        urls = list(
            braceexpand(
                "pipe:aws s3 cp s3://laion-watermark/clear/{00001..00162}.tar -"
            )
        )
        total_size = len(urls)
        print(f"Using {len(urls)} files of {total_size} for rank {rank} of {shards}")
        dataset = glide_wds_loader(
            urls=urls,
            enable_text=use_captions,
            enable_metadata=False,  # TODO
            image_key=image_key,
            caption_key=caption_key,
            metadata_key="json",
            tokenizer=glide_model.tokenizer,
            base_x=side_x,
            base_y=side_y,
            uncond_p=uncond_p,
        )
    else:
        dataset = TextImageDataset(
            base_dir=data_dir,
            side_x=side_x,
            side_y=side_y,
            resize_ratio=resize_ratio,
            uncond_p=uncond_p,
            shuffle=True,
            tokenizer=glide_model.tokenizer,
            text_ctx_len=glide_options["text_ctx"],
            use_captions=use_captions,
            upscale_factor=upsample_factor,
            rank=rank,
            world_size=shards,
        )

    print(f"Loading optimizer with local rank {rank} of {shards}")
    optimizer = th.optim.AdamW(
        glide_model.parameters(), lr=learning_rate, weight_decay=0.05
    )
    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=(device == "cuda"),
    )
    glide_model.to(device)
    glide_model.train()

    # Training setup
    outputs_dir = "./outputs"
    if is_root:
        os.makedirs(outputs_dir, exist_ok=True)
    for epoch in trange(num_epochs):
        if is_root:
            print(f"Starting epoch {epoch}")

    if enable_upsample:
        train_step = upsample_train_step
    else:
        train_step = base_train_step

    os.makedirs(checkpoints_dir, exist_ok=True)
    log = {}
    if is_root:
        print(f"Starting epoch {epoch}")
    for train_idx, batch in enumerate(dataloader):
        log = {}
        with th.cuda.amp.autocast(enabled=True):
            accumulated_loss = train_step(
                glide_model=glide_model,
                glide_diffusion=glide_diffusion,
                batch=batch,
                device=device,
            )
        if distr_backend is not None:
            glide_model.backward(accumulated_loss)
            glide_model.step()
            accumulated_loss = distr_backend.average_all(accumulated_loss)
        else:
            accumulated_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if is_root:
            log = {**log, "iter": train_idx, "loss": accumulated_loss.item()}
            tqdm.write(f"loss: {accumulated_loss.item():.6f}")
        # Sample from the model
        if train_idx % log_frequency == 0 and is_root:
            tqdm.write(f"Sampling from model at iteration {train_idx}")
            with th.cuda.amp.autocast(enabled=True):
                samples, _caption = sample(
                    glide_model=glide_model.module
                    if distr_backend is not None
                    else glide_model,
                    glide_options=glide_options,
                    side_x=side_x,
                    side_y=side_y,
                    data_dir=data_dir,
                    prompt="",
                    batch_size=1,
                    guidance_scale=0,
                    device="cuda",
                    prediction_respacing="100",
                    upsample_enabled=enable_upsample,
                    upsample_factor=4,
                    image_to_upsample="",
                    upsample_temp=1.0,
                )
            sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
            pred_to_pil(samples).save(sample_save_path)
            if exists(wandb_run):
                wandb_run.log(
                    {
                        **log,
                        "iter": train_idx,
                        "samples": wandb.Image(sample_save_path, caption=_caption),
                    }
                )
            tqdm.write(f"Saved sample {sample_save_path}")
        if train_idx % 1000 == 0:
            save_model(
                glide_model=glide_model,
                checkpoints_dir=checkpoints_dir,
                train_idx=train_idx,
                epoch=epoch,
            )
            tqdm.write(
                f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{train_idx}.pt"
            )

        if exists(wandb_run) and is_root:
            wandb_run.log(log)
    if epoch % 5 == 0:
        tqdm.write(f"Finished epoch, saving checkpoint for {epoch}")
        save_model(
            glide_model=glide_model,
            checkpoints_dir=checkpoints_dir,
            train_idx=train_idx,
            epoch=epoch,
        )
