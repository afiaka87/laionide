import time

from termcolor import cprint

import util


def run_glide_base_and_upsample(
    model,
    options,
    model_up,
    options_up,
    data_dir="outputs",
    prompt="",
    batch_size=1,
    side_x=64,
    side_y=64,
    upsample_stage=True,
    upsample_temp=0.998,
    guidance_scale=4.0,
    timestep_respacing="50",
    sr_timestep_respacing="30",
    device="cpu",
):
    side_x, side_y, upsample_temp = int(side_x), int(side_y), float(upsample_temp)
    cprint(
        f"Running base GLIDE text2im model to generate {side_x}x{side_y} samples.",
        "white",
    )
    current_time = time.time()
    low_res_samples, _ = util.sample(
        model,
        options,
        side_x,
        side_y,
        data_dir,
        prompt=prompt,
        batch_size=batch_size,
        guidance_scale=guidance_scale,
        device=device,
        prediction_respacing=timestep_respacing,
        upsample_enabled=False,
    )
    elapsed_time = time.time() - current_time
    cprint(f"Base inference time: {elapsed_time} seconds.", "green")

    low_res_pil_images = util.pred_to_pil(low_res_samples)
    low_res_pil_images.save("base_predictions.png")

    sr_base_x = int(side_x * 4.0)
    sr_base_y = int(side_y * 4.0)

    if upsample_stage:
        cprint(
            f"Upsampling from {side_x}x{side_y} to {sr_base_x}x{sr_base_y}.",
            "white",
        )
        current_time = time.time()
        hi_res_samples, _ = util.sample(
            model_up,
            options_up,
            side_x,
            side_y,
            data_dir,
            prompt=prompt,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            device=device,
            prediction_respacing=sr_timestep_respacing,
            upsample_enabled=True,
            upsample_factor=4,
            low_res=low_res_samples.to(device),
            upsample_temp=upsample_temp,
        )
        elapsed_time = time.time() - current_time
        cprint(f"SR Elapsed time: {elapsed_time} seconds.", "green")

        hi_res_pil_images = util.pred_to_pil(hi_res_samples)
        hi_res_pil_images.save("/src/sr_predictions.png")
        return low_res_pil_images, hi_res_pil_images
    return low_res_pil_images
