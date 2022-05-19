import argparse
import pathlib

import torch as th
from termcolor import cprint

import util
from glide_eval.glide_eval import run_glide_base_and_upsample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", type=str, help="a caption to visualize", required=True
    )
    parser.add_argument("--batch_size", type=int, help="", default=4, required=False)
    parser.add_argument("--sr", action="store_true", help="upsample to 4x")
    parser.add_argument(
        "--guidance_scale", type=float, help="", default=3.0, required=False
    )
    parser.add_argument(
        "--base_x",
        type=int,
        help="width of base gen. has to be multiple of 16",
        default=64,
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed",
        default=0,
    )
    parser.add_argument(
        "--base_y",
        type=int,
        help="width of base gen. has to be multiple of 16",
        default=64,
        required=False,
    )
    parser.add_argument(
        "--respace",
        type=str,
        help="Number of timesteps to use for generation. Lower is faster but less accurate. ",
        default="100",
        required=False,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Output dir for generations. Will be created if it doesn't exist with subfolders for base and upsampled.",
        default="glide_outputs",
        required=False,
    )
    parser.add_argument(
        "--upsample_temp",
        type=float,
        help="0.0 to 1.0. 1.0 can introduce artifacts, lower can introduce blurriness.",
        default=0.998,
        required=False,
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="Path to base generator. If not specified, will be created from scratch.",
        default="",
        required=False,
    )
    parser.add_argument(
        "--upsample_path",
        type=str,
        help="Path to upsampled generator. If not specified, will be created from scratch.",
        default="",
        required=False,
    )
    return parser.parse_args()


def run():
    args = parse_args()
    prompt = args.prompt
    batch_size = args.batch_size
    guidance_scale = args.guidance_scale
    base_x = args.base_x
    base_y = args.base_y
    respace = args.respace
    prefix = args.prefix
    upsample_temp = args.upsample_temp
    seed = args.seed
    base_path = args.base_path
    upsample_path = args.upsample_path
    sr = args.sr
    th.manual_seed(seed)
    cprint(f"Using seed {seed}", "green")

    if len(prompt) == 0:
        cprint("Prompt is empty, exiting.", "red")
        return

    device = th.device("cuda:6" if th.cuda.is_available() else "cpu")
    cprint(f"Selected device: {device}.", "white")
    cprint("Creating model and diffusion.", "white")
    model, _, options = util.init_model(
        model_path=base_path,
        timestep_respacing=respace,
        device=device,
        model_type="base",
    )
    model.eval()
    cprint("Done.", "green")

    cprint("Loading GLIDE upsampling diffusion model.", "white")
    model_up, _, options_up = util.init_model(
        model_path=upsample_path,
        timestep_respacing="40",
        device=device,
        model_type="upsample",
    )
    model_up.eval()
    cprint("Done.", "green")

    outputs = run_glide_base_and_upsample(
        model,
        options,
        model_up,
        options_up,
        data_dir=prefix,
        prompt=prompt,
        batch_size=batch_size,
        side_x=base_x,
        side_y=base_y,
        upsample_stage=sr,
        upsample_temp=upsample_temp,
        guidance_scale=guidance_scale,
        timestep_respacing=respace,
        sr_timestep_respacing="27",
        device=device,
    )
    if sr:
        _, hi_res_outputs = outputs
    hi_res_output_path = pathlib.Path(prefix).joinpath("hi_res_output.png")
    hi_res_outputs.save(hi_res_output_path)
    print(f"Saved hi-res output to {hi_res_output_path}")


if __name__ == "__main__":
    run()
