import os
import sys
import tempfile
from random import randint

import cog
import torch as th
from termcolor import cprint

import util
from glide_eval.glide_eval import run_glide_base_and_upsample

sys.path.append("glide-text2im")

BASE_MODEL_PATH = "laionide-v3-base.pt"


class LaionidePredictor(cog.BasePredictor):
    def setup(self):
        self.data_dir = cog.Path(tempfile.mkdtemp())
        os.makedirs(self.data_dir, exist_ok=True)

        self.device = th.device("cuda:3" if th.cuda.is_available() else "cpu")
        cprint("Creating model and diffusion.", "white")
        self.model, self.diffusion, self.options = util.load_glide_model_for_eval(
            glide_path=BASE_MODEL_PATH,
            model_type="base",
            use_fp16=True,
        )
        self.model.eval()
        self.model.to(self.device)
        cprint("Done.", "green")

        cprint("Loading GLIDE upsampling diffusion model.", "white")
        (
            self.model_up,
            self.diffusion_up,
            self.options_up,
        ) = util.load_glide_model_for_eval(
            model_type="upsample",
            use_fp16=True,
        )
        self.model_up.eval()
        self.model_up.to(self.device)
        cprint("Done.", "green")

    def predict(
        self,
        prompt: str = cog.Input(
            description="Prompt to use.",
        ),
        batch_size: int = cog.Input(
            description="Batch size. Number of generations to predict",
            ge=1,
            le=8,
            default=1,
        ),
        side_x: int = cog.Input(
            description="Must be multiple of 8. Going above 64 is not recommended. Actual image will be 4x larger.",
            choices=[
                32,
                48,
                64,
                80,
                96,
                112,
                128,
            ],
            default=64,
        ),
        side_y: int = cog.Input(
            description="Must be multiple of 8. Going above 64 is not recommended. Actual image size will be 4x larger.",
            choices=[32, 48, 64, 80, 96, 112, 128],
            default=64,
        ),
        upsample_stage: bool = cog.Input(
            description="If true, uses both the base and upsample models. If false, only the (finetuned) base model is used.",
            default=True,
        ),
        upsample_temp: str = cog.Input(
            description="Upsample temperature. Consider lowering to ~0.997 for blurry images with fewer artifacts.",
            choices=["0.996", "0.997", "0.998", "0.999", "1.0"],
            default="0.997",
        ),
        guidance_scale: float = cog.Input(
            description="Classifier-free guidance scale. Higher values move further away from unconditional outputs. Lower values move closer to unconditional outputs. Negative values guide towards semantically opposite classes. 4-16 is a reasonable range.",
            default=4,
        ),
        timestep_respacing: str = cog.Input(
            description="Number of timesteps to use for base model PLMS sampling. Usually don't need more than 50.",
            choices=[
                "15",
                "17",
                "19",
                "21",
                "23",
                "25",
                "27",
                "30",
                "35",
                "40",
                "50",
                "100",
            ],
            default="27",
        ),
        sr_timestep_respacing: str = cog.Input(
            description="Number of timesteps to use for upsample model PLMS sampling. Usually don't need more than 20.",
            choices=["15", "17", "19", "21", "23", "25", "27"],
            default="17",
        ),
        seed: int = cog.Input(description="Seed for reproducibility", default=0),
    ) -> cog.Path:
        if seed == -1:
            seed = randint(0, 2**32 - 1)
            print("Random seed:", seed)
        th.manual_seed(seed)

        outputs = run_glide_base_and_upsample(
            self.model,
            self.options,
            self.model_up,
            self.options_up,
            data_dir=self.data_dir,
            prompt=prompt,
            batch_size=batch_size,
            side_x=side_x,
            side_y=side_y,
            upsample_stage=upsample_stage,
            upsample_temp=float(upsample_temp),
            guidance_scale=guidance_scale,
            timestep_respacing=timestep_respacing,
            sr_timestep_respacing=sr_timestep_respacing,
            device=self.device,
        )
        if upsample_stage:
            _, hi_res_outputs = outputs
        hi_res_output_path = cog.Path(self.data_dir).joinpath("hi_res_output.png")
        hi_res_outputs.save(hi_res_output_path)
        return hi_res_output_path
