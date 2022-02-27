# Laionide 
[![replicate](https://img.shields.io/badge/Replicate-visit%20replicate-lightgrey?style=flat)](https://replicate.com/afiaka87/laionide-v3) [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WUkAE6vpKeri2axo17ROwCvpPN8W6SIy?usp=sharing)

Laionide is a project to create a large, unfiltered, open GLIDE model.

**Update: Laionide ~~V2~~ _V3_, finetuned on CC12M and other datasets. Produces fewer watermarks.**

Much ♥ to stability.ai for donating needed compute to [Laion](https://discord.gg/8pSACZJk)

![](samples/rouge_robot.png?raw=true)

Replicate codebase: https://github.com/afiaka87/pyglide-replicate

Finetuning script: https://github.com/afiaka87/glide-finetune

Inference locally: https://github.com/afiaka87/pyglide

## Laionide (v3)

Files:
- [laionide-v3-base.pt](https://github.com/afiaka87/laionide/releases/download/Checkpoints/laionide-v3-base.pt)

Inference:
- [replicate](https://replicate.com/afiaka87/laionide-v3)
- [colab](https://gist.github.com/afiaka87/8655b15c94bf0e80f586ce54cfe39ab5#file-laionide-v3-ipynb)
- [locally]()

Results:
- [comparison to openai W&B report](https://wandb.ai/afiaka87/laionide-v3-glide/reports/Laionide-Version-3-Benchmark--VmlldzoxNjE0MTE3)

Notes:
- You can use `laionide-v2-sr.pt` to upscale the outputs from `laionide-v3-base.pt`.
- There are watermarks in some outputs. You can try to prompt engineer this away, but it isn't always possible. `royalty free` seems to work well. 

### Training details:
- Finetuned `laionide-v2-base.pt` for 9 epochs on a subset of CC12M (~1.5 million pairs), COCO (~100K pairs), virtual genome (~100K pairs), and open images localized annotations (~800K pairs). 
- To keep consistancy with the paper, training has 20% chance of unconditional/empty token.

## Laionide (v2)

Files:
- [laionide-v2-base.pt](https://github.com/afiaka87/laionide/releases/download/Checkpoints/laionide-v2-base.pt)
- [laionide-v2-sr.pt](https://github.com/afiaka87/laionide/releases/download/Checkpoints/laionide-v2-sr.pt)

Inference:
- [replicate](https://replicate.com/afiaka87/laionide-v2)

Results:
- [comparison to openai W&B report](https://wandb.ai/afiaka87/glide_compare/reports/Finetuning-GLIDE-on-LAION-does-it-work---VmlldzoxNTg3MTkz)

### Training details:
- Data was removed from training given any of the following:
- Value of 'NSFW' or 'LIKELY' in the nsfw column from LAION's metadata.
- Images with ratio greater than 1.3 (or) less than 0.8 were chosen.
- The original width or original height is less than 256 pixels.
- Images, with specific slurs in its caption, were removed.


## Laionide (v1)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/afiaka87/5f64e4de49b50554270a0a6ece243014#file-laionide-ipynb) [![replicate](https://img.shields.io/badge/Replicate-visit%20replicate-lightgrey?style=flat)](https://replicate.com/afiaka87/laionide)

Files (Links currently broken)
- https://www.dropbox.com/s/mchzd28p9ees0db/laionide-base.pt
- https://www.dropbox.com/s/7cxn0gelotpocun/laionide-upsample.pt

### Training details:

- GLIDE (filtered) finetuned on Laion400M 
- 1M captions seen for 2 epochs with 20% chance of unconditional token
- Upsample model saw 200K samples.
