# laionide 


**update: Laionide ~~V2~~ _V3_, finetuned on CC12M and other datasets. Produces fewer watermarks.**

For the code used to create the replicate demo:
https://github.com/afiaka87/pyglide-replicate

Shout out to stability.ai for donating the compute to [laion](https://discord.gg/8pSACZJk) needed for this to be possible.

## Laionide (v3)

Files:
- [laionide-v3-base.pt](https://github.com/afiaka87/laionide/releases/download/Checkpoints/laionide-v3-base.pt)

Inference:
- [replicate](https://replicate.com/afiaka87/laionide-v3)
- [colab](https://gist.github.com/afiaka87/b1500fd06a9d5991d7bd90aa9c2dc6da#file-laionide-v2-ipynb)
- [locally](https://github.com/afiaka87/pyglide)

Results:
- [comparison to openai W&B report](https://wandb.ai/afiaka87/laionide-v3-glide/reports/Laionide-Version-3-Benchmark--VmlldzoxNjE0MTE3)

Notes:
- You can use `laionide-v2-sr.pt` to upscale the outputs from `laionide-v3-base.pt`.
- There are watermarks in some outputs. You can try to prompt engineer this away, but it isn't always possible. `royalty free` seems to work well. 

Training details:
- finetuned `laionide-v2-base.pt` for 9 epochs on a subset of CC12M (~1.5 million pairs), COCO (~100K pairs), virtual genome (~100K pairs), and open images localized annotations (~800K pairs). 
- 20% of unconditional/empty token, per the paper.

## Laionide (v2)

Files:
- [laionide-v2-base.pt](https://github.com/afiaka87/laionide/releases/download/Checkpoints/laionide-v2-base.pt)
- [laionide-v2-sr.pt](https://github.com/afiaka87/laionide/releases/download/Checkpoints/laionide-v2-sr.pt)

Inference:
- [replicate](https://replicate.com/afiaka87/laionide-v2)

Results:
- [comparison to openai W&B report](https://wandb.ai/afiaka87/glide_compare/reports/Finetuning-GLIDE-on-LAION-does-it-work---VmlldzoxNTg3MTkz)

Training details:
- Data was removed from training given any of the following:
- Value of 'NSFW' or 'LIKELY' in the nsfw column from LAION's metadata (better than nothing)
- Images which originally had an aspect ratio greater than 1.3 (or) less than 0.8
- The original width or original height is less than 256 pixels.
- Captions were checked against a list of slurs. If a slur is in the caption, it is removed. I won't be publishing the slurs.


## Laionide (v1)

Files (Links currently broken)
- https://www.dropbox.com/s/mchzd28p9ees0db/laionide-base.pt
- https://www.dropbox.com/s/7cxn0gelotpocun/laionide-upsample.pt

Training details

- GLIDE finetune on laion over both the base model 
- 1M captions seen for 2 epochs with 20% chance of unconditional token
- Upsample model sees 200K samples.

Inference
- [colab](https://gist.github.com/afiaka87/5f64e4de49b50554270a0a6ece243014#file-laionide-ipynb)
- [replicate](https://replicate.com/afiaka87/laionide)

### Deliverables

code used to finetune:
https://github.com/afiaka87/glide-finetune

the code to reconstruct the replicate demo is at this branch:
https://github.com/afiaka87/pyglide-replicate/tree/laionide
