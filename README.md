# laionide 


**update: LaionideV2, trained even more.**

![/compare.png](/compare.png)
_Top - OpenAI, Bottom - LaionideV2_

For the code used to create the replicate demo:
https://github.com/afiaka87/pyglide-replicate

## Laionide (v2)

### Run inference (v2)

- [replicate](https://replicate.com/afiaka87/laionide)
- [colab](https://gist.github.com/afiaka87/b1500fd06a9d5991d7bd90aa9c2dc6da#file-laionide-v2-ipynb)

### Checkpoints (v2)
https://www.dropbox.com/s/2kf6nsrtis3bcwh/laion-glide-v2-final.pt
https://www.dropbox.com/s/pbai2yk479ia2qy/laion-glide-v2-sr-final.pt

### Watermarks (v2)

There are watermarks in some outputs. You can try to prompt engineer this away, but it isn't always possible. `royalty free` seems to work well. 

### What is better? What is worse?

You can see a comparison over the same captions in this [weights & biases report](https://wandb.ai/afiaka87/glide_compare/reports/Finetuning-GLIDE-on-LAION-does-it-work---VmlldzoxNTg3MTkz)

### Data filtering

Data was removed from training given any of the following:
- Value of 'NSFW' or 'LIKELY' in the nsfw column from LAION's metadata (better than nothing)
- Images which originally had an aspect ratio greater than 1.3 (or) less than 0.8
- The original width or original height is less than 256 pixels.
- Captions were checked against a list of slurs. If a slur is in the caption, it is removed. I won't be publishing the slurs.

Shout out to stability.ai for donating the compute to [laion](https://discord.gg/8pSACZJk) needed for this to be possible.

## Laionide (v1)
### Checkpoints (v1)

- https://www.dropbox.com/s/mchzd28p9ees0db/laionide-base.pt

- https://www.dropbox.com/s/7cxn0gelotpocun/laionide-upsample.pt

### Training details (v1)

GLIDE finetune on laion over both the base model (1M captions seen for 2 epochs with 20% chance of unconditional token) and the upsample model (tune for around 200K samples, same as above with 20% unconditional)

### Inference (v1)

### colab notebook
https://gist.github.com/afiaka87/5f64e4de49b50554270a0a6ece243014#file-laionide-ipynb

### replicate
https://replicate.com/afiaka87/laionide

### locally

https://github.com/afiaka87/pyglide
(just specify `--glide_base_path` and `--glide_sr_path`)

### Deliverables

code used to finetune:
https://github.com/afiaka87/glide-finetune

the code to reconstruct the replicate demo is at this branch:
https://github.com/afiaka87/pyglide-replicate/tree/laionide
