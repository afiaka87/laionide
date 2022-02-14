# laionide

checkpoints for glide finetuned on laion and other datasets. wip.

## Checkpoints (plan to upload via github releases, work-in-progress)
https://www.dropbox.com/s/mchzd28p9ees0db/laionide-base.pt
https://www.dropbox.com/s/7cxn0gelotpocun/laionide-upsample.pt

## Training details:

GLIDE finetune on laion over both the base model (1M captions seen for 2 epochs with 20% chance of unconditional token) and the upsample model (tune for around 200K samples, same as above with 20% unconditional)

## Inference: 

### colab notebook
https://gist.github.com/afiaka87/5f64e4de49b50554270a0a6ece243014#file-laionide-ipynb

### replicate
https://replicate.com/afiaka87/laionide

### locally

https://github.com/afiaka87/pyglide
(just specify `--glide_base_path` and `--glide_sr_path`)

## Deliverables

code used to finetune:
https://github.com/afiaka87/glide-finetune

the code to reconstruct the replicate demo is at this branch:
https://github.com/afiaka87/pyglide-replicate/tree/laionide
