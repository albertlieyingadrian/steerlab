"""The note's cover: an illustration drawn by FLUX.1-schnell itself, the model the note studies (no text in it).

modal run make_cover.py                 # 4 candidates into report/figures/cover_candidates/
modal run make_cover.py --pick 3        # copy candidate seed 3 to report/figures/hero.jpg and the thumbnail
"""

import io
import os
import shutil

import modal

from modal_app import cache, gen_image, hf_secret

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "report", "figures")
THUMB = os.path.join(HERE, "..", "shared", "thumbs", "t2i-geneval.jpg")
PROMPT = (
    "A flat gouache illustration on a cream paper background, split down the middle. On the left, a warm cluttered "
    "kitchen table at golden hour with cups, a vase, apples and a cat, objects overlapping. On the right, the same "
    "objects separated and laid out in a tidy grid on a plain white background, each one alone with space around it, "
    "like a product catalogue. Muted palette with orange accents, soft grain, no text, no letters, no words."
)
app = modal.App("t2i-geneval-cover")


@app.function(
    image=gen_image.add_local_python_source("modal_app", "geneval_core"),
    gpu="H100",
    volumes={"/cache": cache},
    secrets=[hf_secret],
    timeout=1200,
)
def draw(seeds):
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    out = {}
    for s in seeds:
        img = pipe(
            PROMPT,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=1536,
            height=1024,
            max_sequence_length=256,
            generator=torch.Generator("cuda").manual_seed(s),
        ).images[0]
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        out[s] = buf.getvalue()
    return out


@app.local_entrypoint()
def main(pick: int = -1, seeds: str = "1,2,3,4"):
    cand = os.path.join(FIG, "cover_candidates")
    if pick >= 0:
        from PIL import Image

        src = os.path.join(cand, f"c{pick}.jpg")
        shutil.copy(src, os.path.join(FIG, "hero.jpg"))
        im = Image.open(src).convert("RGB")
        w, h = im.size
        th = round(w * 2 / 3)
        im.crop((0, (h - th) // 2, w, (h - th) // 2 + th)).resize((720, 480), Image.LANCZOS).save(THUMB, quality=88)
        print("hero.jpg and thumbnail from", src)
        return
    os.makedirs(cand, exist_ok=True)
    for s, b in draw.remote([int(x) for x in seeds.split(",")]).items():
        with open(os.path.join(cand, f"c{s}.jpg"), "wb") as f:
            f.write(b)
        print("wrote", f"c{s}.jpg")
