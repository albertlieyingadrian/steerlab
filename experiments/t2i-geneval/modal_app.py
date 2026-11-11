"""Modal side of the text-to-image GenEval pilots.

Three classes share one Volume (images live there, only paths and scores travel):
- Grader (A10G): the official GenEval checker, on the old stack it needs (torch 1.13 + mmcv-full 1.7.2 + mmdet 2.28.2).
- Gen (H100): a diffusers text-to-image pipeline (FLUX.1-schnell by default).
- Scorer (A10G): PickScore (preference), an independent checker (OWLv2 + a different CLIP), BLIP-VQA (CompBench guard).

Deploy once with `modal deploy modal_app.py`; the local scripts look the classes up by name.
"""

import io
import os
import time

import modal

APP = "t2i-geneval"
VOL = "/vol"
app = modal.App(APP)
vol = modal.Volume.from_name("t2i-geneval", create_if_missing=True)
cache = modal.Volume.from_name("t2i-geneval-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
HERE = os.path.dirname(os.path.abspath(__file__))

M2F_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
)

grader_image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.10")
    .apt_install("git", "wget", "libglib2.0-0", "libgl1")
    .run_commands("printf 'torch==1.13.1+cu117\\ntorchvision==0.14.1+cu117\\nnumpy==1.26.4\\n' > /constraints.txt")
    .env({"PIP_CONSTRAINT": "/constraints.txt"})
    .pip_install(
        "torch==1.13.1+cu117", "torchvision==0.14.1+cu117", extra_index_url="https://download.pytorch.org/whl/cu117"
    )
    .pip_install("mmcv-full==1.7.2", find_links="https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html")
    .pip_install("open-clip-torch==2.20.0", "clip-benchmark==1.4.0", "pandas", "opencv-python-headless==4.8.1.78")
    .run_commands(
        "git clone --depth 1 --branch v2.28.2 https://github.com/open-mmlab/mmdetection.git /opt/mmdetection",
        "pip install wheel setuptools==69.5.1 && pip install --no-build-isolation -e /opt/mmdetection",
        f"mkdir -p /models && wget -q {M2F_URL} -O /models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
        "python -c \"import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')\"",
    )
    .add_local_file(f"{HERE}/data/official/object_names.txt", "/opt/object_names.txt", copy=True)
    .add_local_python_source("geneval_core")
)

gen_image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.35.2",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "peft==0.17.1",
        "sentencepiece",
        "protobuf",
        "pillow",
        "safetensors",
        "hf_transfer",
        "open-clip-torch==3.2.0",
        "spacy==3.8.7",
        "numpy<2.3",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .pip_install("scipy")
    .pip_install("pyarrow")
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("indep_checker")
)


def _save_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


@app.cls(image=grader_image, gpu="A10G", volumes={VOL: vol}, timeout=3600, scaledown_window=90, max_containers=4)
class Grader:
    @modal.enter()
    def load(self):
        from geneval_core import Checker, default_paths

        self.checker = Checker(*default_paths())

    @modal.method()
    def grade(self, items):
        """items: [{"path": relative path in the volume, "meta": GenEval metadata}] -> one result per item."""
        vol.reload()
        out = []
        t0 = time.time()
        for it in items:
            p = os.path.join(VOL, it["path"])
            if not os.path.exists(p):
                out.append({"path": it["path"], "error": "missing"})
                continue
            r = self.checker.evaluate_image(p, it["meta"])
            r["path"] = it["path"]
            out.append(r)
        print(f"graded {len(items)} in {time.time() - t0:.1f}s")
        return out


@app.cls(
    image=gen_image,
    gpu="H100",
    volumes={VOL: vol, "/cache": cache},
    secrets=[hf_secret],
    timeout=3600,
    scaledown_window=60,
    max_containers=4,
)
class Gen:
    model_id: str = modal.parameter(default="black-forest-labs/FLUX.1-schnell")

    @modal.enter()
    def load(self):
        import torch
        from diffusers import DiffusionPipeline

        self.pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        cache.commit()

    @modal.method()
    def generate(self, jobs, steps=4, guidance=0.0, size=1024, batch=8, max_seq=256, lora=None):
        """jobs: [{"prompt", "seed", "path"}]. Writes PNGs into the volume; skips paths that exist."""
        import torch

        vol.reload()
        if lora:
            self.pipe.load_lora_weights(os.path.join(VOL, lora))
        todo = [j for j in jobs if not os.path.exists(os.path.join(VOL, j["path"]))]
        t0 = time.time()
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            gens = [torch.Generator("cuda").manual_seed(int(j["seed"])) for j in chunk]
            kw = {
                "prompt": [j["prompt"] for j in chunk],
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "height": size,
                "width": size,
                "generator": gens,
            }
            if "flux" in self.model_id.lower():
                kw["max_sequence_length"] = max_seq
            images = self.pipe(**kw).images
            for j, img in zip(chunk, images, strict=False):
                _save_png(img, os.path.join(VOL, j["path"]))
        vol.commit()
        if lora:
            self.pipe.unload_lora_weights()
        secs = time.time() - t0
        print(f"generated {len(todo)} in {secs:.1f}s")
        return {"n": len(todo), "seconds": secs}


@app.cls(
    image=gen_image,
    gpu="A10G",
    volumes={VOL: vol, "/cache": cache},
    secrets=[hf_secret],
    timeout=3600,
    scaledown_window=90,
    max_containers=4,
)
class Scorer:
    @modal.enter()
    def load(self):
        self._pick = self._owl = self._blip = None

    def _load_pick(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        if self._pick is None:
            proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to("cuda", torch.float16)
            self._pick = (proc, model)
        return self._pick

    @modal.method()
    def pickscore(self, items):
        """items: [{"path", "prompt"}] -> PickScore (CLIP-H logit scale x cosine), one per item."""
        import torch
        from PIL import Image

        vol.reload()
        proc, model = self._load_pick()
        out = []
        for i in range(0, len(items), 32):
            chunk = items[i : i + 32]
            imgs = [Image.open(os.path.join(VOL, it["path"])).convert("RGB") for it in chunk]
            with torch.no_grad():
                pi = proc(images=imgs, return_tensors="pt").to("cuda")
                pt = proc(
                    text=[it["prompt"] for it in chunk],
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to("cuda")
                ie = model.get_image_features(pixel_values=pi["pixel_values"].half())
                te = model.get_text_features(**pt)
                ie = ie / ie.norm(dim=-1, keepdim=True)
                te = te / te.norm(dim=-1, keepdim=True)
                s = model.logit_scale.exp() * (ie * te).sum(-1)
            out.extend(float(x) for x in s.float().cpu())
        cache.commit()
        return out

    @modal.method()
    def indep_check(self, items):
        """items: [{"path", "meta"}] -> the independent checker's verdict (OWLv2 + CLIP ViT-H colour)."""
        from indep_checker import IndepChecker

        vol.reload()
        if self._owl is None:
            self._owl = IndepChecker()
            cache.commit()
        return [self._owl.check(os.path.join(VOL, it["path"]), it["meta"]) for it in items]

    @modal.method()
    def bvqa(self, items):
        """items: [{"path", "prompt"}] -> T2I-CompBench B-VQA: product over noun phrases of P(yes | "<phrase>?")."""
        import spacy
        import torch
        from PIL import Image
        from transformers import BlipForQuestionAnswering, BlipProcessor

        vol.reload()
        if self._blip is None:
            proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").eval().to("cuda")
            self._blip = (proc, model, spacy.load("en_core_web_sm"))
            cache.commit()
        proc, model, nlp = self._blip
        yes_id = proc.tokenizer("yes", add_special_tokens=False).input_ids[0]
        no_id = proc.tokenizer("no", add_special_tokens=False).input_ids[0]
        bos = model.config.text_config.bos_token_id
        out = []
        for it in items:
            img = Image.open(os.path.join(VOL, it["path"])).convert("RGB")
            phrases = [c.text for c in nlp(it["prompt"]).noun_chunks]
            score = 1.0
            for ph in phrases:
                enc = proc(images=img, text=f"{ph}?", return_tensors="pt").to("cuda")
                with torch.no_grad():
                    vis = model.vision_model(pixel_values=enc["pixel_values"])[0]
                    q = model.text_encoder(
                        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], encoder_hidden_states=vis
                    )[0]
                    dec = model.text_decoder(
                        input_ids=torch.tensor([[bos]], device="cuda"), encoder_hidden_states=q
                    ).logits[0, -1]
                    p = torch.softmax(dec[[yes_id, no_id]].float(), dim=0)[0].item()
                score *= p
            out.append({"bvqa": score, "phrases": phrases})
        return out


VQA_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VQA_QUESTION = 'Does this figure show "{}"? Please answer yes or no.'


@app.cls(
    image=gen_image,
    gpu="L40S",
    volumes={VOL: vol, "/cache": cache},
    secrets=[hf_secret],
    timeout=3600,
    scaledown_window=90,
    max_containers=4,
)
class VQA:
    """VQAScore (Lin et al. 2024): P("Yes") for the question 'Does this figure show "<prompt>"?', here with
    Qwen2.5-VL-7B as the answering model. A detector-free guard: no object detector is involved."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.proc = AutoProcessor.from_pretrained(VQA_MODEL, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
        self.proc.tokenizer.padding_side = "left"
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(VQA_MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
        )
        self.yes = self.proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
        cache.commit()

    @modal.method()
    def score(self, items):
        """items: [{"path", "prompt"}] -> VQAScore in [0, 1], one per item."""
        import torch
        from PIL import Image

        vol.reload()
        out = []
        for i in range(0, len(items), 8):
            chunk = items[i : i + 8]
            msgs = [
                [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": VQA_QUESTION.format(it["prompt"])}],
                    }
                ]
                for it in chunk
            ]
            texts = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            imgs = [Image.open(os.path.join(VOL, it["path"])).convert("RGB") for it in chunk]
            enc = self.proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1].float()
            out.extend(float(x) for x in torch.softmax(logits, -1)[:, self.yes].cpu())
        return out


@app.function(image=gen_image, volumes={VOL: vol, "/cache": cache}, secrets=[hf_secret], timeout=3600, memory=16384)
def genai_prepare(image_shards: int = 1):
    """GenAI-Bench (Apache-2.0): every prompt with its tags and human ratings; images of the six rated models for
    the first `image_shards` parquet shards are written to the volume (genai_human/<idx>/<model>.png) for the
    judge check."""
    import io as _io

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    models = ["DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo"]
    rows = []
    for s in range(12):
        f = hf_hub_download("BaiqiL/GenAI-Bench", f"data/train-{s:05d}-of-00012.parquet", repo_type="dataset")
        cols = ["Index", "Prompt", "Tags", "HumanRatings"] + (models if s < image_shards else [])
        t = pq.read_table(f, columns=cols).to_pylist()
        for r in t:
            row = {"idx": int(r["Index"]), "prompt": r["Prompt"], "tags": r["Tags"], "ratings": r["HumanRatings"]}
            if s < image_shards:
                row["images"] = {}
                for m in models:
                    rel = f"genai_human/{row['idx']:05d}/{m}.png"
                    img = Image.open(_io.BytesIO(r[m]["bytes"])).convert("RGB")
                    _save_png(img, os.path.join(VOL, rel))
                    row["images"][m] = rel
            rows.append(row)
    vol.commit()
    cache.commit()
    return rows


@app.function(image=grader_image, gpu="A10G", volumes={VOL: vol}, timeout=900)
def smoke_grader():
    """Grade a blank image against "a photo of a dog": it must fail."""
    from PIL import Image

    from geneval_core import Checker, default_paths

    c = Checker(*default_paths())
    os.makedirs(f"{VOL}/smoke", exist_ok=True)
    Image.new("RGB", (512, 512), "white").save(f"{VOL}/smoke/blank.png")
    meta = {"tag": "single_object", "include": [{"class": "dog", "count": 1}], "prompt": "a photo of a dog"}
    return c.evaluate_image(f"{VOL}/smoke/blank.png", meta)


@app.function(image=gen_image, volumes={VOL: vol}, timeout=1200)
def fetch(paths, size=512):
    """Return JPEG bytes for a few volume paths (for figures and the blind look), downsized to `size` px."""
    from PIL import Image

    vol.reload()
    out = {}
    for p in paths:
        img = Image.open(os.path.join(VOL, p)).convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        out[p] = buf.getvalue()
    return out


@app.local_entrypoint()
def smoke():
    print(smoke_grader.remote())
