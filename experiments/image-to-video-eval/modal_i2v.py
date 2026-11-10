"""Score videos with VBench++'s own code, and generate videos with four open image-to-video models, on Modal.

    modal deploy experiments/image-to-video-eval/modal_i2v.py
    python experiments/image-to-video-eval/4_scores.py      # calls Scorer
    python experiments/image-to-video-eval/3_generate.py    # calls gen_*

Scoring (one NVIDIA L4, or an A10G when no L4 is free, per container) calls the functions of the `vbench` package
(0.1.5, VBench++) on each clip as re-encoded by 1_human_pairs.py (480 px short side, frame rate and length unchanged):

- subject consistency (DINO ViT-B/16) and background consistency (CLIP ViT-B/32): how alike each frame is to the
  first and to the previous one;
- motion smoothness (AMT-S): how well a frame-interpolation model predicts each dropped frame from its neighbours;
- temporal flickering: how little consecutive frames differ pixel by pixel;
- dynamic degree (RAFT): VBench's yes/no "does it move"; its mean optical-flow size (the top 5% of each frame pair's
  flow, at about 8 frames a second) is kept too, as a graded version;
- aesthetic quality (LAION predictor on CLIP ViT-L/14) and imaging quality (MUSIQ), averaged over every frame;
- I2V subject (DINO) and I2V background (DreamSim): how alike the frames are to the input still;
- camera motion (CoTracker2 on a 10 x 10 grid of points, VBench++'s heuristic): which of pan left/right, tilt up/down,
  zoom in/out or static the clip shows. The graded version is the points' mean shift from the first to the last frame
  in the asked direction (radial shift for zooms), as a share of the short side.

Plus the scores the image-to-3D note used: CLIP (ViT-L/14, OpenAI) and DINOv2 (base) cosine between the input still and
8 evenly spaced frames, averaged.

Generation (one NVIDIA H100 80 GB per model, pinned so Modal does not move a job to an H200) uses each model's
released diffusers pipeline at its model card's default settings, seed 0. Seconds per clip run from the still to
decoded frames, with the model loaded and after one warm-up clip; peak memory is PyTorch's max_memory_reserved over
the same span.
"""

from __future__ import annotations

import json
import sys
import time
import types

import modal

app = modal.App("image-to-video-eval")
volume = modal.Volume.from_name("image-to-video-eval", create_if_missing=True)
ENV = {"HF_HOME": "/cache/hf", "TORCH_HOME": "/cache/torch", "VBENCH_CACHE_DIR": "/cache/vbench",
       "PYTHONUNBUFFERED": "1"}  # fmt: skip
HF_SECRET = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

score_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget", "unzip", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy<2",
        "pillow",
        "pandas",
        "scipy",
        "scikit-image",
        "opencv-python-headless",
        "decord==0.6.0",
        "timm==1.0.12",
        "openai-clip==1.0.1",
        "open_clip_torch==2.29.0",
        "transformers==4.37.2",
        "easydict",
        "omegaconf",
        "einops",
        "pyiqa==0.1.13",
        "dreamsim==0.2.1",
        "peft==0.10.0",
        "imageio",
        "imageio-ffmpeg",
        "tqdm",
    )
    .pip_install("vbench==0.1.5", extra_options="--no-deps")
    .env(ENV)
)

gen_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "diffusers==0.35.2",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "sentencepiece",
        "protobuf",
        "ftfy",
        "imageio",
        "imageio-ffmpeg",
        "pillow",
        "numpy<2.3",
        "av",
    )
    .env(ENV)
)


# ------------------------------------------------------------------------------------------------------- scoring


def _camera_predictor():
    """vbench2_beta_i2v.camera_motion imports a CoTracker visualiser that the wheel does not ship; it is only used to
    save debug videos, so stand in an empty module for it."""
    name = "vbench2_beta_i2v.third_party.cotracker.utils.visualizer"
    for part in ["vbench2_beta_i2v.third_party", "vbench2_beta_i2v.third_party.cotracker",
                 "vbench2_beta_i2v.third_party.cotracker.utils", name]:  # fmt: skip
        sys.modules.setdefault(part, types.ModuleType(part))
    sys.modules[name].Visualizer = None
    from vbench2_beta_i2v.camera_motion import CameraPredict

    return CameraPredict


def graded_camera(tracks, short_side: float, movement: str) -> float:
    """Mean shift of the tracked points from the first to the last frame, in the direction the asked move should push
    the picture, as a share of the short side. Panning left moves the picture right (+x); tilting up moves it down
    (+y); zooming in pushes points away from the centre (+radial)."""
    import numpy as np

    p0, p1 = tracks[0].astype(float), tracks[-1].astype(float)
    d = p1 - p0
    out = p0 - p0.mean(0)
    radial = (d * out).sum(1) / (np.linalg.norm(out, axis=1) + 1e-6)
    shift = {"pan_left": d[:, 0].mean(), "pan_right": -d[:, 0].mean(), "tilt_up": d[:, 1].mean(),
             "tilt_down": -d[:, 1].mean(), "zoom_in": radial.mean(), "zoom_out": -radial.mean()}  # fmt: skip
    return float(shift[movement] / short_side)


@app.cls(image=score_image, gpu=["L4", "A10G"], volumes={"/cache": volume}, timeout=3 * 3600, scaledown_window=60,
         max_containers=12)  # fmt: skip
class Scorer:
    @modal.enter()
    def load(self):
        import clip
        import open_clip
        import torch
        from dreamsim import dreamsim
        from easydict import EasyDict
        from pyiqa.archs.musiq_arch import MUSIQ
        from transformers import AutoImageProcessor, AutoModel
        from vbench.aesthetic_quality import get_aesthetic_model
        from vbench.dynamic_degree import DynamicDegree
        from vbench.motion_smoothness import MotionSmoothness
        from vbench.utils import CACHE_DIR, init_submodules

        dev = self.device = "cuda"
        self.torch = torch
        sub = init_submodules(["motion_smoothness", "dynamic_degree", "imaging_quality", "aesthetic_quality"])
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(dev).eval()
        self.clip_b, _ = clip.load("ViT-B/32", device=dev)
        self.clip_l, _ = clip.load("ViT-L/14", device=dev)
        self.aesthetic = get_aesthetic_model(f"{CACHE_DIR}/aesthetic_model/emb_reader").to(dev)
        self.musiq = MUSIQ(pretrained_model_path=sub["imaging_quality"]["model_path"]).to(dev)
        self.musiq.training = False
        self.smooth = MotionSmoothness(sub["motion_smoothness"]["config"], sub["motion_smoothness"]["ckpt"], dev)

        class Dynamic(DynamicDegree):
            def check_move(self, score_list):
                self.last_scores = score_list
                return super().check_move(score_list)

        args = EasyDict({"model": sub["dynamic_degree"]["model"], "small": False, "mixed_precision": False,
                         "alternate_corr": False})  # fmt: skip
        self.dynamic = Dynamic(args, dev)
        self.dreamsim, _ = dreamsim(pretrained=True, cache_dir="/cache/dreamsim", device=dev)
        self.camera = _camera_predictor()(dev, {"repo": "facebookresearch/co-tracker", "model": "cotracker2"})
        self.oclip, _, self.oclip_pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.oclip = self.oclip.to(dev).eval()
        self.dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dinov2 = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
        volume.commit()

    def _embed_sim(self, path: str, still: str) -> dict:
        import numpy as np
        from PIL import Image
        from vbench.utils import load_video

        torch = self.torch
        frames = load_video(path, return_tensor=False)
        idx = np.linspace(0, len(frames) - 1, 8).round().astype(int)
        ims = [Image.fromarray(frames[i]) for i in idx]
        ref = Image.open(still).convert("RGB")
        out = {}
        c = self.oclip.encode_image(torch.stack([self.oclip_pre(im) for im in [ref, *ims]]).to(self.device))
        d = self.dinov2(**self.dproc(images=[ref, *ims], return_tensors="pt").to(self.device))
        for name, e in (("clip", c.float()), ("dinov2", d.last_hidden_state[:, 0].float())):
            e = torch.nn.functional.normalize(e, dim=-1)
            out[name] = float((e[1:] @ e[0]).mean())
        return out

    def _camera(self, path: str, movement: str) -> dict:
        tracks = self.camera.infer(path)
        g = self.camera.grid_size
        classes = self.camera.camera_classify(tracks[0].reshape(g, g, 2), tracks[-1].reshape(g, g, 2))
        return {"camera_classes": "|".join(sorted(classes)), "camera_hit": float(movement in classes),
                "camera_shift": graded_camera(tracks, self.camera.scale, movement)}  # fmt: skip

    @modal.method()
    def score(self, items: list[dict]) -> list[dict]:
        """items: {"video", "path" (on the volume), "still" (on the volume), "dims": "full" or "camera",
        "movement" (optional)}. Returns one row of scores per item; a failed score leaves an error_<name> column."""
        from vbench.aesthetic_quality import laion_aesthetic
        from vbench.background_consistency import background_consistency
        from vbench.imaging_quality import technical_quality
        from vbench.subject_consistency import subject_consistency
        from vbench.temporal_flickering import temporal_flickering
        from vbench2_beta_i2v.i2v_background import i2v_background
        from vbench2_beta_i2v.i2v_subject import i2v_subject

        torch, dev = self.torch, self.device
        rows = []
        for it in items:
            p, s = it["path"], it["still"]
            row = {"video": it["video"]}
            t0 = time.perf_counter()
            jobs = {
                "clip_dinov2": lambda p=p, s=s: self._embed_sim(p, s),
                "dynamic": lambda p=p: {"dynamic_degree": float(self.dynamic.infer(p)),
                                        "flow": float(sum(self.dynamic.last_scores) / len(self.dynamic.last_scores))},
            }  # fmt: skip
            if it["dims"] == "full":
                jobs.update({
                    "subject_consistency": lambda p=p: subject_consistency(self.dino, [p], dev, False)[0],
                    "background_consistency": lambda p=p: background_consistency(self.clip_b, None, [p], dev, False)[0],
                    "motion_smoothness": lambda p=p: self.smooth.motion_score(p),
                    "temporal_flickering": lambda p=p: temporal_flickering([p])[0],
                    "aesthetic_quality": lambda p=p: laion_aesthetic(self.aesthetic, self.clip_l, [p], dev)[0],
                    "imaging_quality": lambda p=p: technical_quality(self.musiq, [p], dev)[0],
                    "i2v_subject": lambda p=p, s=s: i2v_subject(self.dino, [(s, p)], dev)[0],
                    "i2v_background": lambda p=p, s=s: i2v_background(self.dreamsim, [(s, p)], dev)[0],
                })  # fmt: skip
            if it.get("movement"):
                jobs["camera"] = lambda p=p, m=it["movement"]: self._camera(p, m)
            for name, fn in jobs.items():
                try:
                    with torch.no_grad():
                        v = fn()
                    row.update(v if isinstance(v, dict) else {name: float(v)})
                except Exception as err:  # noqa: BLE001, RUF100 - one failed score should not lose the rest
                    row[f"error_{name}"] = repr(err)[:200]
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------------------------------- generation


class _Timed:
    """Seconds and peak memory around one generation (from experiments/image-to-3d-eval/modal_i3d.py)."""

    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0
        self.peak_reserved_gb = self.torch.cuda.max_memory_reserved() / 2**30
        self.peak_allocated_gb = self.torch.cuda.max_memory_allocated() / 2**30


def _fit(image, width: int, height: int):
    """Centre-crop a still to the model's aspect ratio, then resize it to the model's resolution."""
    from PIL import Image

    w, h = image.size
    target = width / height
    if w / h > target:
        nw = round(h * target)
        image = image.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = round(w / target)
        image = image.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return image.resize((width, height), Image.LANCZOS)


def _run_all(gen, name: str, jobs: list[dict], fps: int) -> list[dict]:
    """Run gen(still, prompt) -> frames for every job (the first twice, as warm-up), saving mp4s to the volume."""
    import os

    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    rows = []
    os.makedirs(f"/cache/gen/{name}", exist_ok=True)
    # Resume: a clip already on the volume is not made again (a stopped container's call is rescheduled by Modal).
    jobs = [j for j in jobs if not os.path.exists(f"/cache/gen/{name}/{j['id']}.mp4")]
    if not jobs:
        return rows
    for i, job in enumerate([jobs[0], *jobs]):
        still = Image.open(f"/cache/stills/{job['still']}").convert("RGB")
        try:
            with _Timed(torch) as t:
                frames = gen(still, job["prompt"])
        except Exception as err:  # noqa: BLE001, RUF100 - a failed clip is a result, not a crash
            rows.append({"model": name, "id": job["id"], "error": repr(err)[:300]})
            print(rows[-1])
            continue
        if i == 0:
            print(f"{name}: warm-up {t.seconds:.1f} s")
            continue
        export_to_video(frames, f"/cache/gen/{name}/{job['id']}.mp4", fps=fps)
        row = {"model": name, "id": job["id"], "seconds": t.seconds, "peak_reserved_gb": t.peak_reserved_gb,
               "peak_allocated_gb": t.peak_allocated_gb, "frames": len(frames), "fps": fps,
               "width": frames[0].size[0], "height": frames[0].size[1],
               "gpu": torch.cuda.get_device_name()}  # fmt: skip
        rows.append(row)
        print(json.dumps(row))
        volume.commit()
    return rows


@app.function(image=gen_image, gpu="H100!", volumes={"/cache": volume}, secrets=[HF_SECRET], timeout=4 * 3600)
def gen_wan(jobs: list[dict]) -> list[dict]:
    """Wan 2.2 TI2V-5B (Apache-2.0): 1280x704 area at the still's aspect, 121 frames at 24 fps, 50 steps, CFG 5."""
    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

    def gen(still, prompt):
        aspect = still.height / still.width
        h = round(np.sqrt(704 * 1280 * aspect)) // mod * mod
        w = round(np.sqrt(704 * 1280 / aspect)) // mod * mod
        g = torch.Generator("cuda").manual_seed(0)
        return pipe(image=_fit(still, w, h), prompt=prompt, height=h, width=w, num_frames=121, num_inference_steps=50,
                    guidance_scale=5.0, generator=g, output_type="pil").frames[0]  # fmt: skip

    # Budget: realism stills only (the camera stills were cut after the timing run).
    return _run_all(gen, "wan22_ti2v_5b", [j for j in jobs if j["id"].startswith("r")], fps=24)


@app.function(image=gen_image, gpu="H100!", volumes={"/cache": volume}, secrets=[HF_SECRET], timeout=4 * 3600)
def gen_ltx(jobs: list[dict]) -> list[dict]:
    """LTX-Video 0.9.7 distilled 13B (LTX-Video Open Weights licence): 832x480 (480x832 for a portrait still),
    121 frames at 24 fps, the distilled eight-step schedule from its diffusers example, one stage (no latent
    upscaler)."""
    import torch
    from diffusers import LTXConditionPipeline
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

    pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-distilled", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    negative = "worst quality, inconsistent motion, blurry, jittery, distorted"

    def gen(still, prompt):
        w, h = (832, 480) if still.width >= still.height else (480, 832)
        cond = LTXVideoCondition(image=_fit(still, w, h), frame_index=0)
        g = torch.Generator("cuda").manual_seed(0)
        return pipe(conditions=[cond], prompt=prompt, negative_prompt=negative, width=w, height=h,
                    num_frames=121, timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03], decode_timestep=0.05,
                    decode_noise_scale=0.025, image_cond_noise_scale=0.0, guidance_scale=1.0, guidance_rescale=0.7,
                    generator=g, output_type="pil").frames[0]  # fmt: skip

    return _run_all(gen, "ltx_video_097_distilled", jobs, fps=24)


@app.function(image=gen_image, gpu="H100!", volumes={"/cache": volume}, secrets=[HF_SECRET], timeout=4 * 3600)
def gen_cogvideox(jobs: list[dict]) -> list[dict]:
    """CogVideoX-5B-I2V (CogVideoX licence): 720x480 (its only size; portrait stills are cropped), 49 frames at 8 fps,
    50 steps, CFG 6 with dynamic CFG."""
    import torch
    from diffusers import CogVideoXImageToVideoPipeline

    pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    def gen(still, prompt):
        g = torch.Generator("cuda").manual_seed(0)
        return pipe(image=_fit(still, 720, 480), prompt=prompt, num_frames=49, num_inference_steps=50,
                    guidance_scale=6.0, use_dynamic_cfg=True, generator=g, output_type="pil").frames[0]  # fmt: skip

    # Budget: realism stills only (the camera stills were cut after the timing run).
    return _run_all(gen, "cogvideox_5b_i2v", [j for j in jobs if j["id"].startswith("r")], fps=8)


@app.function(image=gen_image, gpu="H100!", volumes={"/cache": volume}, secrets=[HF_SECRET], timeout=4 * 3600)
def gen_svd(jobs: list[dict]) -> list[dict]:
    """Stable Video Diffusion XT 1.0 (Stability AI Community licence; 1.1 is gated): 1024x576 (576x1024 for a portrait
    still), 25 frames at 7 fps, 25 steps, motion bucket 127. It takes no text, so it ignores the prompt."""
    import torch
    from diffusers import StableVideoDiffusionPipeline

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    def gen(still, prompt):
        g = torch.Generator("cuda").manual_seed(0)
        w, h = (1024, 576) if still.width >= still.height else (576, 1024)
        return pipe(_fit(still, w, h), width=w, height=h, num_frames=25, decode_chunk_size=8, generator=g).frames[0]

    return _run_all(gen, "svd_xt", jobs, fps=7)
