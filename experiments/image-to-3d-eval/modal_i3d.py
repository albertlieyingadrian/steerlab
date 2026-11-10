"""Generate 3D objects with five open image-to-3D models on one rented GPU class, then render and score them all.

    modal deploy experiments/image-to-3d-eval/modal_i3d.py
    python experiments/image-to-3d-eval/1_generate.py      # calls the functions below

Every model runs on an NVIDIA L40S (48 GB) with its released default settings. Seconds per object run from the input
image to the written 3D file, with the model already loaded and after one warm-up object; peak memory is PyTorch's
`max_memory_reserved` over the same span (what the process held, the closer match to what a card must have free).

Models and pinned code:
- TRELLIS (microsoft/TRELLIS-image-large, MIT): one run gives Gaussian splats and a textured mesh (GLB); both are
  scored, as "TRELLIS mesh" and "TRELLIS splats".
- TripoSR (stabilityai/TripoSR, MIT): mesh with vertex colours, marching cubes at 256.
- Stable Fast 3D (stabilityai/stable-fast-3d, Stability AI Community licence, gated): textured mesh.
- Idol (speridlabs/idol, Apache-2.0): Gaussian splats, 262,144 per object (its default).
- TRELLIS.2 (microsoft/TRELLIS.2-4B, MIT; its encoders DINOv3 and RMBG-2.0 are gated): textured mesh (GLB), added
  2026-11-10 after the RMBG-2.0 licence was accepted.

Each model is given the same 512x512 RGBA render of a Google Scanned Objects mesh (the alpha channel is the object's
mask, so no model needs its background remover) and crops it with its own default framing.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import modal

app = modal.App("image-to-3d-eval")
volume = modal.Volume.from_name("image-to-3d-eval", create_if_missing=True)
GPU = "L40S"
TRELLIS_SHA = "193e0502b5cb3eb5997ec457d5340e95d09b158e"
TRIPOSR_SHA = "107cefdc244c39106fa830359024f6a2f1c78871"
SF3D_SHA = "ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2"
IDOL_SHA = "8db700349a5570aac390e4780fa32e3eed6b587a"
ARCH = "8.6;8.9;9.0"
# Modal's standalone Python records clang as its compiler; the CUDA extensions are built with gcc instead.
BUILD_ENV = {"TORCH_CUDA_ARCH_LIST": ARCH, "FORCE_CUDA": "1", "CC": "gcc", "CXX": "g++"}
ENV = {"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "TORCH_EXTENSIONS_DIR": "/cache/torch_ext"}

# One image for TRELLIS, TripoSR and the renderer: CUDA 12.1, PyTorch 2.4, TRELLIS's compiled extensions.
main_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "libegl1", "libgomp1", "ffmpeg")
    .env(BUILD_ENV)
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("xformers==0.0.27.post2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy<2",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "tqdm",
        "easydict",
        "opencv-python-headless",
        "scipy",
        "ninja",
        "rembg",
        "onnxruntime",
        "open3d",
        "trimesh==4.4.9",
        "xatlas",
        "pyvista",
        "pymeshfix",
        "igraph",
        "transformers==4.46.3",
        "plyfile",
        "lpips",
        "open_clip_torch==2.29.0",
        "scikit-image",
        "omegaconf",
        "einops",
        "huggingface_hub",
        "safetensors",
        "spconv-cu120",
        "setuptools<75",
        "wheel",
        "scikit-build-core",
        "cmake",
        "pybind11",
    )
    .pip_install("git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8")
    .pip_install("kaolin==0.17.0", find_links="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html")
    .pip_install("gsplat==1.5.3", extra_index_url="https://docs.gsplat.studio/whl/pt24cu121")
    .run_commands(
        "git clone https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast && cd /opt/nvdiffrast"
        " && git checkout v0.3.3 && pip install . --no-build-isolation",
        "git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /opt/diffoctreerast"
        " && pip install /opt/diffoctreerast --no-build-isolation",
        "git clone https://github.com/autonomousvision/mip-splatting.git /opt/mip-splatting"
        " && pip install /opt/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation",
        "pip install git+https://github.com/tatsy/torchmcubes.git --no-build-isolation",
        "git clone https://github.com/microsoft/TRELLIS.git /opt/TRELLIS"
        f" && cd /opt/TRELLIS && git checkout {TRELLIS_SHA}",
        f"git clone https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR && cd /opt/TripoSR"
        f" && git checkout {TRIPOSR_SHA}",
    )
    .run_commands("cd /opt/TRELLIS && git submodule update --init --recursive")  # FlexiCubes, for the mesh decoder
    .env({**ENV, "PYTHONPATH": "/opt/TRELLIS:/opt/TripoSR:/root", "ATTN_BACKEND": "xformers", "SPCONV_ALGO": "native"})
    .add_local_file(__file__.replace("modal_i3d.py", "eval3d.py"), "/root/eval3d.py")
)

sf3d_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "libgomp1")
    .env(BUILD_ENV)
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("setuptools<75", "wheel", "ninja")
    # Build the texture baker with CUDA (no GPU is visible at build time) and for a generic CPU.
    .env({"USE_CUDA": "1", "USE_NATIVE_ARCH": "0"})
    .run_commands(
        "git clone https://github.com/Stability-AI/stable-fast-3d.git /opt/sf3d"
        f" && cd /opt/sf3d && git checkout {SF3D_SHA}"
        " && pip install --no-build-isolation -r requirements.txt"
    )
    .env({**ENV, "PYTHONPATH": "/opt/sf3d"})
)

idol_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    # Install with torch held at 2.8 (CUDA 12.8), so the resolver does not pull a CUDA 13 build.
    .run_commands(
        f"git clone https://github.com/speridlabs/idol-sdk.git /opt/idol && cd /opt/idol && git checkout {IDOL_SHA}"
        " && pip install -e . torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .env(ENV)
)

# TRELLIS.2 as its setup.sh installs it: PyTorch 2.6 on CUDA 12.4, flash-attn, and Jeffrey Xiang's mesh extensions.
TRELLIS2_SHA = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
CUMESH_SHA = "12289e1062f0603f2f0d0771b02e1395d247f26f"
FLEXGEMM_SHA = "6dd94a859c26ee8246888502eada3dd8ad85532e"
NVDIFFREC_SHA = "b296927cc7fd01c2ac1087c8065c4d7248f72da4"
FLASH_ATTN = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/"
    "flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)
trellis2_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "libegl1", "libgomp1", "libjpeg-dev"
    )
    .env(BUILD_ENV)
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "numpy<2",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "tqdm",
        "easydict",
        "opencv-python-headless",
        "ninja",
        "trimesh",
        "transformers>=4.56,<5",
        "pandas",
        "lpips",
        "zstandard",
        "kornia",
        "timm",
        "huggingface_hub",
        "safetensors",
        "setuptools<75",
        "wheel",
        FLASH_ATTN,
    )
    .pip_install("git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8")
    .run_commands(
        "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /opt/nvdiffrast"
        " && pip install /opt/nvdiffrast --no-build-isolation",
        "git clone https://github.com/JeffreyXiang/nvdiffrec.git /opt/nvdiffrec"
        f" && cd /opt/nvdiffrec && git checkout {NVDIFFREC_SHA} && pip install . --no-build-isolation",
        "git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /opt/CuMesh"
        f" && cd /opt/CuMesh && git checkout {CUMESH_SHA} && git submodule update --init --recursive"
        " && pip install . --no-build-isolation",
        "git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /opt/FlexGEMM"
        f" && cd /opt/FlexGEMM && git checkout {FLEXGEMM_SHA} && git submodule update --init --recursive"
        " && pip install . --no-build-isolation",
        "git clone --recursive https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS2"
        f" && cd /opt/TRELLIS2 && git checkout {TRELLIS2_SHA} && git submodule update --init --recursive"
        " && pip install ./o-voxel --no-build-isolation",
    )
    .env({**ENV, "PYTHONPATH": "/opt/TRELLIS2", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

HF_SECRET = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])


def _rgba(png: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(png)).convert("RGBA")


class _Timed:
    """Seconds and peak memory around one generation."""

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


def _run_all(gen, name: str, objects: list[str]) -> list[dict]:
    """Run gen(png) -> {suffix: bytes} for every object (the first twice, as warm-up), saving to the volume."""
    import os

    import torch

    rows = []
    for i, obj in enumerate([objects[0], *objects]):
        png = Path(f"/cache/inputs/{obj}.png").read_bytes()
        try:
            with _Timed(torch) as t:
                files = gen(png)
        except Exception as err:  # noqa: BLE001, RUF100 - a failed object is a result, not a crash
            rows.append({"model": name, "object": obj, "error": repr(err)[:300]})
            print(rows[-1])
            continue
        if i == 0:
            print(f"{name}: warm-up {t.seconds:.1f} s")
            continue
        os.makedirs(f"/cache/out/{name}", exist_ok=True)
        extra = {k.lstrip("_"): files.pop(k) for k in list(files) if k.startswith("_")}  # e.g. a split of the time
        for suffix, data in files.items():
            with open(f"/cache/out/{name}/{obj}{suffix}", "wb") as handle:
                handle.write(data)
        row = {"model": name, "object": obj, "seconds": t.seconds, "peak_reserved_gb": t.peak_reserved_gb,
               "peak_allocated_gb": t.peak_allocated_gb, "gpu": torch.cuda.get_device_name(), **extra}  # fmt: skip
        rows.append(row)
        print(json.dumps(row))
    volume.commit()
    return rows


@app.function(image=main_image, gpu=GPU, volumes={"/cache": volume}, timeout=3 * 3600)
def gen_trellis(objects: list[str]) -> list[dict]:
    import tempfile

    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()

    def gen(png: bytes) -> dict[str, bytes]:
        out = pipe.run(Image.open(io.BytesIO(png)), seed=1, formats=["gaussian", "mesh"])
        glb = postprocessing_utils.to_glb(out["gaussian"][0], out["mesh"][0], simplify=0.95, texture_size=1024)
        with tempfile.TemporaryDirectory() as tmp:
            glb.export(f"{tmp}/a.glb")
            out["gaussian"][0].save_ply(f"{tmp}/a.ply")
            return {".glb": open(f"{tmp}/a.glb", "rb").read(), ".ply": open(f"{tmp}/a.ply", "rb").read()}

    return _run_all(gen, "trellis", objects)


@app.function(image=main_image, gpu=GPU, volumes={"/cache": volume}, timeout=3 * 3600)
def gen_triposr(objects: list[str]) -> list[dict]:
    import numpy as np
    import torch
    from PIL import Image
    from tsr.system import TSR
    from tsr.utils import resize_foreground

    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to("cuda")

    def gen(png: bytes) -> dict[str, bytes]:
        # run.py's preprocessing after background removal: frame at 0.85, composite on grey.
        image = np.asarray(resize_foreground(_rgba(png), 0.85), np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        with torch.no_grad():
            codes = model([Image.fromarray((image * 255).astype(np.uint8))], device="cuda")
        mesh = model.extract_mesh(codes, True, resolution=256)[0]
        return {".glb": mesh.export(file_type="glb")}

    return _run_all(gen, "triposr", objects)


@app.function(image=sf3d_image, gpu=GPU, volumes={"/cache": volume}, secrets=[HF_SECRET], timeout=3 * 3600)
def gen_sf3d(objects: list[str]) -> list[dict]:
    import tempfile

    import torch
    from sf3d.system import SF3D
    from sf3d.utils import resize_foreground

    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d", config_name="config.yaml", weight_name="model.safetensors"
    )
    model.to("cuda").eval()

    def gen(png: bytes) -> dict[str, bytes]:
        image = resize_foreground(_rgba(png), 0.85)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh, _ = model.run_image([image], bake_resolution=1024, remesh="none", vertex_count=-1)
        with tempfile.TemporaryDirectory() as tmp:
            mesh.export(f"{tmp}/a.glb", include_normals=True)
            return {".glb": open(f"{tmp}/a.glb", "rb").read()}

    return _run_all(gen, "sf3d", objects)


@app.function(image=trellis2_image, gpu=GPU, cpu=8.0, volumes={"/cache": volume}, secrets=[HF_SECRET], timeout=6 * 3600)
def gen_trellis2(objects: list[str]) -> list[dict]:
    """TRELLIS.2-4B with its released defaults: the 1024 cascade, then a GLB with the export settings of its own demo
    (app.py: 500,000-face target, 2,048-pixel texture, remeshed), as every model here uses its own defaults. PNG
    textures instead of WebP, which trimesh cannot read. The export (remeshing, UV unwrapping, baking) runs mostly on
    the CPU, so this job gets 8 CPU cores (the others ran with Modal's default share); its time is also split into
    generation and export."""
    import tempfile

    import o_voxel
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()

    def gen(png: bytes) -> dict:
        t0 = time.perf_counter()
        mesh = pipe.run(Image.open(io.BytesIO(png)))[0]
        mesh.simplify(16777216)  # nvdiffrast's limit, as in example.py
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=100000, texture_size=2048, remesh=True, remesh_band=1, remesh_project=0,
        )  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            glb.export(f"{tmp}/a.glb")
            data = Path(f"{tmp}/a.glb").read_bytes()
        return {".glb": data, "_seconds_generate": t1 - t0, "_seconds_export": time.perf_counter() - t1}

    return _run_all(gen, "trellis2", objects)


@app.function(image=idol_image, gpu=GPU, volumes={"/cache": volume}, secrets=[HF_SECRET], timeout=3 * 3600)
def gen_idol(objects: list[str]) -> list[dict]:
    import tempfile

    from idol import IdolPipeline

    pipe = IdolPipeline.from_pretrained("speridlabs/idol")

    def gen(png: bytes) -> dict[str, bytes]:
        with tempfile.TemporaryDirectory() as tmp:
            Path(f"{tmp}/in.png").write_bytes(png)
            out = pipe(f"{tmp}/in.png", mode="synthetic", seed=42)
            out.gaussian.save_ply(f"{tmp}/a.ply")
            return {".ply": open(f"{tmp}/a.ply", "rb").read()}

    return _run_all(gen, "idol", objects)


# ------------------------------------------------------------------------------------------ ground truth, scoring

FUEL = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/{}.zip"


@app.function(image=main_image, gpu=GPU, volumes={"/cache": volume}, timeout=3600)
def prepare(objects: list[str]) -> list[dict]:
    """Download each GSO object, render the 512x512 input image (view 0) and note its size."""
    import os
    import shutil
    import urllib.request
    import zipfile

    import numpy as np
    from PIL import Image

    import eval3d

    renderer = eval3d.Renderer()
    rows = []
    os.makedirs("/cache/inputs", exist_ok=True)
    for obj in objects:
        folder = f"/cache/gso/{obj}"
        if not os.path.exists(f"{folder}/meshes/model.obj"):
            data = urllib.request.urlopen(FUEL.format(obj), timeout=120).read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(folder)
        # The MTL names texture.png beside the mesh; GSO ships it in materials/textures/.
        if not os.path.exists(f"{folder}/meshes/texture.png"):
            shutil.copy(f"{folder}/materials/textures/texture.png", f"{folder}/meshes/texture.png")
        gt, _ = eval3d.normalise(eval3d.load_asset(f"{folder}/meshes/model.obj"))
        lo, hi = gt.bounds()
        view = renderer.render(gt, [eval3d.camera(eval3d.INPUT_AZIM_DEG)], eval3d.RES)
        rgba = np.concatenate([view["rgb"][0], view["alpha"][0][..., None]], -1)
        Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(f"/cache/inputs/{obj}.png")
        rows.append({"object": obj, "extent": (hi - lo).round(4).tolist(), "faces": sum(len(p.f) for p in gt.parts)})
        print(json.dumps(rows[-1]))
    volume.commit()
    return rows


def _corrupt(gt, kind: str, seed: int = 0):
    """Known-worse copies of the ground truth, used to check the metrics and the judges."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    import eval3d

    rng = np.random.default_rng(seed)
    parts = []
    for p in gt.parts:
        v, tex, vcol = p.v.copy(), p.tex, p.vcol
        if kind == "bumpy":  # every vertex pushed by noise with a spread of 1.2% of the object's size
            v = v + rng.normal(0, 0.012, v.shape).astype(np.float32)
        elif kind == "squashed":  # 40% shorter
            v[:, 2] *= 0.6
        elif kind == "blurred":  # texture blurred and washed out
            if tex is not None:
                tex = gaussian_filter(tex, sigma=(12, 12, 0)) * 0.6 + 0.4 * tex.mean((0, 1))
            if vcol is not None:
                vcol = vcol * 0.6 + 0.4 * vcol.mean(0)
        parts.append(eval3d.Part(v, p.f, p.uv, tex, vcol))
    return eval3d.Mesh(parts)


SYSTEMS = {
    "trellis_mesh": ("trellis", ".glb"),
    "trellis_splats": ("trellis", ".ply"),
    "triposr": ("triposr", ".glb"),
    "sf3d": ("sf3d", ".glb"),
    "idol": ("idol", ".ply"),
    "trellis2": ("trellis2", ".glb"),
    "gt_bumpy": (None, "bumpy"),
    "gt_squashed": (None, "squashed"),
    "gt_blurred": (None, "blurred"),
}


@app.cls(image=main_image, gpu=GPU, volumes={"/cache": volume}, timeout=3 * 3600, scaledown_window=60)
class Scorer:
    @modal.enter()
    def load(self):
        import eval3d

        self.renderer, self.metrics = eval3d.Renderer(), eval3d.ImageMetrics()

    @modal.method()
    def score(self, obj: str, systems: list[str]) -> list[dict]:
        """Scores, an 8-view sheet (for the judge) and a 36-frame turntable (for the gallery) per system."""
        import os

        import imageio
        import numpy as np
        from PIL import Image

        import eval3d

        gt, _ = eval3d.normalise(eval3d.load_asset(f"/cache/gso/{obj}/meshes/model.obj"))
        rgba = np.asarray(Image.open(f"/cache/inputs/{obj}.png"), np.float32) / 255
        input_rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        gt_views = eval3d.turntable(self.renderer, gt)
        rows = []

        def save(system: str, asset, views) -> None:
            folder = f"/cache/renders/{system}"
            os.makedirs(folder, exist_ok=True)
            # The judge's sheet: colour views 0, 2, 4, 6 above their normal maps, 256 px each.
            tiles = [views[k][i] for k in ("rgb", "normal") for i in (0, 2, 4, 6)]
            tiles = [Image.fromarray((t * 255).astype(np.uint8)).resize((256, 256), Image.LANCZOS) for t in tiles]
            tiles = [np.asarray(t) for t in tiles]
            sheet = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
            Image.fromarray(sheet).save(f"{folder}/{obj}.png")
            cams = [eval3d.camera(eval3d.INPUT_AZIM_DEG + 10 * k, res=256) for k in range(36)]
            frames = (self.renderer.render(asset, cams, 256)["rgb"] * 255).astype(np.uint8)
            imageio.mimwrite(f"{folder}/{obj}.mp4", list(frames), fps=12, codec="libx264", quality=7)

        save("gt", gt, gt_views)
        for system in systems:
            source, kind = SYSTEMS[system]
            t0 = time.time()
            if source is None:
                pred = _corrupt(gt, kind)
            else:
                path = f"/cache/out/{source}/{obj}{kind}"
                if not os.path.exists(path):
                    rows.append({"object": obj, "system": system, "error": "missing"})
                    continue
                pred = eval3d.load_asset(path)
            try:
                result = eval3d.evaluate(self.renderer, self.metrics, pred, gt, input_rgb, gt_views)
            except Exception as err:  # noqa: BLE001, RUF100 - a broken output is a result, not a crash
                rows.append({"object": obj, "system": system, "error": repr(err)[:300]})
                print(rows[-1])
                continue
            save(system, result["pred_asset"], result["pred_views"])
            rows.append({"object": obj, "system": system, **result["scores"], "score_seconds": time.time() - t0})
            print(json.dumps(rows[-1]))
        # The ground truth's own CLIP similarity to its input: the ceiling for that score.
        e_in = self.metrics.clip_embed([input_rgb])[0]
        clip_gt = float((self.metrics.clip_embed(list(gt_views["rgb"])) @ e_in).mean())
        rows.append({"object": obj, "system": "gt", "clip": clip_gt})
        volume.commit()
        return rows


@app.local_entrypoint()
def smoke():
    obj = "Weisshai_Great_White_Shark"
    print(prepare.remote([obj]))
    print(gen_triposr.remote([obj]))
    print(Scorer().score.remote(obj, ["triposr", "gt_bumpy", "gt_squashed", "gt_blurred"]))
