"""eval3d: render any generated 3D object the same way, and score it against a ground-truth mesh.

    python eval3d.py --pred out.glb --gt model.obj --input input.png --out scores/   # needs an NVIDIA GPU

Meshes (.glb, .obj, .ply with faces) and Gaussian splats (.ply in the standard 3DGS layout) go through one path:

1. Load. A mesh keeps its texture or vertex colours; a splat keeps its centres, sizes, rotations, opacities and base
   colour (the view-dependent part of the colour is dropped, so splats and meshes are both shown with one colour per
   point of the surface).
2. Normalise. Centre on the bounding box and scale so its longest side is 1 (for splats, the box of the centres with
   opacity above 0.05, trimmed at the 0.5th and 99.5th percentiles, so stray floaters do not set the size).
3. Sample the visible surface. Depth maps from 24 cameras spread over a sphere are turned back into 3D points. The
   same rule serves meshes and splats, so neither gets credit for hidden inside parts.
4. Align to the ground truth. Each of the 24 axis-aligned starting rotations is refined by ICP (rotation, shift and
   one scale), and the start that ends with the lowest Chamfer distance wins. Models pick their own "up" and "front",
   so this is needed; it flatters every model the same way.
5. Score geometry: Chamfer distance (mean distance from each point to the nearest point of the other cloud, averaged
   over both directions, in units of the object's size) and F-score at 2% and 5% of the size.
6. Render 8 turntable views (colour on white, unlit, plus normals from the depth map) of the aligned prediction and of
   the ground truth from the same cameras, and score the 7 views the input image did not show: PSNR, SSIM, LPIPS and
   silhouette IoU. CLIP similarity compares every view with the input image.

Rendering uses nvdiffrast for meshes and gsplat for splats. Cameras follow the OpenCV convention (x right, y down,
z forward); the world is z-up, as Google Scanned Objects are.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RES = 512
FOV_DEG = 45.0
DIST = 2.4
ELEV_DEG = 20.0
INPUT_AZIM_DEG = 30.0
TURNTABLE = [INPUT_AZIM_DEG + 45.0 * k for k in range(8)]  # view 0 is the input view
SH_C0 = 0.28209479177387814


# ----------------------------------------------------------------------------------------------------- assets


@dataclass
class Part:
    v: np.ndarray  # (V, 3) float32
    f: np.ndarray  # (F, 3) int32
    uv: np.ndarray | None = None  # (V, 2), with a texture
    tex: np.ndarray | None = None  # (H, W, 3) float32 in [0, 1]
    vcol: np.ndarray | None = None  # (V, 3) float32 in [0, 1]


@dataclass
class Mesh:
    parts: list[Part] = field(default_factory=list)

    def points(self) -> np.ndarray:
        return np.concatenate([p.v for p in self.parts])

    def transformed(self, T: np.ndarray) -> Mesh:
        out = []
        for p in self.parts:
            v = (p.v @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
            out.append(Part(v, p.f, p.uv, p.tex, p.vcol))
        return Mesh(out)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = self.points()
        return pts.min(0), pts.max(0)


@dataclass
class Splats:
    means: np.ndarray  # (N, 3)
    quats: np.ndarray  # (N, 4) w, x, y, z
    scales: np.ndarray  # (N, 3), linear
    opac: np.ndarray  # (N,), in [0, 1]
    rgb: np.ndarray  # (N, 3), in [0, 1]

    def transformed(self, T: np.ndarray) -> Splats:
        A = T[:3, :3]
        s = float(np.cbrt(np.linalg.det(A)))
        R = A / s
        means = (self.means @ A.T + T[:3, 3]).astype(np.float32)
        quats = quat_mul(mat_to_quat(R)[None], self.quats).astype(np.float32)
        return Splats(means, quats, (self.scales * s).astype(np.float32), self.opac, self.rgb)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = self.means[self.opac > 0.05]
        if len(pts) < 100:
            pts = self.means
        return np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _image_array(img) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _colour(c) -> np.ndarray:
    c = np.asarray(c, np.float32)[:3]
    return c / 255.0 if c.max() > 1 else c


def load_mesh(path: str | Path) -> Mesh:
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for geom in scene.dump(concatenate=False):
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        v, f = np.asarray(geom.vertices, np.float32), np.asarray(geom.faces, np.int32)
        vis = geom.visual
        uv = tex = vcol = None
        if isinstance(vis, trimesh.visual.TextureVisuals):
            mat = vis.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            factor = getattr(mat, "baseColorFactor", None)
            if img is not None and vis.uv is not None and len(vis.uv) == len(v):
                uv, tex = np.asarray(vis.uv, np.float32), _image_array(img)
                if factor is not None:
                    tex = tex * _colour(factor)
            else:
                c = factor if factor is not None else getattr(mat, "diffuse", None)
                if c is not None:
                    vcol = np.tile(_colour(c), (len(v), 1))
        elif getattr(vis, "kind", None) in ("vertex", "face"):
            vcol = np.asarray(vis.vertex_colors[:, :3], np.float32) / 255.0
        if uv is None and vcol is None:
            vcol = np.full((len(v), 3), 0.7, np.float32)
        parts.append(Part(v, f, uv, tex, vcol))
    if not parts:
        raise ValueError(f"{path}: no triangles")
    return Mesh(parts)


def load_asset(path: str | Path) -> Mesh | Splats:
    path = Path(path)
    if path.suffix.lower() == ".ply":
        from plyfile import PlyData

        ply = PlyData.read(str(path))
        if "f_dc_0" in ply["vertex"].data.dtype.names:
            d = ply["vertex"].data
            means = np.stack([d["x"], d["y"], d["z"]], -1).astype(np.float32)
            dc = np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], -1)
            rgb = np.clip(0.5 + SH_C0 * dc, 0, 1).astype(np.float32)
            opac = (1 / (1 + np.exp(-d["opacity"]))).astype(np.float32)
            scales = np.exp(np.stack([d[f"scale_{i}"] for i in range(3)], -1)).astype(np.float32)
            q = np.stack([d[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
            q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
            return Splats(means, q, scales, opac, rgb)
    return load_mesh(path)


def normalise(asset: Mesh | Splats) -> tuple[Mesh | Splats, np.ndarray]:
    """Centre on the bounding box and scale the longest side to 1. Returns the asset and the 4x4 transform used."""
    lo, hi = asset.bounds()
    s = 1.0 / float(max(hi - lo))
    T = np.eye(4)
    T[:3, :3] *= s
    T[:3, 3] = -s * (lo + hi) / 2
    return asset.transformed(T), T


# ---------------------------------------------------------------------------------------------------- cameras


def camera(azim_deg: float, elev_deg: float = ELEV_DEG, dist: float = DIST, res: int = RES):
    """World-to-camera 4x4 (OpenCV axes) and 3x3 intrinsics for a camera on a sphere looking at the origin, z up."""
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    eye = dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    fwd = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 0.0, 1.0])
    if abs(fwd @ up) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # rows: camera x, y, z in world coordinates
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    f = 0.5 * res / math.tan(math.radians(FOV_DEG) / 2)
    K = np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1.0]])
    return V, K


def sphere_cameras(n: int = 24, res: int = 256):
    """n cameras spread evenly over the sphere (a Fibonacci lattice)."""
    cams = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        azim = math.degrees(i * math.pi * (3 - math.sqrt(5)))
        cams.append(camera(azim, math.degrees(math.asin(z)), res=res))
    return cams


# --------------------------------------------------------------------------------------------------- rendering


class Renderer:
    """Unlit colour, alpha and depth for meshes (nvdiffrast) and splats (gsplat), from the same cameras."""

    def __init__(self, device: str = "cuda"):
        import nvdiffrast.torch as dr
        import torch

        self.torch, self.dr, self.device = torch, dr, device
        self.ctx = dr.RasterizeCudaContext(device=device)

    def render(self, asset: Mesh | Splats, cams, res: int) -> dict[str, np.ndarray]:
        """Returns rgb (C, H, W, 3) on white, alpha (C, H, W) and depth (C, H, W; 0 where empty)."""
        if isinstance(asset, Splats):
            return self._render_splats(asset, cams, res)
        return self._render_mesh(asset, cams, res)

    def _render_mesh(self, mesh: Mesh, cams, res: int):
        torch, dr, dev = self.torch, self.dr, self.device
        v = torch.tensor(np.concatenate([p.v for p in mesh.parts]), device=dev)
        offsets = np.cumsum([0] + [len(p.v) for p in mesh.parts])
        faces = np.concatenate([p.f + o for p, o in zip(mesh.parts, offsets[:-1], strict=True)])
        f = torch.tensor(faces, device=dev, dtype=torch.int32)
        part_of_face = np.concatenate([np.full(len(p.f), i) for i, p in enumerate(mesh.parts)])
        face_part = torch.tensor(part_of_face, device=dev, dtype=torch.int64)
        uv = torch.zeros((len(v), 2), device=dev)
        vcol = torch.full((len(v), 3), 0.7, device=dev)
        for p, o in zip(mesh.parts, offsets[:-1], strict=True):
            if p.uv is not None:
                uv[o : o + len(p.v)] = torch.tensor(p.uv, device=dev)
            if p.vcol is not None:
                vcol[o : o + len(p.v)] = torch.tensor(p.vcol, device=dev)
        texs = [torch.tensor(p.tex, device=dev)[None].contiguous() if p.tex is not None else None for p in mesh.parts]
        flip = torch.tensor([1.0, -1.0], device=dev), torch.tensor([0.0, 1.0], device=dev)  # glTF uv origin: top
        rgbs, alphas, depths = [], [], []
        near, far = 0.05, 20.0
        for V, K in cams:
            Vt = torch.tensor(V, device=dev, dtype=torch.float32)
            cam_pts = v @ Vt[:3, :3].T + Vt[:3, 3]
            x, y, z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
            clip = torch.stack(
                [
                    (2 * K[0, 0] / res) * x + (2 * K[0, 2] / res - 1) * z,
                    (2 * K[1, 1] / res) * y + (2 * K[1, 2] / res - 1) * z,
                    z * (far + near) / (far - near) - 2 * far * near / (far - near),
                    z,
                ],
                -1,
            )[None].contiguous()
            rast, _ = dr.rasterize(self.ctx, clip, f, resolution=[res, res])
            mask = rast[..., 3] > 0
            part = face_part[(rast[..., 3].long() - 1).clamp(min=0)]
            col, _ = dr.interpolate(vcol[None], rast, f)
            uv_i, _ = dr.interpolate(uv[None], rast, f)
            for i, tex in enumerate(texs):
                if tex is not None:
                    sampled = dr.texture(tex, (uv_i * flip[0] + flip[1]).contiguous(), filter_mode="linear")
                    col = torch.where(((part == i) & mask)[..., None], sampled, col)
            zbuf, _ = dr.interpolate(z[None, :, None].contiguous(), rast, f)
            alpha = mask.float()[..., None]
            rgb = dr.antialias((col * alpha + (1 - alpha)).contiguous(), rast, clip, f)
            rgbs.append(rgb[0].clamp(0, 1).cpu().numpy())
            alphas.append(mask[0].float().cpu().numpy())
            depths.append((zbuf[0, ..., 0] * mask[0]).cpu().numpy())
        return {"rgb": np.stack(rgbs), "alpha": np.stack(alphas), "depth": np.stack(depths)}

    def _render_splats(self, sp: Splats, cams, res: int):
        import gsplat

        torch, dev = self.torch, self.device

        def t(x):
            return torch.tensor(x, device=dev, dtype=torch.float32)

        viewmats = t(np.stack([V for V, _ in cams]))
        Ks = t(np.stack([K for _, K in cams]))
        out, alpha, _ = gsplat.rasterization(
            t(sp.means), t(sp.quats), t(sp.scales), t(sp.opac), t(sp.rgb), viewmats, Ks, res, res,
            render_mode="RGB+ED", near_plane=0.05, far_plane=20.0,
        )  # fmt: skip
        a = alpha[..., 0] > 0.5
        # Colours come back weighted by coverage (on black); put them on white.
        rgb = out[..., :3] + (1 - alpha)
        return {
            "rgb": rgb.clamp(0, 1).cpu().numpy(),
            "alpha": a.float().cpu().numpy(),
            "depth": (out[..., 3] * a).cpu().numpy(),
        }


def unproject(depth: np.ndarray, V: np.ndarray, K: np.ndarray) -> np.ndarray:
    """World points (N, 3) for every pixel with depth > 0."""
    vs, us = np.nonzero(depth > 0)
    z = depth[vs, us]
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], -1)
    R, t = V[:3, :3], V[:3, 3]
    return (cam - t) @ R


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-space normals as an RGB image (white where empty), from finite differences of the depth map."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    z = depth
    P = np.stack([(us + 0.5 - K[0, 2]) / K[0, 0] * z, (vs + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dy, dx)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    valid = z > 0
    valid[:, 1:-1] &= (z[:, 2:] > 0) & (z[:, :-2] > 0)
    valid[1:-1] &= (z[2:] > 0) & (z[:-2] > 0)
    valid[[0, -1]] = False
    valid[:, [0, -1]] = False
    # The usual colouring: x right = red, y up = green, toward the viewer = blue.
    img = np.stack([n[..., 0], -n[..., 1], -n[..., 2]], -1) * 0.5 + 0.5
    img[~valid] = 1.0
    return img.astype(np.float32)


def surface_points(renderer: Renderer, asset: Mesh | Splats, n: int = 30_000, seed: int = 0) -> np.ndarray:
    cams = sphere_cameras(24, 256)
    out = renderer.render(asset, cams, 256)
    pts = np.concatenate([unproject(d, V, K) for d, (V, K) in zip(out["depth"], cams, strict=True)])
    rng = np.random.default_rng(seed)
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts.astype(np.float32)


# ----------------------------------------------------------------------------------------------------- geometry


def chamfer_fscore(a: np.ndarray, b: np.ndarray, taus=(0.02, 0.05)) -> dict[str, float]:
    from scipy.spatial import cKDTree

    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    out = {"chamfer": float((d_ab.mean() + d_ba.mean()) / 2)}
    for tau in taus:
        precision, recall = float((d_ab < tau).mean()), float((d_ba < tau).mean())
        f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out[f"fscore@{tau:g}"] = f
    return out


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / ((xs**2).sum() / len(src)))
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


def cube_rotations() -> list[np.ndarray]:
    """The 24 rotations that map a cube onto itself."""
    import itertools

    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs, strict=True)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                rots.append(R)
    return rots


def align(pred: np.ndarray, gt: np.ndarray, iters: int = 40, n: int = 4000, seed: int = 0) -> tuple[np.ndarray, float]:
    """Similarity transform (4x4) taking pred points onto gt points: 24 starts, ICP from each, best Chamfer wins."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    p = pred[rng.choice(len(pred), min(n, len(pred)), replace=False)]
    g = gt[rng.choice(len(gt), min(n, len(gt)), replace=False)]
    tree_g = cKDTree(g)
    best_T, best = np.eye(4), np.inf
    for start in cube_rotations():
        T = np.eye(4)
        T[:3, :3] = start
        for _ in range(iters):
            q = p @ T[:3, :3].T + T[:3, 3]
            _, idx = tree_g.query(q)
            # Match both ways (each gt point pulls its nearest predicted point too), so ICP cannot shrink the object.
            _, idx2 = cKDTree(q).query(g)
            T = _umeyama(np.concatenate([p, p[idx2]]), np.concatenate([g[idx], g]))
        cd = chamfer_fscore(p @ T[:3, :3].T + T[:3, 3], g)["chamfer"]
        if cd < best:
            best, best_T = cd, T
    return best_T, best


# ------------------------------------------------------------------------------------------------ image metrics


class ImageMetrics:
    def __init__(self, device: str = "cuda"):
        import lpips
        import open_clip
        import torch

        self.torch, self.device = torch, device
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        self.clip, _, self.clip_pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.clip = self.clip.to(device).eval()

    def clip_embed(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        torch = self.torch
        batch = torch.stack([self.clip_pre(Image.fromarray((im * 255).astype(np.uint8))) for im in images])
        with torch.no_grad():
            e = self.clip.encode_image(batch.to(self.device)).float()
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def compare(self, pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
        """PSNR, SSIM, LPIPS for (C, H, W, 3) images in [0, 1] on white."""
        from skimage.metrics import structural_similarity

        torch = self.torch
        mse = ((pred - gt) ** 2).mean(axis=(1, 2, 3))
        psnr = float(np.mean(-10 * np.log10(np.maximum(mse, 1e-10))))
        pairs = zip(pred, gt, strict=True)
        ssim = float(np.mean([structural_similarity(a, b, channel_axis=-1, data_range=1.0) for a, b in pairs]))
        with torch.no_grad():

            def t(x):
                return torch.tensor(x, device=self.device).permute(0, 3, 1, 2) * 2 - 1

            lp = float(self.lpips(t(pred), t(gt)).mean())
        return {"psnr": psnr, "ssim": ssim, "lpips": lp}


def turntable(renderer: Renderer, asset: Mesh | Splats, res: int = RES) -> dict[str, np.ndarray]:
    cams = [camera(a, res=res) for a in TURNTABLE]
    out = renderer.render(asset, cams, res)
    out["normal"] = np.stack([normals_from_depth(d, K) for d, (_, K) in zip(out["depth"], cams, strict=True)])
    return out


def evaluate(renderer, metrics, pred: Mesh | Splats, gt: Mesh | Splats, input_rgb: np.ndarray, gt_views=None) -> dict:
    """Every score for one prediction. gt must already be normalised; input_rgb is the input image on white."""
    pred_n, _ = normalise(pred)
    p_pts = surface_points(renderer, pred_n)
    g_pts = surface_points(renderer, gt)
    T, _ = align(p_pts, g_pts)
    pred_a = pred_n.transformed(T)
    scores = chamfer_fscore(p_pts @ T[:3, :3].T + T[:3, 3], g_pts)
    tp = turntable(renderer, pred_a)
    tg = gt_views if gt_views is not None else turntable(renderer, gt)
    novel = slice(1, None)
    scores.update(metrics.compare(tp["rgb"][novel], tg["rgb"][novel]))
    inter = (tp["alpha"][novel] * tg["alpha"][novel]).sum(axis=(1, 2))
    union = np.maximum(tp["alpha"][novel], tg["alpha"][novel]).sum(axis=(1, 2))
    scores["mask_iou"] = float(np.mean(inter / np.maximum(union, 1)))
    e_in = metrics.clip_embed([input_rgb])[0]
    scores["clip"] = float((metrics.clip_embed(list(tp["rgb"])) @ e_in).mean())
    scores["normal_lpips"] = metrics.compare(tp["normal"][novel], tg["normal"][novel])["lpips"]
    return {"scores": scores, "align": T.tolist(), "pred_asset": pred_a, "pred_views": tp, "gt_views": tg}


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--input", required=True, help="the image the model was given")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    renderer, metrics = Renderer(), ImageMetrics()
    gt, _ = normalise(load_asset(args.gt))
    rgba = np.asarray(Image.open(args.input).convert("RGBA"), np.float32) / 255
    input_rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
    result = evaluate(renderer, metrics, load_asset(args.pred), gt, input_rgb)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(result["scores"], indent=2))
    for kind in ("rgb", "normal"):
        strip = np.concatenate(list(result["pred_views"][kind]), axis=1)
        Image.fromarray((strip * 255).astype(np.uint8)).save(out / f"{kind}.png")
    print(json.dumps(result["scores"], indent=2))


if __name__ == "__main__":
    main()
