"""CLIP ViT-L/14-336 pieces shared by the Modal cache job and the local checks.

- `preprocess`: the CLIP image transform (resize shortest side to 336 bicubic, centre crop, normalise), as
  `CLIPImageProcessor` does for `openai/clip-vit-large-patch14-336`.
- `forward_cache`: one batch through the vision tower; returns the block-23 CLS residual (the DeBiasLens SAE's input
  row that reaches the embedding), the embedding, the block-22 output tokens (what LLaVA-1.5 reads), and, when asked,
  the exact direct contributions of each head and MLP of the last `n_last` blocks to the CLS row.

The decomposition: r_24[CLS] = r_{24-n}[CLS] + sum over the last n blocks of (sum_h head_h + out_proj bias + mlp).
head_h = softmax(q_cls k^T * scale)_h v_h W_out[:, h] (the CLS query row of head h through its slice of out_proj).
"""

from __future__ import annotations

import torch

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
SIZE = 336


def preprocess(img):
    """PIL image -> float tensor [3, 336, 336], the CLIP transform."""
    import torchvision.transforms.functional as F
    from torchvision.transforms import InterpolationMode

    img = img.convert("RGB")
    img = F.resize(img, SIZE, interpolation=InterpolationMode.BICUBIC)
    img = F.center_crop(img, [SIZE, SIZE])
    t = F.to_tensor(img)
    return F.normalize(t, MEAN, STD)


@torch.no_grad()
def head_contributions(layer, x_in):
    """Exact per-head CLS-row output of one CLIP encoder layer's attention.

    x_in: [B, T, D] input to the layer (float32). Returns heads [B, H, D] and the out_proj bias [D]; their sum over
    heads plus the bias equals the attention block's output at the CLS row.
    """
    attn = layer.self_attn
    h = layer.layer_norm1(x_in)
    B, T, D = h.shape
    H = attn.num_heads
    hd = D // H
    q = attn.q_proj(h[:, :1]).view(B, 1, H, hd).transpose(1, 2)  # [B, H, 1, hd]
    k = attn.k_proj(h).view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]
    v = attn.v_proj(h).view(B, T, H, hd).transpose(1, 2)
    w = torch.softmax((q @ k.transpose(-1, -2)) * attn.scale, dim=-1)  # [B, H, 1, T]
    o = (w @ v).squeeze(2)  # [B, H, hd]
    W = attn.out_proj.weight  # [D, D]; out = o_flat @ W.T + b
    Wh = W.view(D, H, hd).permute(1, 2, 0)  # [H, hd, D]
    heads = torch.einsum("bhk,hkd->bhd", o, Wh)
    return heads, attn.out_proj.bias


@torch.no_grad()
def forward_cache(model, pixels, n_last=0):
    """model: transformers CLIPModel. pixels: [B, 3, 336, 336] on the model's device/dtype."""
    vm = model.vision_model
    out = vm(pixel_values=pixels, output_hidden_states=True)
    hs = out.hidden_states  # hs[0] input to block 0; hs[k] output of block k-1
    L = len(vm.encoder.layers)
    cls23 = hs[L][:, 0].float()
    emb = model.visual_projection(vm.post_layernorm(hs[L][:, 0])).float()
    res = {"cls23": cls23, "emb": emb, "tok22": hs[L - 1][:, 1:], "tok23": hs[L], "cls_in": None}
    if n_last:
        heads, mlps, biases = [], [], []
        for li in range(L - n_last, L):
            layer = vm.encoder.layers[li].float()
            x_in = hs[li].float()
            hh, b = head_contributions(layer, x_in)
            after_attn = x_in + layer.self_attn(layer.layer_norm1(x_in))[0]
            mlp = layer.mlp(layer.layer_norm2(after_attn))[:, 0]
            heads.append(hh)
            mlps.append(mlp)
            biases.append(b.clone())
            vm.encoder.layers[li].to(pixels.dtype)
        res["heads"] = torch.stack(heads, 1)  # [B, n_last, H, D]
        res["mlps"] = torch.stack(mlps, 1)  # [B, n_last, D]
        res["attn_bias"] = torch.stack(biases, 0).float()  # [n_last, D]
        res["cls_in"] = hs[L - n_last][:, 0].float()
    return res
