"""AI-ness rulers: voi-slop-detect-380M locally, our 9B detector (run f9-9b on the slop-detector Modal volume)
for the least-sure 10% (the inkprint cascade). Pangram stays optional and is not called here.

voi-380M is loaded natively (transformers >= 5.22) at a pinned revision; never with dmis-lab's remote code,
which scores every text ~0.73 (lesson from the slop-detector series).
"""

from functools import lru_cache

import numpy as np

VOI380, VOI380_REV = "VoightKampffLabs/voi-slop-detect-380M", "a109d591b8b62541f362d873765900380b2bd0ea"


@lru_cache(None)
def _voi():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(VOI380, revision=VOI380_REV)
    net = AutoModelForSequenceClassification.from_pretrained(VOI380, revision=VOI380_REV).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return tok, net.to(dev), torch, dev


def voi380(texts, bs=16):
    tok, net, torch, dev = _voi()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i : i + bs], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
            out += torch.sigmoid(net(**enc).logits.float()).reshape(-1).cpu().tolist()
    return np.array(out)


def nine_b(texts):
    """Score with the 9B detector on Modal (run f9-9b). Requires the slop-detector app to be deployed."""
    import modal

    fn = modal.Function.from_name("slop-detector", "score_run")
    return np.array(fn.remote("f9-9b", {"x": list(texts)})["x"])


def cascade(texts, frac=0.10, cut=0.5):
    """voi-380M everywhere; the 10% closest to the cut are re-scored by the 9B."""
    s = voi380(texts)
    k = max(1, round(frac * len(texts)))
    idx = np.argsort(np.abs(s - cut))[:k]
    try:
        s9 = nine_b([texts[i] for i in idx])
        s = s.copy()
        s[idx] = s9
        used9 = True
    except Exception as e:  # Modal unavailable: keep 380M scores and say so
        print("9B cascade unavailable:", type(e).__name__)
        used9 = False
    return s, used9


if __name__ == "__main__":
    human = "Phillip,\n\nI talked to Mike about the Q3 numbers. He wants the curve updated by Friday, not Monday. Can you get it to me Thursday night so I can look before the call?\n\nThanks\nJohn"
    ai = "I hope this message finds you well! I wanted to reach out to explore how our innovative solutions can help streamline your operations and unlock new levels of efficiency. Would you be open to a quick call to discuss how we can partner together?"
    print("sanity voi380 human/ai:", voi380([human, ai]).round(3))
