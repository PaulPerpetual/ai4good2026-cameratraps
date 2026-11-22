"""What each backbone costs per photo, at each input size.

    python efficiency_cost.py                          # every backbone
    python efficiency_cost.py --backbones resnet50 dinov2l

For every backbone and input size: parameters, GFLOPs (counted by PyTorch, hardware
independent), and CPU latency for one photo at a time with 4 threads, as a stand-in for a
Raspberry-Pi-class computer (4 cores). The latency is measured on vinzant's CPU, so only the
comparison between models is meaningful, not the absolute times.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import exp3_path  # noqa: F401
from common import RESULTS
from models import Backbone
from settings import N_SPECIES

SIZES = {"mobilenetv3": (224, 448), "resnet50": (112, 224, 336, 448), "dinov2s": (112, 224, 336, 448),
         "dinov2b": (224, 448), "dinov2l": (112, 224, 336, 448)}
THREADS = 4
REPEATS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=list(SIZES), choices=list(SIZES))
    args = parser.parse_args()
    torch.set_num_threads(THREADS)
    rows = []
    for backbone in args.backbones:
        sizes = SIZES[backbone]
        model = Backbone(backbone).eval()
        parameters = sum(p.numel() for p in model.parameters())
        for size in sizes:
            image = torch.randn(1, 3, size, size)
            with torch.no_grad():
                features = model(image).shape[1]
            row = {"backbone": backbone, "size": size, "parameters_million": parameters / 1e6,
                   "gflops": count_flops(model, image) / 1e9,
                   "classifier_gflops": 2 * features * N_SPECIES / 1e9,   # the linear discriminant on top
                   "cpu_ms": latency_ms(model, image)}
            rows.append(row)
            print(f"{backbone:12s} {size:4d}px  {row['parameters_million']:6.1f}M  {row['gflops']:7.2f} GFLOPs  "
                  f"{row['cpu_ms']:7.1f} ms")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "efficiency_cost.json"
    earlier = json.loads(path.read_text())["rows"] if path.exists() else []
    rows += [r for r in earlier if r["backbone"] not in args.backbones]
    path.write_text(json.dumps({"threads": THREADS, "rows": rows}, indent=2))


@torch.inference_mode()
def count_flops(model, image) -> int:
    counter = FlopCounterMode(display=False)
    with counter:
        model(image)
    return counter.get_total_flops()


@torch.inference_mode()
def latency_ms(model, image) -> float:
    """Median time of one forward pass, after a few warm-up passes."""
    for _ in range(3):
        model(image)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        model(image)
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


if __name__ == "__main__":
    main()
