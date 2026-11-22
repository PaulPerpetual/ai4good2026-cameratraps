"""What the two MegaDetectors cost per photo, measured like efficiency_cost.py measures backbones.

    ../.venv-megadetector/bin/python detector_cost.py

Both run at their native 1280 px input. GFLOPs counted by PyTorch; CPU latency for one photo
with 4 threads (vinzant's CPU, so only the comparison is meaningful).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PytorchWildlife.models.detection import MegaDetectorV5, MegaDetectorV6
from torch.utils.flop_counter import FlopCounterMode

OUT = Path("/zfs/ai4good/groups/camera-traps/users/george/followup_results/detector_cost.json")
SIZE = 1280
THREADS = 4
REPEATS = 10


def main():
    torch.set_num_threads(THREADS)
    networks = {
        "md5a": MegaDetectorV5(device="cpu", pretrained=True, version="a").model,
        "md6c": MegaDetectorV6(device="cpu", pretrained=True, version="MDV6-yolov10-c").predictor.model.model,
    }
    rows = []
    for name, network in networks.items():
        network = network.float().eval()
        image = torch.randn(1, 3, SIZE, SIZE)
        row = {"detector": name, "size": SIZE, "parameters_million": sum(p.numel() for p in network.parameters()) / 1e6,
               "gflops": count_flops(network, image) / 1e9, "cpu_ms": latency_ms(network, image)}
        rows.append(row)
        print(f"{name}  {row['parameters_million']:6.1f}M  {row['gflops']:7.1f} GFLOPs  {row['cpu_ms']:7.0f} ms")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"threads": THREADS, "rows": rows}, indent=2))


@torch.inference_mode()
def count_flops(network, image) -> int:
    counter = FlopCounterMode(display=False)
    with counter:
        network(image)
    return counter.get_total_flops()


@torch.inference_mode()
def latency_ms(network, image) -> float:
    for _ in range(2):
        network(image)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        network(image)
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000)


if __name__ == "__main__":
    main()
