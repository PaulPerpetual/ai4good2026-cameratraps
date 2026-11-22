"""George's cache_features_timm.py with the input size forced to 448, to match the DINOv2 caches.

    python cache_dinov3_448.py --model vit_base_patch16_dinov3.lvd1689m --crop --out <dir>
"""

import sys
from functools import partial

import timm

sys.path.insert(0, "../../george/scripts")
import cache_features_timm as cft  # noqa: E402

SIZE = 448
resolve = timm.data.resolve_data_config
timm.create_model = partial(timm.create_model, img_size=SIZE)
timm.data.resolve_data_config = lambda *a, **k: {**resolve(*a, **k), "input_size": (3, SIZE, SIZE)}
cft.main()
