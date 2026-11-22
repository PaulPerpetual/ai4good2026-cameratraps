"""Stand-in so the repo's data module imports on a machine without torchvision."""
class Compose:  # only referenced inside functions the tests never call
    def __init__(self, *a, **k): raise RuntimeError("torchvision stub")
class InterpolationMode: BILINEAR = "bilinear"
def __getattr__(name): return Compose
