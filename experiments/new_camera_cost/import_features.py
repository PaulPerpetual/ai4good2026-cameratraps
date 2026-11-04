"""Bring a teammate's frozen features into the cache format the pilot_a scripts read.

    python import_features.py --npz <their .npz> --order-csv <their md.csv> \
        --index features/dinov2_vitb14_reg_448/index.csv.gz --out features/bioclip_whole_paul

The source is a `.npz` with a float16 array `x`, one row per labelled image of the task
(train, validation AND test), and its file names as a pickled object array. A pickle can
make an ordinary unpickler call any function, so the names are rebuilt by a loader
restricted to NumPy's array-rebuilding callables (`_NumpyArrayOnly`, the approach of the
`pickle` documentation's "Restricting Globals"). It is a restricted loader for the NumPy
files inspected here, not a general security sandbox. It gives the array's true element
order.
Those names, read from the same file as `x`, must equal the plain order CSV row for row.
The numeric array, test rows included, is loaded into memory; the test rows are then
dropped without being looked at or scored.

The output keeps only the labelled images of our own cache index (training and validation
cameras), in our index order, so that runs on it score exactly the same queries as runs
on our cache. Test rows are not copied. There are no empty frames in the output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT = "official_ood"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _NumpyArrayOnly(pickle.Unpickler):
    """A loader for NumPy object arrays: globals resolved through `find_class` are limited
    to NumPy's array-rebuilding callables.

    A restricted loader for the inspected protocol-4 NumPy files, not a general security
    sandbox: for example, extension codes already cached by `copyreg` do not go through
    `find_class`.
    """

    ALLOWED = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def names_from_npz(path: Path, member: str = "file_name") -> np.ndarray:
    """File names stored in a .npz as a pickled object array, in the array's element order.

    The member is a .npy file: a header, parsed with `ast.literal_eval`, then a pickle of
    the array, rebuilt with `_NumpyArrayOnly`. The result must be a one-dimensional object
    array of strings with the header's shape.
    """
    with zipfile.ZipFile(path) as z:
        raw = z.read(member + ".npy")
    assert raw[:6] == b"\x93NUMPY", "not a .npy member"
    size = 2 if raw[6] == 1 else 4
    start = 8 + size + int.from_bytes(raw[8:8 + size], "little")
    header = ast.literal_eval(raw[8 + size:start].decode("latin1"))
    assert header["descr"] == "|O" and not header["fortran_order"] and len(header["shape"]) == 1, header
    arr = _NumpyArrayOnly(io.BytesIO(raw[start:])).load()
    assert isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == tuple(header["shape"]), \
        "the member did not rebuild into a one-dimensional object array of the header's shape"
    assert all(isinstance(v, str) for v in arr), "not every stored name is a string"
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="source features: .npz with a float16 array x")
    ap.add_argument("--order-csv", required=True, help="CSV with the source rows' file_name, in the same order")
    ap.add_argument("--index", required=True, help="our cache's index.csv.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="", help="provenance, written into config.json")
    ap.add_argument("--expect-rows", type=int, required=True, help="number of rows the source must have")
    ap.add_argument("--expect-dim", type=int, required=True, help="feature dimension the source must have")
    a = ap.parse_args()

    out = Path(a.out)
    assert not out.exists(), f"{out} exists; not overwriting"
    with np.load(a.npz, allow_pickle=False) as z:          # reading x never unpickles anything
        x = z["x"]
    assert x.dtype == np.float16 and x.shape == (a.expect_rows, a.expect_dim), f"source is {x.dtype} {x.shape}"
    order = pd.read_csv(a.order_csv, usecols=["file_name"])["file_name"].to_numpy()
    assert len(x) == len(order), f"{len(x)} feature rows but {len(order)} file names"
    assert pd.Index(order).is_unique, "duplicate file names in the order file"
    stored = names_from_npz(Path(a.npz))                     # the names saved next to x, in x's order
    assert len(stored) == len(x) and (stored == order).all(), "the order file does not match the names stored with x"

    ix = pd.read_csv(a.index)
    lab = ix[~ix["is_empty"].astype(bool)].reset_index(drop=True)
    assert not lab[SPLIT].eq("test").any(), "our index must not contain test rows"
    pos = pd.Index(order).get_indexer(lab["file_name"])
    missing = int((pos < 0).sum())
    assert missing == 0, f"{missing} of our labelled images have no source row"
    feats = np.asarray(x[pos], dtype=np.float16)
    assert np.isfinite(feats.astype(np.float32)).all(), "non-finite feature values"

    out.mkdir(parents=True)
    lab.to_csv(out / "index.csv.gz", index=False)
    np.save(out / "features.npy", feats)
    np.save(out / "status.npy", np.ones(len(lab), dtype=np.uint8))
    config = {
        "source_npz": str(Path(a.npz).resolve()), "source_npz_sha256": sha256(Path(a.npz)),
        "order_csv": str(Path(a.order_csv).resolve()), "order_csv_sha256": sha256(Path(a.order_csv)),
        "source_rows": int(len(x)), "rows_kept": int(len(lab)), "dim": int(x.shape[1]), "dtype": "float16",
        "row_order_check": "names rebuilt from the npz's own file_name member by an unpickler restricted to NumPy's "
                           "array-rebuilding callables equal the order CSV",
        "kept": "labelled images of our index (training and validation cameras), in our index order",
        "not_kept": "test rows of the source; there are no empty frames",
        "counts": lab[SPLIT].value_counts().to_dict(), "index_from": str(Path(a.index).resolve()),
        "note": a.note,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ("source_rows", "rows_kept", "dim", "counts")}))


if __name__ == "__main__":
    main()
