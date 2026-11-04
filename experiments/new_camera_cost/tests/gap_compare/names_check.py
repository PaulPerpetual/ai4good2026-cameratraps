"""Tests for import_features.names_from_npz. Run from the pilot_a folder:

    python tests/gap_compare/names_check.py <scratch dir>

1. Names saved by np.savez come back in their exact order.
2. A crafted file whose name literals appear in one order while the array, through memo
   references, holds them in the other: the true array order comes back, so an order file
   that lists the literal order is refused by the import.
3. A crafted file that tries to call os.system is refused, and the command does not run.
"""

import io
import pickle
import pickletools
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from import_features import names_from_npz  # noqa: E402

S = Path(sys.argv[1])
S.mkdir(parents=True, exist_ok=True)


def npz_with_member(path: Path, header_from: np.ndarray, payload: bytes, x: np.ndarray) -> None:
    """An .npz whose file_name.npy has the .npy header of `header_from` and the given pickle."""
    buf = io.BytesIO()
    np.save(buf, header_from, allow_pickle=True)
    raw = buf.getvalue()
    size = 2 if raw[6] == 1 else 4
    start = 8 + size + int.from_bytes(raw[8:8 + size], "little")
    xb = io.BytesIO()
    np.save(xb, x)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.npy", xb.getvalue())
        z.writestr("file_name.npy", raw[:start] + payload)


# 1. ordinary np.savez
names = np.array([f"{i:06x}-21bc-11ea.jpg" for i in range(3000)], dtype=object)
np.savez(S / "plain.npz", x=np.zeros((3000, 4), np.float16), file_name=names)
assert (names_from_npz(S / "plain.npz") == names).all()
print("1. np.savez names come back in their exact order")

# 2. literals in order (b, a), array (a, b) through memo references
base = pickle.dumps(np.array(["a.jpg", "b.jpg"], dtype=object), protocol=3)
ops = list(pickletools.genops(base))
mark = next(pos for (op, arg, pos) in ops if op.name == "MARK" and pos > next(p for (o, a, p) in ops if o.name == "EMPTY_LIST"))
appends = next(pos for (op, arg, pos) in ops if op.name == "APPENDS")
lit = lambda s: b"X" + struct.pack("<I", len(s)) + s.encode()
crafted = (base[:2] + lit("b.jpg") + b"q\xc8" + b"0" + lit("a.jpg") + b"q\xc9" + b"0"
           + base[2:mark + 1] + b"h\xc9" + b"h\xc8" + base[appends:])
assert (pickle.loads(crafted) == np.array(["a.jpg", "b.jpg"], dtype=object)).all()     # our own bytes, safe to load here
literal_order = [a for (o, a, p) in pickletools.genops(crafted) if o.name == "BINUNICODE" and a.endswith(".jpg")]
assert literal_order == ["b.jpg", "a.jpg"]
npz_with_member(S / "reordered.npz", np.array(["a.jpg", "b.jpg"], dtype=object), crafted, np.zeros((2, 4), np.float16))
got = names_from_npz(S / "reordered.npz")
assert list(got) == ["a.jpg", "b.jpg"], got
print("2. literals appear as", literal_order, "but the reader returns the true order", list(got))
ix = pd.DataFrame({"file_name": ["a.jpg", "b.jpg"], "location": [1, 2], "category": "x", "is_empty": False,
                   "burst_id": ["u", "v"], "datetime": "2020-01-01 00:00:00", "hour": 0, "official_ood": "train", "y": 0})
ix.to_csv(S / "two_index.csv.gz", index=False)
pd.DataFrame({"file_name": literal_order}).to_csv(S / "literal_order.csv", index=False)
r = subprocess.run([sys.executable, "import_features.py", "--npz", str(S / "reordered.npz"), "--order-csv",
                    str(S / "literal_order.csv"), "--index", str(S / "two_index.csv.gz"), "--out", str(S / "imp_reordered"),
                    "--expect-rows", "2", "--expect-dim", "4"], capture_output=True, text=True)
assert r.returncode != 0 and "does not match the names stored" in r.stderr, r.stderr[-300:]
print("   an order file listing the literal order is refused by the import")

# 3. a pickle that would call os.system
marker = S / "should_not_exist.txt"
marker.unlink(missing_ok=True)


class Evil:
    def __reduce__(self):
        import os
        return (os.system, (f"echo pwned > {marker}",))


npz_with_member(S / "evil.npz", np.array(["a.jpg"], dtype=object), pickle.dumps(Evil(), protocol=4),
                np.zeros((1, 4), np.float16))
try:
    names_from_npz(S / "evil.npz")
    raise SystemExit("the malicious member was loaded")
except pickle.UnpicklingError as e:
    assert not marker.exists(), "the command ran"
    print("3. refused:", e, "| command did not run")
