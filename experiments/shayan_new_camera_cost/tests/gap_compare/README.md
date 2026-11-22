# Local tests for gap_measure.py (version 2) and gap_compare.py

Run on the laptop on 20 October 2026 with the synthetic caches of `tests/val_compare/make_synth.py`
(real index, made-up 32-dimensional features) and a second cache with the same index and
noisier features. The numbers mean nothing; the tests check code paths.

- The approved gap_measure.py (kept as `gap_measure_v1.py`) and version 2 give identical
  `results` and identical fits (iterations, gradients) on the same cache. Version 2 only adds
  `query_images.csv.gz` and records both optimiser caps through `val_compare.fit_linear`.
- gap_compare.py on the two synthetic runs: every number it re-scores from the saved
  predictions matches both runs' own results.json (asserted).
- gap_compare.py of a run against itself: every difference and interval is exactly zero.
- gap_compare.py of runs on different image sets (a 3,000-row subsample): refused by the
  alignment assertion.

## import_features.py (20 October 2026)

A fake teammate file built from the synthetic cache: a float16 `x` of every labelled image
plus 3,000 fake test rows, in a shuffled order, with the file names pickled as in Paul's
files and the same order in a plain CSV.
- The import keeps exactly our labelled rows in our order, with features matched by file
  name, and no test rows (asserted).
- Refused: an order CSV with a different length; an output folder that already exists.
- gap_measure.py on the imported cache, then gap_compare.py against the run on our own
  cache: the two runs score identical queries (asserted) and every re-scored number matches.
- Version with `names_from_npz` (after Astra's rejection of Run 7): the file names stored in an
  `.npz` as a pickled object array are read by parsing the pickle's opcodes with
  `pickletools`, which executes nothing. On a 5,000-name test file and the fake teammate
  file they come back in array order. An order CSV that no longer matches the stored names
  is refused, and so is a source with the wrong shape. The import result is unchanged.
- On the real files (read-only, on the node, 20 October): in both `bioclip.npz` and
  `bioclip-crop.npz`, the 127,681 names stored next to `x` equal `md.csv` row for row, and
  `x` is float16 127,681 x 512.
- Version with the restricted unpickler (after Astra's second rejection of Run 7), tests in
  `names_check.py`: names saved by np.savez come back in exact order; a crafted file whose
  name literals appear in the order (b, a) while the array holds (a, b) through memo
  references returns (a, b), and an order file listing the literal order is refused by the
  import; a crafted member that would call os.system is refused (`refusing to load
  posix.system`) and the command does not run. Re-checked on the real files: in both npz
  files the 127,681 rebuilt names equal md.csv row for row.
