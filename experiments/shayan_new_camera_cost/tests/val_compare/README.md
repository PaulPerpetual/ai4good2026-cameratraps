# Local tests for val_compare.py

Run on the laptop on 11 October 2026, before the script went to Astra. They test code
paths only. The features are made up (32 dimensions, class mean plus camera shift plus
noise) on the real index, so no number they produce says anything about the real data.

- `make_synth.py` builds the synthetic cache from the real index. Subsamples of 256 and
  1,500 rows were cut from it to get species that are missing from training.
- `harness.py` runs `val_compare.main()` with the grid-edge stop switched off, so that
  stage 2 is exercised. The approved script has no such switch.
- `checks.py` recomputes every reported number from the saved prediction files with an
  independent macro-F1, and checks the selection, the contrasts and the outcome rule.
- `lda_check.py` compares the linear discriminant with the Mahalanobis form computed
  directly, and checks that pooling adjusted scores equals adjusting pooled scores.
- `fit_check.py` checks that `fit_linear` returns the same weights as `pilot_a.fit_head`.
- `nested_check.py` recomputes the nested cross-validation for the linear discriminant with its
  own loops and indexing and compares the predictions and the per-fold choices.
- `pilot_harness.py` runs `pilot_a.py` itself with temperature 1.5 and exponent 0.5 fixed,
  to check that val_compare.py's copy of the pilot pipeline gives the same number.
- `stubs/` stands in for torchvision, which the laptop does not have.

Stops tested: grid edge (exit 5), optimiser cap (exit 5), results or marker already
present (exit 6), earlier selection disagrees (exit 7), `--select-only`.
