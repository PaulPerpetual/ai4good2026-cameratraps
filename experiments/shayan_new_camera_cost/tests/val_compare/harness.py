"""Test harness: run val_compare.main() with the edge stop disabled. Synthetic data only."""
import sys
import val_compare
val_compare.ARBITRARY_EDGES = {"l2": set(), "tau": set()}
sys.argv = ["val_compare.py"] + sys.argv[1:]
val_compare.main()
