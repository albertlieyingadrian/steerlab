import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert largest_subset([ 1, 3, 6, 13, 17, 18 ]) == 4
except Exception:
    failed += 1
try:
    assert largest_subset([10, 5, 3, 15, 20]) == 3
except Exception:
    failed += 1
try:
    assert largest_subset([18, 1, 3, 6, 13, 17]) == 4
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
