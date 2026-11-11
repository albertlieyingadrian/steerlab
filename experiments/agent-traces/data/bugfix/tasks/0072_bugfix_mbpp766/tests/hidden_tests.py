import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert pair_wise([1,1,2,3,3,4,4,5])==[(1, 1), (1, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5)]
except Exception:
    failed += 1
try:
    assert pair_wise([1,5,7,9,10])==[(1, 5), (5, 7), (7, 9), (9, 10)]
except Exception:
    failed += 1
try:
    assert pair_wise([5,1,9,7,10])==[(5, 1), (1, 9), (9, 7), (7, 10)]
except Exception:
    failed += 1
try:
    assert pair_wise([1,2,3,4,5,6,7,8,9,10])==[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
