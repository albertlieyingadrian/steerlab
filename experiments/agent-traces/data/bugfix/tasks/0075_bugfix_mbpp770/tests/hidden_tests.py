import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert odd_num_sum(2) == 82
except Exception:
    failed += 1
try:
    assert odd_num_sum(3) == 707
except Exception:
    failed += 1
try:
    assert odd_num_sum(4) == 3108
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
