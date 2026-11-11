import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert sum_Of_product(3) == 15
except Exception:
    failed += 1
try:
    assert sum_Of_product(4) == 56
except Exception:
    failed += 1
try:
    assert sum_Of_product(1) == 1
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
