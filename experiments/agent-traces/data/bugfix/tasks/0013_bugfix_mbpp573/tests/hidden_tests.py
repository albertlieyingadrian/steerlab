import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert unique_product([10, 20, 30, 40, 20, 50, 60, 40]) ==  720000000
except Exception:
    failed += 1
try:
    assert unique_product([1, 2, 3, 1,]) == 6
except Exception:
    failed += 1
try:
    assert unique_product([7, 8, 9, 0, 1, 1]) == 0
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
