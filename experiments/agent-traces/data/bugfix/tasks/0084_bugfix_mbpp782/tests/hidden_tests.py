import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert odd_length_sum([1,2,4]) == 14
except Exception:
    failed += 1
try:
    assert odd_length_sum([1,2,1,2]) == 15
except Exception:
    failed += 1
try:
    assert odd_length_sum([1,7]) == 8
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
