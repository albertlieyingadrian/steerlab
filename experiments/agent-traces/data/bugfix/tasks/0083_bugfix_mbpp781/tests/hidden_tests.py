import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert count_divisors(10)
except Exception:
    failed += 1
try:
    assert not count_divisors(100)
except Exception:
    failed += 1
try:
    assert count_divisors(125)
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
