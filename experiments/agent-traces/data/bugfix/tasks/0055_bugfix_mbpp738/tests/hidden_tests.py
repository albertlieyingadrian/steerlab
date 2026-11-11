import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert geometric_sum(7) == 1.9921875
except Exception:
    failed += 1
try:
    assert geometric_sum(4) == 1.9375
except Exception:
    failed += 1
try:
    assert geometric_sum(8) == 1.99609375
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
