import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert sum_average(10)==(55, 5.5)
except Exception:
    failed += 1
try:
    assert sum_average(15)==(120, 8.0)
except Exception:
    failed += 1
try:
    assert sum_average(20)==(210, 10.5)
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
