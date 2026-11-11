import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert is_polite(7) == 11
except Exception:
    failed += 1
try:
    assert is_polite(4) == 7
except Exception:
    failed += 1
try:
    assert is_polite(9) == 13
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
