import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert not is_perfect_square(10)
except Exception:
    failed += 1
try:
    assert is_perfect_square(36)
except Exception:
    failed += 1
try:
    assert not is_perfect_square(14)
except Exception:
    failed += 1
try:
    assert is_perfect_square(14*14)
except Exception:
    failed += 1
try:
    assert not is_perfect_square(125)
except Exception:
    failed += 1
try:
    assert is_perfect_square(125*125)
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
