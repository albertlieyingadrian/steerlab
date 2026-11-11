import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert is_not_prime(2) == False
except Exception:
    failed += 1
try:
    assert is_not_prime(10) == True
except Exception:
    failed += 1
try:
    assert is_not_prime(35) == True
except Exception:
    failed += 1
try:
    assert is_not_prime(37) == False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
