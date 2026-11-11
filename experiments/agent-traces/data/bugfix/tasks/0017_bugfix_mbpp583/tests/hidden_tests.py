import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert catalan_number(10)==16796
except Exception:
    failed += 1
try:
    assert catalan_number(9)==4862
except Exception:
    failed += 1
try:
    assert catalan_number(7)==429
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
