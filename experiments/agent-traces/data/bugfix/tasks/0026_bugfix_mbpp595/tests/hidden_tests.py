import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert min_Swaps("1101","1110") == 1
except Exception:
    failed += 1
try:
    assert min_Swaps("111","000") == "Not Possible"
except Exception:
    failed += 1
try:
    assert min_Swaps("111","110") == "Not Possible"
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
