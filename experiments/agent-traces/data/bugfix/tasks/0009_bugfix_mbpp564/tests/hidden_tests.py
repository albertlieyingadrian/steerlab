import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert count_Pairs([1,2,1],3) == 2
except Exception:
    failed += 1
try:
    assert count_Pairs([1,1,1,1],4) == 0
except Exception:
    failed += 1
try:
    assert count_Pairs([1,2,3,4,5],5) == 10
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
