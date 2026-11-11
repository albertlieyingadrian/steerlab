import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert find_Odd_Pair([5,4,7,2,1],5) == 6
except Exception:
    failed += 1
try:
    assert find_Odd_Pair([7,2,8,1,0,5,11],7) == 12
except Exception:
    failed += 1
try:
    assert find_Odd_Pair([1,2,3],3) == 2
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
