import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert issort_list([1,2,4,6,8,10,12,14,16,17])==True
except Exception:
    failed += 1
try:
    assert issort_list([1, 2, 4, 6, 8, 10, 12, 14, 20, 17])==False
except Exception:
    failed += 1
try:
    assert issort_list([1, 2, 4, 6, 8, 10,15,14,20])==False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
