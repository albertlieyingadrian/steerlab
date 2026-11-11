import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert unique_Element([1,1,1]) == True
except Exception:
    failed += 1
try:
    assert unique_Element([1,2,1,2]) == False
except Exception:
    failed += 1
try:
    assert unique_Element([1,2,3,4,5]) == False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
