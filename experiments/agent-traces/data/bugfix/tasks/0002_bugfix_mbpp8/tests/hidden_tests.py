import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])==[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
except Exception:
    failed += 1
try:
    assert square_nums([10,20,30])==([100,400,900])
except Exception:
    failed += 1
try:
    assert square_nums([12,15])==([144,225])
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
