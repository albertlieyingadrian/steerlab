import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert differ_At_One_Bit_Pos(13,9) == True
except Exception:
    failed += 1
try:
    assert differ_At_One_Bit_Pos(15,8) == False
except Exception:
    failed += 1
try:
    assert differ_At_One_Bit_Pos(2,4) == False
except Exception:
    failed += 1
try:
    assert differ_At_One_Bit_Pos(2, 3) == True
except Exception:
    failed += 1
try:
    assert differ_At_One_Bit_Pos(5, 1) == True
except Exception:
    failed += 1
try:
    assert differ_At_One_Bit_Pos(1, 5) == True
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
