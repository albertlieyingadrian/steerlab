import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert min_k([('Manjeet', 10), ('Akshat', 4), ('Akash', 2), ('Nikhil', 8)], 2) == [('Akash', 2), ('Akshat', 4)]
except Exception:
    failed += 1
try:
    assert min_k([('Sanjeev', 11), ('Angat', 5), ('Akash', 3), ('Nepin', 9)], 3) == [('Akash', 3), ('Angat', 5), ('Nepin', 9)]
except Exception:
    failed += 1
try:
    assert min_k([('tanmay', 14), ('Amer', 11), ('Ayesha', 9), ('SKD', 16)], 1) == [('Ayesha', 9)]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
