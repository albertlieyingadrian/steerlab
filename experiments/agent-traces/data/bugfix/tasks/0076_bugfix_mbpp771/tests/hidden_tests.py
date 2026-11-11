import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert check_expression("{()}[{}]") == True
except Exception:
    failed += 1
try:
    assert check_expression("{()}[{]") == False
except Exception:
    failed += 1
try:
    assert check_expression("{()}[{}][]({})") == True
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
