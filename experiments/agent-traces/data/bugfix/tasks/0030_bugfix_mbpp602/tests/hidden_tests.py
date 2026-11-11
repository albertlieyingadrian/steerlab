import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert first_repeated_char("abcabc") == "a"
except Exception:
    failed += 1
try:
    assert first_repeated_char("abc") == None
except Exception:
    failed += 1
try:
    assert first_repeated_char("123123") == "1"
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
