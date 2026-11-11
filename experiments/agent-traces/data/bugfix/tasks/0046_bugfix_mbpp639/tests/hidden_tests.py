import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert sample_nam(['sally', 'Dylan', 'rebecca', 'Diana', 'Joanne', 'keith'])==16
except Exception:
    failed += 1
try:
    assert sample_nam(["php", "res", "Python", "abcd", "Java", "aaa"])==10
except Exception:
    failed += 1
try:
    assert sample_nam(["abcd", "Python", "abba", "aba"])==6
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
