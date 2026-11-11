import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert find_adverbs("Clearly, he has no excuse for such behavior.") == '0-7: Clearly'
except Exception:
    failed += 1
try:
    assert find_adverbs("Please handle the situation carefuly") == '28-36: carefuly'
except Exception:
    failed += 1
try:
    assert find_adverbs("Complete the task quickly") == '18-25: quickly'
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
