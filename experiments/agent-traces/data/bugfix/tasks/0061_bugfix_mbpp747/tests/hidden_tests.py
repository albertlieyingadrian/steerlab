import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert lcs_of_three('AGGT12', '12TXAYB', '12XBA') == 2
except Exception:
    failed += 1
try:
    assert lcs_of_three('Reels', 'Reelsfor', 'ReelsforReels') == 5
except Exception:
    failed += 1
try:
    assert lcs_of_three('abcd1e2', 'bc12ea', 'bd1ea') == 3
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
