import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert rgb_to_hsv(255, 255, 255)==(0, 0.0, 100.0)
except Exception:
    failed += 1
try:
    assert rgb_to_hsv(0, 215, 0)==(120.0, 100.0, 84.31372549019608)
except Exception:
    failed += 1
try:
    assert rgb_to_hsv(10, 215, 110)==(149.26829268292684, 95.34883720930233, 84.31372549019608)
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
