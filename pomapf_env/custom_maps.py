from pathlib import Path

import yaml

with open(Path(__file__).parent / "maps.yaml", "r") as f:
    maps = yaml.safe_load(f)

maps = {**maps}

MAPS_REGISTRY = maps
_test_regexp = '(wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)'
