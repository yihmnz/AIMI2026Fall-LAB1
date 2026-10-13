"""Filename-derived group split. IDs are proxies, not verified patient metadata."""
import random
import re
from pathlib import Path
from collections import defaultdict


def group_id(path):
    name = Path(path).name
    match = re.match(r'(person\d+)_', name, re.I)
    if match:
        return match[1].lower()
    match = re.match(r'(NORMAL2-IM-\d+|IM-\d+)-', name, re.I)
    if match:
        return match[1].upper()
    raise ValueError(f'Unknown filename grouping pattern: {name}; provide verified metadata.')


def group_split(samples, fraction, seed):
    groups = defaultdict(list)
    for sample in samples:
        groups[group_id(sample[0])].append(sample)
    for gid, items in groups.items():
        if len({label for _, label in items}) != 1:
            raise ValueError(f'Group {gid} has conflicting class labels')
    rng = random.Random(seed)
    chosen = set()
    for label in (0, 1):
        ids = sorted(gid for gid, items in groups.items() if items[0][1] == label)
        if len(ids) < 2:
            raise ValueError('Need at least two groups per class')
        rng.shuffle(ids)
        target = sum(len(groups[gid]) for gid in ids) * fraction
        # Choose a seeded random prefix whose image count is nearest the target.
        cumulative = 0
        candidates = []
        for n, gid in enumerate(ids[:-1], 1):
            cumulative += len(groups[gid])
            candidates.append((abs(cumulative - target), n))
        n = min(candidates)[1]
        chosen.update(ids[:n])
    train = [s for s in samples if group_id(s[0]) not in chosen]
    val = [s for s in samples if group_id(s[0]) in chosen]
    assert {group_id(p) for p, _ in train}.isdisjoint({group_id(p) for p, _ in val})
    return train, val


def exclude_test_groups(train, test):
    test_ids = {group_id(p) for p, _ in test}
    excluded = [s for s in train if group_id(s[0]) in test_ids]
    kept = [s for s in train if group_id(s[0]) not in test_ids]
    return kept, excluded
