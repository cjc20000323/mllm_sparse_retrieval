"""Finite DSPy aspect selection, independent of DSPy, Torch and model services."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
from typing import Optional


def _words(values, label):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f'{label} must be a list of strings.')
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f'{label} contains an empty or non-string aspect.')
    words = tuple(value.strip().lower() for value in values)
    if len(words) != len(set(words)):
        raise ValueError(f'{label} contains duplicate normalized aspects.')
    return words


@dataclass(frozen=True)
class DspyAspectPool:
    aspects: tuple[str, ...]
    initial_aspects: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        words = _words(self.aspects, 'aspects')
        if len(words) < 3:
            raise ValueError('The aspect pool must contain at least 3 different aspects.')
        object.__setattr__(self, 'aspects', words)
        initial = self.initial_aspects if self.initial_aspects is not None else words[:5]
        object.__setattr__(self, 'initial_aspects', tuple(self.validate(initial)))

    @classmethod
    def from_file(cls, path):
        payload = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or 'aspects' not in payload:
            raise ValueError('Aspect pool JSON must be an object containing aspects.')
        if 'initial_aspects' in payload and payload['initial_aspects'] is None:
            raise ValueError('initial_aspects must be a list when provided.')
        return cls(payload['aspects'], payload.get('initial_aspects'))

    @property
    def max_aspects(self):
        return min(6, len(self.aspects))

    @property
    def capacity(self):
        return sum(comb(len(self.aspects), size) for size in range(3, self.max_aspects + 1))

    def validate(self, candidate):
        # Check raw length/duplicates BEFORE any truncation or deduplication.
        words = _words(candidate, 'candidate')
        if not 3 <= len(words) <= self.max_aspects:
            raise ValueError(f'Pool candidates must contain 3 to {self.max_aspects} aspects.')
        unknown = set(words) - set(self.aspects)
        if unknown:
            raise ValueError(f'Aspects outside the configured pool: {sorted(unknown)}')
        chosen = set(words)
        return [word for word in self.aspects if word in chosen]

    def next_unused(self, forbidden):
        used = {frozenset(key) for key in forbidden}
        for size in range(3, self.max_aspects + 1):
            for candidate in combinations(self.aspects, size):
                if frozenset(candidate) not in used:
                    return list(candidate)
        raise ValueError('All legal aspect pool combinations have been used.')

    def identity(self):
        payload = {'mode': 'pool', 'version': 1, 'aspects': list(self.aspects),
                   'initial_aspects': list(self.initial_aspects), 'min_aspects': 3,
                   'max_aspects': self.max_aspects}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return {**payload, 'fingerprint': digest}

    def constraint(self):
        return (
            f'Mandatory closed aspect pool: {json.dumps(self.aspects, ensure_ascii=False)}. '
            f'Output exactly 3 to {self.max_aspects} different entries copied from this pool. '
            'Do not invent words, paraphrase, combine entries, add suffixes, or repeat a complete '
            'previously evaluated set. Preserve the pool file order in the output. '
            'Optimize the choice of subset, not the wording of the entries.'
        )


@dataclass
class DspyAspectSelectionArguments:
    dspy_aspect_mode: str = field(default='free', metadata={'choices': ['free', 'pool']})
    dspy_aspect_pool_file: Optional[str] = None

    def load_pool(self):
        if self.dspy_aspect_mode == 'free':
            return None
        if self.dspy_aspect_mode != 'pool':
            raise ValueError('dspy_aspect_mode must be free or pool.')
        if not self.dspy_aspect_pool_file:
            raise ValueError('pool mode requires --dspy_aspect_pool_file.')
        return DspyAspectPool.from_file(self.dspy_aspect_pool_file)
