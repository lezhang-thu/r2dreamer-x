"""Counted anchor representatives and source prefixes for return exploration.

Similarity need not be transitive (or even reflexive). Representatives are
therefore built in source order, without merging clusters. The similarity
callback takes two NumPy arrays [N, D], [M, D] and returns probabilities [N, M].
Only the counters change after construction; new trajectories never add targets
or source prefixes to the current round.
"""

import numpy as np


class AnchorBank:
    def __init__(self, references, counts, prefixes, origins, similarity,
                 threshold=0.5, chunk_size=256):
        self.threshold = float(threshold)
        self.chunk_size = int(chunk_size)
        if not 0 <= self.threshold <= 1 or self.chunk_size < 1:
            raise ValueError('Invalid anchor similarity threshold or chunk size')
        self.similarity = similarity
        self.references = self._references(references).copy()
        if not len(self.references):
            raise ValueError('No anchors available for return exploration')
        counts = np.asarray(counts)
        if (counts.shape != (len(self.references),)
                or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 1)):
            raise ValueError('Every representative needs a positive integer counter')
        self.counts = counts.astype(np.int64, copy=True)
        self.prefixes = tuple(tuple((int(ep), int(pos)) for ep, pos in members)
                              for members in prefixes)
        self.origins = tuple((int(ep), int(pos)) for ep, pos in origins)
        if len(self.prefixes) != len(self) or len(self.origins) != len(self):
            raise ValueError('Representative, prefix, and origin counts differ')
        if any(not members or len(set(members)) != len(members)
               or origin not in members
               or any(pos < 0 for _, pos in members)
               for origin, members in zip(self.origins, self.prefixes)):
            raise ValueError('Each representative needs unique prefixes including its origin')
        self.references.setflags(write=False)

    def __len__(self):
        return len(self.references)

    @staticmethod
    def _references(values):
        values = np.asarray(values)
        if (values.ndim != 2 or values.shape[1] < 1
                or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all()):
            raise ValueError('Anchor embeddings must be a finite [N, D] array')
        return values

    @staticmethod
    def _matches(similarity, left, right, threshold):
        probabilities = np.asarray(similarity(left, right))
        if (probabilities.shape != (len(left), len(right))
                or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))):
            raise ValueError('Similarity must return finite probabilities with shape [N, M]')
        return probabilities >= threshold

    @classmethod
    def build(cls, demos, similarity, threshold=0.5, chunk_size=256):
        """Build A/counters in order, then index prefixes against the final A.

        Each demo provides an integer ``id`` and a ``references`` array in
        anchor order. A prefix reference is ``(episode_id, anchor_position)``;
        the associated goal sequence includes that position.
        """
        threshold, chunk_size = float(threshold), int(chunk_size)
        if not 0 <= threshold <= 1 or chunk_size < 1:
            raise ValueError('Invalid anchor similarity threshold or chunk size')
        sources, seen, dimension = [], set(), None
        for demo in demos:
            episode_id = int(demo['id'])
            if episode_id in seen:
                raise ValueError('Source episode IDs must be unique')
            seen.add(episode_id)
            refs = cls._references(demo['references'])
            if dimension is not None and refs.shape[1] != dimension:
                raise ValueError('Source anchor embedding dimensions differ')
            dimension = refs.shape[1]
            sources.append((episode_id, refs))

        representatives, counts, origins = [], [], []
        for episode_id, refs in sources:
            for position, reference in enumerate(refs):
                found = False
                for start in range(0, len(representatives), chunk_size):
                    right = np.stack(representatives[start:start + chunk_size])
                    matched = cls._matches(similarity, reference[None], right, threshold)[0]
                    for index in np.flatnonzero(matched):
                        counts[start + int(index)] += 1
                        found = True
                if not found:
                    representatives.append(reference.copy())
                    counts.append(1)
                    origins.append((episode_id, position))
        if not representatives:
            raise ValueError('No anchors available for return exploration')

        references = np.stack(representatives)
        prefixes = [[] for _ in representatives]
        origin_indices = {origin: index for index, origin in enumerate(origins)}
        # A representative added late can match an earlier nonrepresentative
        # occurrence. A second pass is essential when similarity is nontransitive.
        for episode_id, refs in sources:
            for start in range(0, len(refs), chunk_size):
                left = refs[start:start + chunk_size]
                for target_start in range(0, len(references), chunk_size):
                    right = references[target_start:target_start + chunk_size]
                    matches = cls._matches(similarity, left, right, threshold)
                    for row, position in enumerate(range(start, start + len(left))):
                        prefix = (episode_id, position)
                        origin_index = origin_indices.get(prefix)
                        if (origin_index is not None
                                and target_start <= origin_index < target_start + len(right)):
                            # Exact source identity supplies a valid prefix even
                            # if the learned classifier fails a self comparison.
                            # This affects membership only, never the counters.
                            matches[row, origin_index - target_start] = True
                        for column in np.flatnonzero(matches[row]):
                            prefixes[target_start + int(column)].append(prefix)
        return cls(references, counts, prefixes, origins, similarity,
                   threshold=threshold, chunk_size=chunk_size)

    def probabilities(self):
        """Inverse-count sampling favors less frequently observed representatives."""
        weights = 1.0 / self.counts.astype(np.float64)
        return weights / weights.sum()

    def sample(self, rng):
        """Sample a target by count, then uniformly choose a matching source prefix."""
        index = int(rng.choice(len(self), p=self.probabilities()))
        members = self.prefixes[index]
        return index, members[int(rng.integers(len(members)))]

    def update(self, references):
        """Count all matches from a completed rollout without extending A.

        Returns the increments for reporting. Unseen anchors contribute zero.
        Source-prefix memberships remain those of the round's original S.
        """
        references = self._references(references)
        if references.shape[1] != self.references.shape[1]:
            raise ValueError('Rollout and representative embedding dimensions differ')
        increments = np.zeros(len(self), dtype=np.int64)
        for start in range(0, len(references), self.chunk_size):
            left = references[start:start + self.chunk_size]
            for target_start in range(0, len(self), self.chunk_size):
                right = self.references[target_start:target_start + self.chunk_size]
                matches = self._matches(self.similarity, left, right, self.threshold)
                increments[target_start:target_start + len(right)] += matches.sum(axis=0)
        self.counts += increments
        return increments

    def state_dict(self):
        """JSON-serializable state; the classifier callback is supplied on load."""
        return dict(format_version=1, threshold=self.threshold, chunk_size=self.chunk_size,
                    references=self.references.tolist(), reference_dtype=str(self.references.dtype),
                    counts=self.counts.tolist(),
                    prefixes=[[list(prefix) for prefix in members] for members in self.prefixes],
                    origins=[list(origin) for origin in self.origins])

    @classmethod
    def from_state_dict(cls, state, similarity):
        if state.get('format_version') != 1:
            raise ValueError('Unsupported anchor bank format')
        references = np.asarray(state['references'], dtype=state['reference_dtype'])
        return cls(references, state['counts'], state['prefixes'], state['origins'],
                   similarity, threshold=state['threshold'], chunk_size=state['chunk_size'])
