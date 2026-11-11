"""Who owns what, before and after a merger."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def ownership_matrix(firm_ids: Sequence) -> np.ndarray:
    """Omega[j, k] = 1 if products j and k have the same owner, else 0.

    This matrix is the only place where market structure enters the pricing model: a merger
    changes some zeros into ones, and nothing else.
    """
    ids = np.asarray(firm_ids)
    return (ids[:, None] == ids[None, :]).astype(float)


@dataclass(frozen=True)
class Merger:
    """A hypothetical merger of two or more firms.

    Why a class: the screening, simulation and reporting code all need the same three answers
    (which products are involved, who owns each product afterwards, what to call the deal), and
    keeping them in one small immutable object means the merger is defined once per script.

    `firms` holds the firm IDs that combine; after the merger every product of these firms is
    owned by the first ID in the tuple.
    """

    name: str
    firms: tuple

    def __post_init__(self):
        if len(self.firms) < 2:
            raise ValueError("a merger needs at least two firms")

    def involves(self, firm_ids: Sequence) -> np.ndarray:
        """Boolean mask: is each product owned by one of the merging firms?"""
        return np.isin(np.asarray(firm_ids), self.firms)

    def post_firm_ids(self, firm_ids: Sequence) -> np.ndarray:
        """Firm IDs after the merger: the merging firms' products all move to `firms[0]`."""
        ids = np.asarray(firm_ids).copy()
        ids[np.isin(ids, self.firms)] = self.firms[0]
        return ids

    def ownership(self, firm_ids: Sequence, post: bool = True) -> np.ndarray:
        """Ownership matrix before (`post=False`) or after (`post=True`) the merger."""
        return ownership_matrix(self.post_firm_ids(firm_ids) if post else firm_ids)
