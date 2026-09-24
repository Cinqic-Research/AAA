"""Versioned promotion decision contracts.

``aaa.promotion.crossed.v1`` is the prospective successor to the ``aaa.1k.v2``
K5 path (``AAA-180``): a frozen specification (:mod:`.contract`), a primary
index-resampling evaluator (:mod:`.primary`), a structurally independent
count-weighted recomputation (:mod:`.independent`) and fail-closed
adjudication (:mod:`.adjudicate`). The frozen v2 path itself is unchanged and
listed in :data:`.contract.FORBIDDEN_CONTRACTS`.
"""

from .adjudicate import adjudicate
from .contract import (
    CONTRACT_ID,
    FORBIDDEN_CONTRACTS,
    Contract,
    ContractError,
    Criterion,
    GroupDeclaration,
    PrimitiveError,
    require_admissible,
)

__all__ = [
    "CONTRACT_ID",
    "FORBIDDEN_CONTRACTS",
    "Contract",
    "ContractError",
    "Criterion",
    "GroupDeclaration",
    "PrimitiveError",
    "adjudicate",
    "require_admissible",
]
