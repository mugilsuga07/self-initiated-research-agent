"""
Claim data models for evidence extraction.

STEP 5: Evidence Extraction
Represents atomic claims extracted from source content.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ClaimType(Enum):
    """Type of claim for categorization."""
    BENEFIT = "benefit"      # Positive outcomes, advantages
    RISK = "risk"            # Dangers, concerns, threats
    LIMITATION = "limitation" # Current constraints, gaps
    EXAMPLE = "example"      # Case study, real-world instance
    METRIC = "metric"        # Quantitative data, statistics
    PRACTICE = "practice"    # Best practice, recommendation
    FAILURE = "failure"      # Reported failure, problem
    UNKNOWN = "unknown"


@dataclass
class Claim:
    """
    Represents a single atomic claim extracted from a source.
    
    A claim is a specific, verifiable assertion that can
    support or challenge a decision.
    """
    text: str
    source_url: str
    source_title: str
    claim_type: ClaimType = ClaimType.UNKNOWN
    confidence: float = 0.8  # Extraction confidence
    
    def __str__(self) -> str:
        type_label = self.claim_type.value.upper()
        return f"[{type_label}] {self.text}"
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "claim_type": self.claim_type.value,
            "confidence": self.confidence,
        }
    
    @property
    def is_actionable(self) -> bool:
        """Check if claim is actionable (risk, limitation, practice, failure)."""
        return self.claim_type in [
            ClaimType.RISK,
            ClaimType.LIMITATION,
            ClaimType.PRACTICE,
            ClaimType.FAILURE,
        ]


@dataclass
class SourceClaims:
    """Claims extracted from a single source."""
    source_url: str
    source_title: str
    claims: list[Claim] = field(default_factory=list)
    extraction_error: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.claims)
    
    @property
    def actionable_count(self) -> int:
        return sum(1 for c in self.claims if c.is_actionable)


@dataclass
class EvidenceSummary:
    """Summary of all extracted evidence across sources."""
    all_claims: list[Claim] = field(default_factory=list)
    claims_by_source: list[SourceClaims] = field(default_factory=list)
    
    @property
    def total_claims(self) -> int:
        return len(self.all_claims)
    
    @property
    def sources_processed(self) -> int:
        return len(self.claims_by_source)
    
    @property
    def actionable_ratio(self) -> float:
        if not self.all_claims:
            return 0.0
        actionable = sum(1 for c in self.all_claims if c.is_actionable)
        return actionable / len(self.all_claims)
    
    def claims_by_type(self) -> dict[ClaimType, list[Claim]]:
        """Group claims by type."""
        result = {}
        for claim in self.all_claims:
            if claim.claim_type not in result:
                result[claim.claim_type] = []
            result[claim.claim_type].append(claim)
        return result
    
    def summary_stats(self) -> dict:
        """Get summary statistics."""
        by_type = self.claims_by_type()
        return {
            "total_claims": self.total_claims,
            "sources_processed": self.sources_processed,
            "actionable_ratio": self.actionable_ratio,
            "by_type": {t.value: len(claims) for t, claims in by_type.items()},
        }

