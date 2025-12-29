"""
Source Ranking

STEP 6: Intelligent Ranking
Scores and ranks sources by quality for decision-making.

Scoring dimensions:
- Recency: Prefer recent sources (last 12-18 months)
- Credibility: Prefer reputable domains
- Evidence richness: Prefer sources with metrics/examples
- Claim density: Prefer sources with more actionable claims
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from src.models.source import Source
from src.models.claim import Claim, ClaimType


# Credibility tiers for domains
TIER_1_DOMAINS = [
    # Major tech companies engineering blogs
    "engineering.fb.com", "engineering.linkedin.com", "netflixtechblog.com",
    "uber.com", "blog.google", "aws.amazon.com", "cloud.google.com",
    "azure.microsoft.com", "openai.com", "anthropic.com", "deepmind.com",
    "stripe.com", "shopify.engineering", "github.blog", "slack.engineering",
    # Research
    "arxiv.org", "acm.org", "ieee.org", "nature.com", "science.org",
    # Tier 1 media
    "nytimes.com", "wsj.com", "economist.com", "ft.com",
    # Tier 1 analysts
    "gartner.com", "mckinsey.com", "hbr.org", "forrester.com",
]

TIER_2_DOMAINS = [
    # Tech media
    "techcrunch.com", "wired.com", "arstechnica.com", "theverge.com",
    "thenewstack.io", "infoq.com", "zdnet.com", "venturebeat.com",
    # Developer focused
    "stackoverflow.blog", "dev.to", "hackernoon.com", "dzone.com",
    # Company blogs (general)
    "medium.com",  # Mixed quality but often has good engineering posts
]

LOW_CREDIBILITY_PATTERNS = [
    "top 10", "top 5", "best of",
    "beginners", "101", "ultimate guide",
    "seo", "marketing", "affiliate",
]


@dataclass
class SourceScore:
    """Detailed scoring breakdown for a source."""
    source: Source
    
    # Individual scores (0.0 to 1.0)
    recency_score: float = 0.0
    credibility_score: float = 0.0
    evidence_score: float = 0.0
    claim_score: float = 0.0
    
    # Overall score (weighted combination)
    total_score: float = 0.0
    
    # Ranking metadata
    rank: int = 0
    justification: str = ""
    
    # Associated claims
    claims: list[Claim] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"#{self.rank} ({self.total_score:.2f}) {self.source.title[:50]}..."


@dataclass
class RankingResult:
    """Result of source ranking."""
    ranked_sources: list[SourceScore]
    top_sources_justification: str
    
    @property
    def top_3(self) -> list[SourceScore]:
        return self.ranked_sources[:3]
    
    @property
    def total_sources(self) -> int:
        return len(self.ranked_sources)


class SourceRanker:
    """
    Ranks sources by quality for decision-making.
    
    Scoring weights:
    - Recency: 25% (prefer recent, but don't penalize foundational sources too much)
    - Credibility: 35% (domain reputation is important)
    - Evidence richness: 25% (metrics, examples, case studies)
    - Claim density: 15% (actionable insights per source)
    """
    
    # Scoring weights
    WEIGHT_RECENCY = 0.25
    WEIGHT_CREDIBILITY = 0.35
    WEIGHT_EVIDENCE = 0.25
    WEIGHT_CLAIMS = 0.15
    
    # Recency parameters
    OPTIMAL_AGE_MONTHS = 6  # Sources from last 6 months get full score
    MAX_AGE_MONTHS = 24     # Sources older than 24 months get minimum score
    
    def __init__(self):
        self.tier1_domains = set(TIER_1_DOMAINS)
        self.tier2_domains = set(TIER_2_DOMAINS)
    
    def rank_sources(
        self, 
        sources: list[Source], 
        claims_by_source: dict[str, list[Claim]] = None
    ) -> RankingResult:
        """
        Rank sources by quality.
        
        Args:
            sources: List of sources to rank
            claims_by_source: Optional dict mapping source URL to claims
            
        Returns:
            RankingResult with ranked sources and justification
        """
        claims_by_source = claims_by_source or {}
        
        # Score each source
        scored_sources = []
        for source in sources:
            claims = claims_by_source.get(source.url, [])
            score = self._score_source(source, claims)
            scored_sources.append(score)
        
        # Sort by total score (descending)
        scored_sources.sort(key=lambda s: s.total_score, reverse=True)
        
        # Assign ranks
        for i, score in enumerate(scored_sources):
            score.rank = i + 1
        
        # Generate justification for top sources
        justification = self._generate_justification(scored_sources[:3])
        
        return RankingResult(
            ranked_sources=scored_sources,
            top_sources_justification=justification,
        )
    
    def _score_source(self, source: Source, claims: list[Claim]) -> SourceScore:
        """Calculate scores for a single source."""
        recency = self._score_recency(source)
        credibility = self._score_credibility(source)
        evidence = self._score_evidence(source, claims)
        claim_score = self._score_claims(claims)
        
        # Weighted total
        total = (
            recency * self.WEIGHT_RECENCY +
            credibility * self.WEIGHT_CREDIBILITY +
            evidence * self.WEIGHT_EVIDENCE +
            claim_score * self.WEIGHT_CLAIMS
        )
        
        # Generate justification
        justification = self._source_justification(
            recency, credibility, evidence, claim_score, source
        )
        
        return SourceScore(
            source=source,
            recency_score=recency,
            credibility_score=credibility,
            evidence_score=evidence,
            claim_score=claim_score,
            total_score=total,
            justification=justification,
            claims=claims,
        )
    
    def _score_recency(self, source: Source) -> float:
        """Score based on publication date recency."""
        if not source.published_date:
            return 0.5  # Unknown date gets middle score
        
        now = datetime.now()
        age = now - source.published_date
        age_months = age.days / 30
        
        if age_months <= self.OPTIMAL_AGE_MONTHS:
            return 1.0
        elif age_months >= self.MAX_AGE_MONTHS:
            return 0.2
        else:
            # Linear interpolation between optimal and max
            range_months = self.MAX_AGE_MONTHS - self.OPTIMAL_AGE_MONTHS
            position = (age_months - self.OPTIMAL_AGE_MONTHS) / range_months
            return 1.0 - (position * 0.8)  # Scale from 1.0 to 0.2
    
    def _score_credibility(self, source: Source) -> float:
        """Score based on domain reputation."""
        domain = source.domain.lower()
        url_lower = source.url.lower()
        title_lower = source.title.lower()
        
        # Check for low credibility patterns in title
        for pattern in LOW_CREDIBILITY_PATTERNS:
            if pattern in title_lower:
                return 0.2
        
        # Check tier 1 domains
        for tier1 in self.tier1_domains:
            if tier1 in domain or tier1 in url_lower:
                return 1.0
        
        # Check tier 2 domains
        for tier2 in self.tier2_domains:
            if tier2 in domain or tier2 in url_lower:
                return 0.7
        
        # Check for engineering blog patterns
        if "engineering" in url_lower or "techblog" in url_lower:
            return 0.8
        
        # Check for blog patterns (medium credibility)
        if "blog" in url_lower:
            return 0.5
        
        # Unknown domain
        return 0.4
    
    def _score_evidence(self, source: Source, claims: list[Claim]) -> float:
        """Score based on evidence richness (metrics, examples)."""
        score = 0.0
        
        # Check claim types
        has_metric = any(c.claim_type == ClaimType.METRIC for c in claims)
        has_example = any(c.claim_type == ClaimType.EXAMPLE for c in claims)
        has_failure = any(c.claim_type == ClaimType.FAILURE for c in claims)
        
        if has_metric:
            score += 0.4
        if has_example:
            score += 0.3
        if has_failure:
            score += 0.3
        
        # Also check content for evidence patterns
        content = (source.content or "").lower()
        
        # Numbers/percentages suggest metrics
        import re
        if re.search(r'\d+%|\d+\s*percent', content):
            score += 0.1
        
        # Case study mentions
        if "case study" in content or "real-world" in content:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _score_claims(self, claims: list[Claim]) -> float:
        """Score based on claim quantity and quality."""
        if not claims:
            return 0.3  # No claims gets low score
        
        # Base score from claim count (3-7 is ideal)
        count = len(claims)
        if count >= 5:
            count_score = 1.0
        elif count >= 3:
            count_score = 0.8
        else:
            count_score = 0.5
        
        # Bonus for actionable claims
        actionable = sum(1 for c in claims if c.is_actionable)
        actionable_ratio = actionable / len(claims) if claims else 0
        
        return count_score * 0.6 + actionable_ratio * 0.4
    
    def _source_justification(
        self, 
        recency: float, 
        credibility: float, 
        evidence: float, 
        claims: float,
        source: Source
    ) -> str:
        """Generate a brief justification for a source's ranking."""
        reasons = []
        
        if credibility >= 0.8:
            reasons.append("reputable source")
        elif credibility >= 0.6:
            reasons.append("known outlet")
        
        if recency >= 0.8:
            reasons.append("recent")
        elif recency <= 0.3:
            reasons.append("older source")
        
        if evidence >= 0.7:
            reasons.append("data-rich")
        
        if claims >= 0.7:
            reasons.append("many insights")
        
        if not reasons:
            reasons.append("general coverage")
        
        return ", ".join(reasons)
    
    def _generate_justification(self, top_sources: list[SourceScore]) -> str:
        """Generate justification for why top sources ranked high."""
        if not top_sources:
            return "No sources to rank."
        
        lines = ["Top sources ranked by:"]
        
        for score in top_sources:
            domain = score.source.domain
            reasons = score.justification
            lines.append(f"  • {domain}: {reasons}")
        
        return "\n".join(lines)

