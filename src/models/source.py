"""
Source data models for research results.

STEP 3: Autonomous Discovery
Represents sources found during web search.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class SourceType(Enum):
    """Type of source for credibility scoring."""
    NEWS = "news"
    BLOG = "blog"
    RESEARCH = "research"
    OFFICIAL = "official"  # Company blogs, official docs
    SOCIAL = "social"
    UNKNOWN = "unknown"


@dataclass
class Source:
    """
    Represents a single source found during search.
    
    Contains metadata needed for:
    - Display to user
    - Content extraction (Step 4)
    - Credibility ranking (Step 6)
    """
    url: str
    title: str
    snippet: str
    sub_question: str  # Which sub-question this source answers
    
    # Optional metadata
    published_date: Optional[datetime] = None
    domain: str = ""
    source_type: SourceType = SourceType.UNKNOWN
    
    # Populated in later steps
    content: str = ""  # Full extracted content (Step 4)
    claims: list = field(default_factory=list)  # Extracted claims (Step 5)
    credibility_score: float = 0.0  # Ranking score (Step 6)
    
    def __post_init__(self):
        """Extract domain from URL if not provided."""
        if not self.domain and self.url:
            self.domain = self._extract_domain(self.url)
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""
    
    def __str__(self) -> str:
        date_str = ""
        if self.published_date:
            date_str = f" ({self.published_date.strftime('%Y-%m-%d')})"
        return f"{self.title}{date_str}\n  {self.url}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "domain": self.domain,
            "sub_question": self.sub_question,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "source_type": self.source_type.value,
        }


@dataclass
class SearchResults:
    """
    Collection of sources for all sub-questions.
    """
    sub_question: str
    sources: list[Source] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.sources)
    
    def __str__(self) -> str:
        return f"{self.sub_question[:50]}... → {len(self.sources)} sources"


@dataclass
class DiscoveryResults:
    """
    Complete discovery results across all sub-questions.
    """
    results_by_question: dict[str, SearchResults] = field(default_factory=dict)
    all_sources: list[Source] = field(default_factory=list)
    
    @property
    def total_sources(self) -> int:
        return len(self.all_sources)
    
    @property
    def unique_domains(self) -> set[str]:
        return {s.domain for s in self.all_sources}
    
    def summary(self) -> str:
        """Generate a summary of discovery results."""
        lines = [f"Total sources: {self.total_sources}"]
        lines.append(f"Unique domains: {len(self.unique_domains)}")
        lines.append("")
        for sq, results in self.results_by_question.items():
            lines.append(f"• {sq[:60]}...")
            lines.append(f"  → {len(results)} sources")
        return "\n".join(lines)

