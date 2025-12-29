"""
Claim Extraction

STEP 5: Evidence Extraction
Extracts atomic claims from source content using LLM.

Features:
- Extracts 3-7 specific claims per source
- Categorizes claims by type (risk, benefit, limitation, etc.)
- Filters out generic/fluffy statements
- Ensures actionable claims are present
"""

from typing import Optional
from src.llm.client import LLMClient
from src.models.source import Source
from src.models.claim import Claim, ClaimType, SourceClaims, EvidenceSummary


# System prompt for claim extraction
EXTRACTION_SYSTEM_PROMPT = """You are an evidence extraction assistant. Your job is to extract specific, atomic claims from article content.

Rules:
1. Extract 3-7 claims per article
2. Each claim must be a specific, concrete assertion
3. Claims must be attributable to the source (things the article actually states)
4. Categorize each claim as one of: benefit, risk, limitation, example, metric, practice, failure

AVOID extracting:
- Generic statements like "AI is transforming industries"
- Vague claims like "companies are adopting AI"
- Marketing fluff or hype
- Opinions without evidence

PREFER extracting:
- Specific failures or problems reported
- Quantitative data or metrics
- Concrete examples or case studies
- Specific risks or limitations mentioned
- Best practices or recommendations with reasoning

At least 1 claim should be about risks, limitations, or failures.

Output format: JSON with key "claims" containing array of objects:
{
  "claims": [
    {"text": "claim text here", "type": "risk|benefit|limitation|example|metric|practice|failure"}
  ]
}"""


EXTRACTION_PROMPT_TEMPLATE = """Extract 3-7 specific, evidence-based claims from this article content.

ARTICLE TITLE: {title}
ARTICLE URL: {url}

CONTENT:
{content}

---

Extract claims that would help someone DECIDE whether to adopt or trust this topic.
Focus on: specific evidence, reported problems, quantitative data, concrete examples.
Avoid: generic statements, marketing language, vague assertions.

Return JSON: {{"claims": [{{"text": "...", "type": "risk|benefit|limitation|example|metric|practice|failure"}}]}}"""


# Patterns that indicate generic/fluffy claims
GENERIC_PATTERNS = [
    "is transforming",
    "is revolutionizing",
    "is changing the world",
    "has the potential",
    "is becoming increasingly",
    "is gaining traction",
    "is on the rise",
    "is here to stay",
    "is the future",
    "companies are adopting",
    "organizations are using",
    "the industry is moving",
]


class ClaimExtractor:
    """
    Extracts evidence claims from source content using LLM.
    
    Ensures claims are:
    - Specific and concrete
    - Properly categorized
    - Not generic fluff
    - Include actionable insights
    """
    
    MIN_CLAIMS = 3
    MAX_CLAIMS = 7
    MIN_CLAIM_LENGTH = 20
    MAX_CONTENT_FOR_LLM = 8000  # Limit content to avoid token overflow
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    def extract_from_source(self, source: Source) -> SourceClaims:
        """
        Extract claims from a single source.
        
        Args:
            source: Source with content to extract from
            
        Returns:
            SourceClaims with extracted claims
        """
        # Check if source has content
        content = source.content
        if not content or len(content) < 100:
            return SourceClaims(
                source_url=source.url,
                source_title=source.title,
                claims=[],
                extraction_error="Insufficient content",
            )
        
        # Truncate content if too long
        if len(content) > self.MAX_CONTENT_FOR_LLM:
            content = content[:self.MAX_CONTENT_FOR_LLM] + "..."
        
        try:
            # Build prompt
            prompt = EXTRACTION_PROMPT_TEMPLATE.format(
                title=source.title,
                url=source.url,
                content=content,
            )
            
            # Call LLM
            response = self.llm.complete_json(
                prompt=prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                temperature=0.3,  # Lower temperature for more factual extraction
            )
            
            # Parse claims
            raw_claims = response.get("claims", [])
            claims = self._parse_claims(raw_claims, source)
            
            # Filter and validate
            claims = self._filter_claims(claims)
            
            return SourceClaims(
                source_url=source.url,
                source_title=source.title,
                claims=claims,
            )
            
        except Exception as e:
            return SourceClaims(
                source_url=source.url,
                source_title=source.title,
                claims=[],
                extraction_error=str(e)[:100],
            )
    
    def _parse_claims(self, raw_claims: list, source: Source) -> list[Claim]:
        """Parse raw claim data into Claim objects."""
        claims = []
        
        for raw in raw_claims:
            if not isinstance(raw, dict):
                continue
            
            text = raw.get("text", "").strip()
            type_str = raw.get("type", "unknown").lower()
            
            # Skip empty claims
            if not text or len(text) < self.MIN_CLAIM_LENGTH:
                continue
            
            # Parse claim type
            claim_type = self._parse_claim_type(type_str)
            
            claims.append(Claim(
                text=text,
                source_url=source.url,
                source_title=source.title,
                claim_type=claim_type,
            ))
        
        return claims
    
    def _parse_claim_type(self, type_str: str) -> ClaimType:
        """Parse claim type string to enum."""
        type_map = {
            "benefit": ClaimType.BENEFIT,
            "risk": ClaimType.RISK,
            "limitation": ClaimType.LIMITATION,
            "example": ClaimType.EXAMPLE,
            "metric": ClaimType.METRIC,
            "practice": ClaimType.PRACTICE,
            "failure": ClaimType.FAILURE,
        }
        return type_map.get(type_str, ClaimType.UNKNOWN)
    
    def _filter_claims(self, claims: list[Claim]) -> list[Claim]:
        """Filter out generic/fluffy claims."""
        filtered = []
        
        for claim in claims:
            text_lower = claim.text.lower()
            
            # Check for generic patterns
            is_generic = any(pattern in text_lower for pattern in GENERIC_PATTERNS)
            
            if not is_generic:
                filtered.append(claim)
        
        # Limit to max claims
        return filtered[:self.MAX_CLAIMS]
    
    def extract_all(self, sources: list[Source]) -> EvidenceSummary:
        """
        Extract claims from all sources.
        
        Args:
            sources: List of sources with content
            
        Returns:
            EvidenceSummary with all extracted claims
        """
        all_claims = []
        claims_by_source = []
        
        for source in sources:
            # Skip sources without content
            if not source.content:
                continue
            
            source_claims = self.extract_from_source(source)
            claims_by_source.append(source_claims)
            all_claims.extend(source_claims.claims)
        
        return EvidenceSummary(
            all_claims=all_claims,
            claims_by_source=claims_by_source,
        )

