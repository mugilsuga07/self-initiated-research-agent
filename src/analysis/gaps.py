"""
Gap Analysis

STEP 7: Gap Analysis (Unknowns & Conflicts)
Identifies gaps in the evidence for decision-making.

Detects:
- Unknowns: Missing data, unanswered questions
- Conflicts: Sources that disagree
- Assumptions: Implicit assumptions in the evidence
"""

from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import LLMClient
from src.models.claim import Claim, EvidenceSummary


@dataclass
class Unknown:
    """A gap in knowledge - something we don't know."""
    description: str
    importance: str  # "high", "medium", "low"
    related_claims: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"[{self.importance.upper()}] {self.description}"


@dataclass
class Conflict:
    """A disagreement between sources."""
    description: str
    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    
    def __str__(self) -> str:
        return f"CONFLICT: {self.description}"


@dataclass
class Assumption:
    """An implicit assumption in the evidence."""
    description: str
    risk: str  # What could go wrong if assumption is false
    
    def __str__(self) -> str:
        return f"ASSUMES: {self.description}"


@dataclass
class GapAnalysisResult:
    """Complete gap analysis for the research."""
    original_question: str
    unknowns: list[Unknown] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    
    @property
    def total_gaps(self) -> int:
        return len(self.unknowns) + len(self.conflicts) + len(self.assumptions)
    
    @property
    def has_critical_gaps(self) -> bool:
        """Check if there are high-importance unknowns."""
        return any(u.importance == "high" for u in self.unknowns)
    
    def summary(self) -> dict:
        return {
            "unknowns": len(self.unknowns),
            "conflicts": len(self.conflicts),
            "assumptions": len(self.assumptions),
            "total_gaps": self.total_gaps,
            "has_critical": self.has_critical_gaps,
        }


# System prompt for gap analysis
GAP_ANALYSIS_SYSTEM_PROMPT = """You are a critical analysis assistant. Your job is to identify gaps, conflicts, and assumptions in research evidence.

You help decision-makers understand:
1. What we DON'T know (unknowns/gaps)
2. Where sources DISAGREE (conflicts)
3. What is being ASSUMED without evidence (assumptions)

Rules:
- Be specific and actionable
- Every decision has uncertainty - always find at least 3 unknowns
- Look for implicit assumptions that could change the decision
- Identify when sources contradict each other
- Focus on gaps that would affect a real decision

For UNKNOWNS, consider:
- Missing long-term data
- Lack of cost/ROI information
- Unclear failure rates or reliability metrics
- Missing information about edge cases
- Gaps in specific industry/context applicability

For CONFLICTS, look for:
- Sources claiming opposite outcomes
- Disagreement on best practices
- Contradictory statistics or metrics

For ASSUMPTIONS, identify:
- Implicit context (e.g., "assumes enterprise scale")
- Hidden prerequisites (e.g., "assumes skilled team")
- Unstated conditions for success

Output JSON format:
{
  "unknowns": [
    {"description": "...", "importance": "high|medium|low"}
  ],
  "conflicts": [
    {"description": "...", "claim_a": "...", "claim_b": "...", "source_a": "...", "source_b": "..."}
  ],
  "assumptions": [
    {"description": "...", "risk": "what could go wrong if false"}
  ]
}"""


GAP_ANALYSIS_PROMPT_TEMPLATE = """Analyze this research evidence for gaps, conflicts, and assumptions.

ORIGINAL QUESTION: {question}

CLAIMS EXTRACTED FROM SOURCES:
{claims_text}

---

Identify:
1. UNKNOWNS: What important information is MISSING? What questions remain unanswered? (at least 3)
2. CONFLICTS: Do any sources DISAGREE with each other? (identify if present)
3. ASSUMPTIONS: What is being ASSUMED without explicit evidence? (at least 2)

Think like a skeptical engineer reviewing this for a design doc.

Return JSON with unknowns, conflicts, and assumptions."""


class GapDetector:
    """
    Analyzes evidence to identify gaps, conflicts, and assumptions.
    
    Uses LLM to critically analyze the collected claims and
    identify what's missing or uncertain.
    """
    
    MIN_UNKNOWNS = 3
    MIN_ASSUMPTIONS = 2
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    def analyze(
        self, 
        question: str, 
        evidence: EvidenceSummary
    ) -> GapAnalysisResult:
        """
        Analyze evidence for gaps.
        
        Args:
            question: The original research question
            evidence: Summary of all extracted claims
            
        Returns:
            GapAnalysisResult with unknowns, conflicts, assumptions
        """
        # Format claims for analysis
        claims_text = self._format_claims(evidence)
        
        if not claims_text:
            return self._empty_result(question)
        
        try:
            # Build prompt
            prompt = GAP_ANALYSIS_PROMPT_TEMPLATE.format(
                question=question,
                claims_text=claims_text,
            )
            
            # Call LLM
            response = self.llm.complete_json(
                prompt=prompt,
                system_prompt=GAP_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.4,
            )
            
            # Parse response
            return self._parse_response(question, response)
            
        except Exception as e:
            # Return minimal result on error
            return GapAnalysisResult(
                original_question=question,
                unknowns=[Unknown(
                    description=f"Analysis failed: {str(e)[:50]}",
                    importance="high"
                )],
            )
    
    def _format_claims(self, evidence: EvidenceSummary) -> str:
        """Format claims for LLM analysis."""
        lines = []
        
        for i, claim in enumerate(evidence.all_claims[:30], 1):  # Limit to 30 claims
            source_short = claim.source_title[:30] + "..." if len(claim.source_title) > 30 else claim.source_title
            lines.append(f"{i}. [{claim.claim_type.value.upper()}] {claim.text}")
            lines.append(f"   Source: {source_short}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_response(self, question: str, response: dict) -> GapAnalysisResult:
        """Parse LLM response into GapAnalysisResult."""
        unknowns = []
        conflicts = []
        assumptions = []
        
        # Parse unknowns
        for item in response.get("unknowns", []):
            if isinstance(item, dict) and item.get("description"):
                unknowns.append(Unknown(
                    description=item["description"],
                    importance=item.get("importance", "medium"),
                ))
        
        # Parse conflicts
        for item in response.get("conflicts", []):
            if isinstance(item, dict) and item.get("description"):
                conflicts.append(Conflict(
                    description=item["description"],
                    claim_a=item.get("claim_a", ""),
                    claim_b=item.get("claim_b", ""),
                    source_a=item.get("source_a", ""),
                    source_b=item.get("source_b", ""),
                ))
        
        # Parse assumptions
        for item in response.get("assumptions", []):
            if isinstance(item, dict) and item.get("description"):
                assumptions.append(Assumption(
                    description=item["description"],
                    risk=item.get("risk", "Unknown risk"),
                ))
        
        return GapAnalysisResult(
            original_question=question,
            unknowns=unknowns,
            conflicts=conflicts,
            assumptions=assumptions,
        )
    
    def _empty_result(self, question: str) -> GapAnalysisResult:
        """Return result when no claims are available."""
        return GapAnalysisResult(
            original_question=question,
            unknowns=[
                Unknown(
                    description="Insufficient evidence gathered to analyze",
                    importance="high"
                ),
                Unknown(
                    description="No sources could be processed for claims",
                    importance="high"
                ),
                Unknown(
                    description="Research may need to be repeated with different sources",
                    importance="medium"
                ),
            ],
        )

