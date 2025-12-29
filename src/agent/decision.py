"""
Decision Making

STEP 9: Final Recommendation
Produces a structured decision with reasoning, trade-offs, and citations.

Features:
- Clear recommendation (not "it depends")
- Key reasons with evidence
- Trade-offs and risks
- Action plan
- Citations to top sources
"""

from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import LLMClient
from src.models.claim import EvidenceSummary, Claim
from src.analysis.gaps import GapAnalysisResult
from src.research.ranker import RankingResult


@dataclass
class Recommendation:
    """
    Structured recommendation with reasoning.
    
    This is the final output of the research agent.
    """
    # Core recommendation
    decision: str  # 1-2 line clear recommendation
    confidence: str  # "high", "medium", "low"
    
    # Reasoning
    key_reasons: list[str]  # 3-6 evidence-based reasons
    trade_offs: list[dict]  # {pro: str, con: str}
    risks: list[str]  # Risks and limitations
    
    # Action plan
    next_steps: list[str]  # What to do next
    
    # Citations
    top_sources: list[dict]  # {title: str, url: str, why: str}
    
    # Disclaimer
    disclaimer: str = (
        "This analysis is for informational purposes only. "
        "It is not professional advice for legal, medical, or financial decisions. "
        "Always consult qualified professionals for important decisions."
    )
    
    def __str__(self) -> str:
        return f"[{self.confidence.upper()}] {self.decision}"
    
    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "key_reasons": self.key_reasons,
            "trade_offs": self.trade_offs,
            "risks": self.risks,
            "next_steps": self.next_steps,
            "top_sources": self.top_sources,
            "disclaimer": self.disclaimer,
        }


# System prompt for decision making
DECISION_SYSTEM_PROMPT = """You are a decision support assistant producing a final recommendation.

Your job is to synthesize research evidence into a nuanced, actionable recommendation.

CONFIDENCE CALIBRATION:
- HIGH: Only if evidence is overwhelming, consistent, and gaps are minimal
- MEDIUM: Default when evidence is mixed, gaps exist, or context matters significantly
- LOW: When evidence is sparse, highly conflicting, or major unknowns remain

Most real-world questions should be MEDIUM confidence because:
- Research rarely covers all contexts and use cases
- Gaps and unknowns introduce uncertainty
- Trade-offs mean the answer depends on priorities

RECOMMENDATION STYLE:
- Be NUANCED, not absolute. Avoid "Do X" or "Do not X" without qualification.
- Prefer conditional phrasing: "X is not recommended for Y context at this time"
- Acknowledge that different contexts may warrant different decisions
- Specify WHEN or WHERE the recommendation applies
- Use phrases like: "at this time", "without proper safeguards", "for general/broad use"

BAD examples (too absolute):
- "Do not adopt AI agents" 
- "AI agents are not ready"
- "You should definitely use X"

GOOD examples (nuanced):
- "Broad consumer or unsupervised commercial adoption is not recommended at this time"
- "Adoption may be viable in controlled enterprise settings with proper oversight"
- "For mission-critical applications, additional validation is needed before deployment"

Structure your response as:
1. DECISION: 1-2 sentence nuanced recommendation with context qualifiers
2. CONFIDENCE: high/medium/low based on evidence strength and gaps
3. KEY REASONS: 3-6 bullets citing specific evidence
4. TRADE-OFFS: pros and cons to consider
5. RISKS: what could go wrong, limitations
6. NEXT STEPS: 3-5 actionable items

Output JSON format:
{
  "decision": "Nuanced 1-2 sentence recommendation with context",
  "confidence": "medium",
  "confidence_reason": "Why this confidence level",
  "key_reasons": [
    "Reason 1 citing evidence",
    "Reason 2 citing evidence"
  ],
  "trade_offs": [
    {"pro": "advantage", "con": "disadvantage"}
  ],
  "risks": [
    "Risk 1",
    "Risk 2"
  ],
  "next_steps": [
    "Step 1",
    "Step 2"
  ]
}"""


DECISION_PROMPT_TEMPLATE = """Based on this research, produce a final recommendation.

ORIGINAL QUESTION: {question}

KEY EVIDENCE (from ranked sources):
{evidence}

IDENTIFIED GAPS:
{gaps}

TOP SOURCES:
{sources}

---

Synthesize this into a nuanced, actionable recommendation.

IMPORTANT:
- Factor the GAPS into your confidence level. More gaps = lower confidence.
- Use CONDITIONAL language that specifies context (e.g., "for broad adoption", "at this time", "without safeguards")
- Avoid absolute statements. Real decisions depend on context.
- Default to MEDIUM confidence unless evidence is overwhelming and consistent.
- Cite specific evidence for your reasons.

Return JSON with decision, confidence, key_reasons, trade_offs, risks, next_steps."""


class DecisionMaker:
    """
    Synthesizes research into a final recommendation.
    
    Uses evidence, gaps, and ranked sources to produce
    a structured, actionable recommendation.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    def make_recommendation(
        self,
        question: str,
        evidence: EvidenceSummary,
        gaps: GapAnalysisResult,
        ranking: Optional[RankingResult] = None,
    ) -> Recommendation:
        """
        Produce a final recommendation.
        
        Args:
            question: The original research question
            evidence: All extracted claims
            gaps: Gap analysis results
            ranking: Optional source ranking results
            
        Returns:
            Structured Recommendation
        """
        # Format inputs for prompt
        evidence_text = self._format_evidence(evidence)
        gaps_text = self._format_gaps(gaps)
        sources_text = self._format_sources(ranking, evidence)
        
        try:
            # Build prompt
            prompt = DECISION_PROMPT_TEMPLATE.format(
                question=question,
                evidence=evidence_text,
                gaps=gaps_text,
                sources=sources_text,
            )
            
            # Call LLM
            response = self.llm.complete_json(
                prompt=prompt,
                system_prompt=DECISION_SYSTEM_PROMPT,
                temperature=0.4,
            )
            
            # Parse response with gap-aware confidence adjustment
            return self._parse_response(response, ranking, evidence, gaps)
            
        except Exception as e:
            # Return fallback recommendation on error
            return Recommendation(
                decision=f"Unable to produce recommendation: {str(e)[:50]}",
                confidence="low",
                key_reasons=["Error occurred during analysis"],
                trade_offs=[],
                risks=["Analysis incomplete"],
                next_steps=["Retry the research", "Consult additional sources"],
                top_sources=[],
            )
    
    def _format_evidence(self, evidence: EvidenceSummary) -> str:
        """Format evidence for the prompt."""
        lines = []
        
        # Group by type for clarity
        by_type = evidence.claims_by_type()
        
        # Prioritize actionable claim types
        priority_order = ['risk', 'failure', 'limitation', 'metric', 'practice', 'example', 'benefit']
        
        count = 0
        for type_name in priority_order:
            from src.models.claim import ClaimType
            try:
                claim_type = ClaimType(type_name)
            except ValueError:
                continue
                
            if claim_type not in by_type:
                continue
            
            claims = by_type[claim_type][:3]  # Max 3 per type
            for claim in claims:
                if count >= 20:  # Limit total
                    break
                lines.append(f"[{type_name.upper()}] {claim.text}")
                lines.append(f"  Source: {claim.source_title[:40]}")
                count += 1
        
        return "\n".join(lines) if lines else "No specific evidence extracted."
    
    def _format_gaps(self, gaps: GapAnalysisResult) -> str:
        """Format gaps for the prompt."""
        lines = []
        
        if gaps.unknowns:
            lines.append("UNKNOWNS:")
            for u in gaps.unknowns[:4]:
                lines.append(f"  - [{u.importance.upper()}] {u.description}")
        
        if gaps.conflicts:
            lines.append("CONFLICTS:")
            for c in gaps.conflicts[:2]:
                lines.append(f"  - {c.description}")
        
        if gaps.assumptions:
            lines.append("ASSUMPTIONS:")
            for a in gaps.assumptions[:3]:
                lines.append(f"  - {a.description}")
        
        return "\n".join(lines) if lines else "No significant gaps identified."
    
    def _format_sources(
        self, 
        ranking: Optional[RankingResult], 
        evidence: EvidenceSummary
    ) -> str:
        """Format top sources for the prompt."""
        lines = []
        
        if ranking:
            for score in ranking.top_3:
                lines.append(f"- {score.source.title[:50]}...")
                lines.append(f"  URL: {score.source.url}")
                lines.append(f"  Score: {score.total_score:.2f} ({score.justification})")
        else:
            # Fall back to sources from claims
            seen_urls = set()
            for claim in evidence.all_claims[:5]:
                if claim.source_url not in seen_urls:
                    seen_urls.add(claim.source_url)
                    lines.append(f"- {claim.source_title[:50]}...")
                    lines.append(f"  URL: {claim.source_url}")
        
        return "\n".join(lines) if lines else "No sources available."
    
    def _parse_response(
        self, 
        response: dict,
        ranking: Optional[RankingResult],
        evidence: EvidenceSummary,
        gaps: Optional[GapAnalysisResult] = None,
    ) -> Recommendation:
        """Parse LLM response into Recommendation with gap-aware confidence."""
        
        # Build top sources list
        top_sources = []
        if ranking:
            for score in ranking.top_3:
                top_sources.append({
                    "title": score.source.title,
                    "url": score.source.url,
                    "why": score.justification,
                })
        
        # Get LLM's confidence
        confidence = response.get("confidence", "medium").lower()
        
        # Adjust confidence based on gaps (programmatic safeguard)
        # If significant gaps exist, cap confidence at medium
        if gaps:
            total_gaps = gaps.total_gaps
            high_importance_unknowns = sum(
                1 for u in gaps.unknowns if u.importance == "high"
            )
            conflicts = len(gaps.conflicts)
            
            # Cap at medium if: many gaps, high-importance unknowns, or conflicts
            if confidence == "high":
                if total_gaps >= 5 or high_importance_unknowns >= 2 or conflicts >= 2:
                    confidence = "medium"
        
        return Recommendation(
            decision=response.get("decision", "No clear recommendation could be made."),
            confidence=confidence,
            key_reasons=response.get("key_reasons", [])[:6],
            trade_offs=response.get("trade_offs", [])[:4],
            risks=response.get("risks", [])[:5],
            next_steps=response.get("next_steps", [])[:5],
            top_sources=top_sources,
        )

