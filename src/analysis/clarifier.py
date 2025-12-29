"""
Clarifying Questions

STEP 8: Inquiry Mode
Generates clarifying questions when gaps affect decisions.

Features:
- Generates 1-3 decision-shaping questions
- Prioritizes questions that most affect the recommendation
- Uses plain English, answerable in one sentence
"""

from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import LLMClient
from src.analysis.gaps import GapAnalysisResult


@dataclass
class ClarifyingQuestion:
    """A question to ask the user for clarification."""
    question: str
    why_it_matters: str  # Why this affects the decision
    priority: int  # 1 = highest priority
    example_answers: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return self.question


@dataclass
class ClarificationRequest:
    """Request for user clarification based on gaps."""
    questions: list[ClarifyingQuestion]
    context: str  # Brief explanation of why we're asking
    
    @property
    def top_question(self) -> Optional[ClarifyingQuestion]:
        """Get the highest priority question."""
        if not self.questions:
            return None
        return min(self.questions, key=lambda q: q.priority)
    
    def __len__(self) -> int:
        return len(self.questions)


# System prompt for generating clarifying questions
CLARIFIER_SYSTEM_PROMPT = """You are a decision support assistant. Based on identified gaps in research, you generate clarifying questions to ask the user.

Your questions must:
1. Be decision-shaping (the answer changes the recommendation)
2. Be answerable in one sentence
3. Use plain English (no jargon)
4. Not be redundant with each other
5. Focus on the most important gaps

Good questions:
- "Is your goal to deploy AI in high-stakes decisions or low-risk assistive tasks?"
- "Do you have an existing team with AI/ML experience?"
- "What is your acceptable failure rate for this system?"
- "Is full automation required, or is human-in-the-loop acceptable?"

Bad questions (avoid):
- "What is your opinion on AI?" (too vague)
- "Have you considered the implications of transformer architectures?" (too technical)
- "Do you want to use AI?" (doesn't shape decision)

Generate 1-3 questions that would most change the recommendation based on the gaps.

Output JSON:
{
  "context": "Brief explanation of why clarification is needed",
  "questions": [
    {
      "question": "The question in plain English",
      "why_it_matters": "How the answer affects the recommendation",
      "priority": 1,
      "example_answers": ["Answer option 1", "Answer option 2"]
    }
  ]
}"""


CLARIFIER_PROMPT_TEMPLATE = """Based on these research gaps, generate 1-3 clarifying questions to ask the user.

ORIGINAL QUESTION: {question}

IDENTIFIED GAPS:

UNKNOWNS:
{unknowns}

CONFLICTS:
{conflicts}

ASSUMPTIONS:
{assumptions}

---

Generate questions that:
1. Would most change the recommendation if answered
2. Are answerable in one sentence
3. Use plain, non-technical language

Return JSON with context and questions array."""


class Clarifier:
    """
    Generates clarifying questions based on research gaps.
    
    Focuses on questions that would most affect the final
    recommendation if answered.
    """
    
    MIN_QUESTIONS = 1
    MAX_QUESTIONS = 3
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    def generate_questions(
        self, 
        question: str, 
        gaps: GapAnalysisResult
    ) -> ClarificationRequest:
        """
        Generate clarifying questions based on gaps.
        
        Args:
            question: The original research question
            gaps: Gap analysis results
            
        Returns:
            ClarificationRequest with questions to ask
        """
        # Format gaps for prompt
        unknowns_text = self._format_unknowns(gaps)
        conflicts_text = self._format_conflicts(gaps)
        assumptions_text = self._format_assumptions(gaps)
        
        # If no significant gaps, return empty
        if not gaps.unknowns and not gaps.conflicts and not gaps.assumptions:
            return ClarificationRequest(
                questions=[],
                context="No significant gaps identified that require clarification."
            )
        
        try:
            # Build prompt
            prompt = CLARIFIER_PROMPT_TEMPLATE.format(
                question=question,
                unknowns=unknowns_text or "None identified",
                conflicts=conflicts_text or "None identified",
                assumptions=assumptions_text or "None identified",
            )
            
            # Call LLM
            response = self.llm.complete_json(
                prompt=prompt,
                system_prompt=CLARIFIER_SYSTEM_PROMPT,
                temperature=0.5,
            )
            
            # Parse response
            return self._parse_response(response)
            
        except Exception as e:
            # Return fallback question on error
            return ClarificationRequest(
                questions=[ClarifyingQuestion(
                    question="What is your primary use case and risk tolerance?",
                    why_it_matters="Understanding your context helps tailor the recommendation",
                    priority=1,
                    example_answers=["Low-risk assistive use", "High-stakes autonomous use"],
                )],
                context=f"Default question due to: {str(e)[:50]}"
            )
    
    def _format_unknowns(self, gaps: GapAnalysisResult) -> str:
        """Format unknowns for the prompt."""
        lines = []
        for u in gaps.unknowns[:5]:
            importance = u.importance.upper()
            lines.append(f"- [{importance}] {u.description}")
        return "\n".join(lines)
    
    def _format_conflicts(self, gaps: GapAnalysisResult) -> str:
        """Format conflicts for the prompt."""
        lines = []
        for c in gaps.conflicts[:3]:
            lines.append(f"- {c.description}")
        return "\n".join(lines)
    
    def _format_assumptions(self, gaps: GapAnalysisResult) -> str:
        """Format assumptions for the prompt."""
        lines = []
        for a in gaps.assumptions[:4]:
            lines.append(f"- {a.description} (Risk: {a.risk})")
        return "\n".join(lines)
    
    def _parse_response(self, response: dict) -> ClarificationRequest:
        """Parse LLM response into ClarificationRequest."""
        context = response.get("context", "Clarification needed to refine recommendation.")
        
        questions = []
        for i, item in enumerate(response.get("questions", [])[:self.MAX_QUESTIONS]):
            if isinstance(item, dict) and item.get("question"):
                questions.append(ClarifyingQuestion(
                    question=item["question"],
                    why_it_matters=item.get("why_it_matters", "Affects the recommendation"),
                    priority=item.get("priority", i + 1),
                    example_answers=item.get("example_answers", []),
                ))
        
        return ClarificationRequest(
            questions=questions,
            context=context,
        )

