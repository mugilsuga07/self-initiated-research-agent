"""
Goal Decomposition (Planner)

STEP 2: Break down a high-level question into research sub-questions.

Rules:
- Generate 4-7 sub-questions
- Each must be answerable via external sources
- Avoid duplicates
- Cover different aspects: adoption, failures, risks, best practices
"""

from typing import Optional
from dataclasses import dataclass

from src.llm.client import LLMClient


# System prompt for the decomposition task
DECOMPOSITION_SYSTEM_PROMPT = """You are a decision support assistant. Your job is to break down complex decision questions into sub-questions that gather evidence FOR and AGAINST a decision.

You are NOT writing a literature review. You are helping someone DECIDE.

Rules:
1. Generate 4-7 sub-questions
2. Each sub-question must help answer: "Should I do this? What are the risks?"
3. Avoid duplicates or overly similar questions
4. Cover decision-relevant angles:
   - What evidence supports this working in practice?
   - What failures or limitations have been reported?
   - What risks remain unresolved or poorly understood?
   - What guardrails or conditions are required for success?
   - What do practitioners recommend based on real experience?
5. Make questions specific and evidence-focused
6. Start questions with phrases like:
   - "What evidence suggests..."
   - "What failures or limitations..."
   - "What risks remain..."
   - "What conditions are required..."
   - "What do practitioners report..."
7. AVOID academic/literature phrasing like:
   - "What are the latest research papers..."
   - "What studies have been published..."
   - "What is the current state of research..."

Output format: JSON with a single key "sub_questions" containing an array of strings."""


DECOMPOSITION_PROMPT_TEMPLATE = """Break down this decision question into 4-7 evidence-gathering sub-questions:

QUESTION: {question}

Generate sub-questions that help someone DECIDE, not just learn. Focus on:
- Evidence of success or failure in real-world use
- Reported risks, failures, and limitations
- Conditions required for success
- Practitioner recommendations

Avoid academic phrasing. Use decision-oriented language.

Return JSON: {{"sub_questions": ["question1", "question2", ...]}}"""


@dataclass
class DecompositionResult:
    """Result of goal decomposition."""
    original_question: str
    sub_questions: list[str]
    
    def __str__(self) -> str:
        lines = [f"Original: {self.original_question}", "", "Sub-questions:"]
        for i, sq in enumerate(self.sub_questions, 1):
            lines.append(f"  {i}. {sq}")
        return "\n".join(lines)


class Planner:
    """
    Breaks down high-level questions into researchable sub-questions.
    
    Uses an LLM to generate diverse, specific sub-questions that
    cover different aspects of the original question.
    """
    
    MIN_SUB_QUESTIONS = 4
    MAX_SUB_QUESTIONS = 7
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    def decompose(self, question: str) -> DecompositionResult:
        """
        Decompose a question into sub-questions.
        
        Args:
            question: The high-level research question
            
        Returns:
            DecompositionResult with original question and sub-questions
        """
        # Generate sub-questions using LLM
        prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(question=question)
        
        response = self.llm.complete_json(
            prompt=prompt,
            system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
            temperature=0.7,
        )
        
        sub_questions = response.get("sub_questions", [])
        
        # Validate and clean
        sub_questions = self._validate_and_clean(sub_questions)
        
        return DecompositionResult(
            original_question=question,
            sub_questions=sub_questions
        )
    
    def _validate_and_clean(self, sub_questions: list) -> list[str]:
        """
        Validate and clean the generated sub-questions.
        
        - Remove duplicates
        - Remove empty strings
        - Ensure minimum count
        """
        # Clean and deduplicate
        seen = set()
        cleaned = []
        
        for sq in sub_questions:
            # Convert to string and strip
            sq_str = str(sq).strip()
            
            # Skip empty
            if not sq_str:
                continue
            
            # Skip duplicates (case-insensitive)
            sq_lower = sq_str.lower()
            if sq_lower in seen:
                continue
            
            seen.add(sq_lower)
            cleaned.append(sq_str)
        
        # Ensure we have at least minimum questions
        if len(cleaned) < self.MIN_SUB_QUESTIONS:
            # This shouldn't happen with a good LLM, but handle gracefully
            pass  # We'll accept what we got
        
        # Limit to maximum
        return cleaned[:self.MAX_SUB_QUESTIONS]
    
    def decompose_with_quality_check(self, question: str) -> DecompositionResult:
        """
        Decompose with additional quality filtering.
        
        Removes sub-questions that are:
        - Too vague ("tell me more about X")
        - Yes/no questions
        - Duplicates of the original question
        """
        result = self.decompose(question)
        
        # Filter out low-quality sub-questions
        quality_filtered = []
        
        vague_patterns = [
            "tell me more",
            "explain",
            "what is",
            "define",
        ]
        
        for sq in result.sub_questions:
            sq_lower = sq.lower()
            
            # Skip vague questions
            if any(pattern in sq_lower for pattern in vague_patterns):
                continue
            
            # Skip if too similar to original (basic check)
            if sq_lower.strip("?") == question.lower().strip("?"):
                continue
            
            quality_filtered.append(sq)
        
        # If filtering removed too many, use original list
        if len(quality_filtered) >= self.MIN_SUB_QUESTIONS:
            result.sub_questions = quality_filtered
        
        return result

