"""
Session management for the Research Agent.

STEP 1: Input & Session Setup
- Accepts user questions with validation
- Generates unique session IDs
- Stores sessions for tracking
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


class InputValidationError(Exception):
    """Raised when user input fails validation."""
    pass


@dataclass
class Session:
    """
    Represents a single research session.
    
    Each session tracks one user question and its associated
    research artifacts throughout the agent workflow.
    """
    session_id: str
    question: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Will be populated in later steps
    sub_questions: list[str] = field(default_factory=list)
    sources: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    clarifications: list[str] = field(default_factory=list)
    recommendation: Optional[dict] = None
    
    def __str__(self) -> str:
        """Display session info in the required format."""
        return f"Session: {self.session_id}\nQuestion: {self.question}"
    
    def to_dict(self) -> dict:
        """Convert session to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "question": self.question,
            "created_at": self.created_at.isoformat(),
            "sub_questions": self.sub_questions,
            "sources_count": len(self.sources),
            "claims_count": len(self.claims),
            "gaps": self.gaps,
            "clarifications": self.clarifications,
            "has_recommendation": self.recommendation is not None,
        }


class SessionManager:
    """
    Manages research sessions with input validation.
    
    Ensures:
    - Each question gets a unique session ID
    - Empty inputs are rejected politely
    - Very long inputs are accepted without crashing
    """
    
    # Reasonable limits (configurable)
    MIN_QUESTION_LENGTH = 5
    MAX_QUESTION_LENGTH = 10000  # Generous limit for complex questions
    
    def __init__(self):
        self.sessions: dict[str, Session] = {}
    
    def validate_input(self, question: str) -> str:
        """
        Validate and clean user input.
        
        Args:
            question: Raw user input
            
        Returns:
            Cleaned question string
            
        Raises:
            InputValidationError: If input is invalid
        """
        # Handle None
        if question is None:
            raise InputValidationError(
                "Please provide a question. Empty input is not allowed."
            )
        
        # Clean whitespace
        cleaned = question.strip()
        
        # Check for empty input
        if not cleaned:
            raise InputValidationError(
                "Please provide a question. Empty input is not allowed."
            )
        
        # Check minimum length (avoid single words like "AI?")
        if len(cleaned) < self.MIN_QUESTION_LENGTH:
            raise InputValidationError(
                f"Question is too short. Please provide at least {self.MIN_QUESTION_LENGTH} characters."
            )
        
        # Check maximum length (prevent abuse, but accept long questions)
        if len(cleaned) > self.MAX_QUESTION_LENGTH:
            raise InputValidationError(
                f"Question is too long ({len(cleaned):,} chars). "
                f"Maximum allowed is {self.MAX_QUESTION_LENGTH:,} characters."
            )
        
        return cleaned
    
    def generate_session_id(self) -> str:
        """
        Generate a unique, human-readable session ID.
        
        Format: run_YYYY_MM_DD_<8-char-hex>
        Example: run_2025_12_28_a1b2c3d4
        """
        timestamp = datetime.now().strftime("%Y_%m_%d")
        unique_hex = uuid.uuid4().hex[:8]
        return f"run_{timestamp}_{unique_hex}"
    
    def create_session(self, question: str) -> Session:
        """
        Create a new research session.
        
        Args:
            question: The user's research question
            
        Returns:
            A new Session object with unique ID
            
        Raises:
            InputValidationError: If question is invalid
        """
        # Validate and clean input
        cleaned_question = self.validate_input(question)
        
        # Generate unique session ID
        session_id = self.generate_session_id()
        
        # Create session
        session = Session(
            session_id=session_id,
            question=cleaned_question
        )
        
        # Store session
        self.sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> list[Session]:
        """List all sessions, most recent first."""
        return sorted(
            self.sessions.values(),
            key=lambda s: s.created_at,
            reverse=True
        )
    
    def session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self.sessions)

