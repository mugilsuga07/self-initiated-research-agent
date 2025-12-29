"""
LLM Client Abstraction Layer

Provides a clean interface to OpenAI's API with:
- Environment-based configuration
- Retry logic
- Structured output support
- Easy model switching
"""

import os
import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Override model from environment if set
        env_model = os.getenv("OPENAI_MODEL")
        if env_model:
            self.model = env_model


class LLMClient:
    """
    Wrapper around OpenAI's API for the Research Agent.
    
    Handles:
    - API configuration
    - Structured JSON outputs
    - Error handling
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Run: pip install openai"
                )
        return self._client
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            The generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Generate a JSON response from the LLM.
        
        Args:
            prompt: The user prompt (should request JSON output)
            system_prompt: Optional system prompt
            temperature: Override default temperature
            
        Returns:
            Parsed JSON as a dictionary
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content
        return json.loads(content)

