"""
Web Search Integration

STEP 3: Autonomous Discovery
Searches external sources for each sub-question.

Supports:
- Tavily API (recommended for AI applications)
- Serper API (Google Search)
- Mock data for testing without API keys
"""

import os
import httpx
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from src.models.source import Source, SourceType, SearchResults, DiscoveryResults


@dataclass
class SearchConfig:
    """Configuration for web search."""
    max_results_per_query: int = 7  # Fetch more to allow filtering
    max_total_sources: int = 30
    min_reputable_per_question: int = 2  # Ensure quality floor
    include_domains: list[str] = None  # Prefer these domains
    exclude_domains: list[str] = None  # Block these domains
    reputable_domains: list[str] = None  # High-quality sources
    low_quality_title_patterns: list[str] = None  # Filter these titles
    
    def __post_init__(self):
        if self.include_domains is None:
            self.include_domains = []
        if self.exclude_domains is None:
            # Low-quality domains to filter out
            self.exclude_domains = [
                "pinterest.com",
                "quora.com", 
                "reddit.com",
                "facebook.com",
                "twitter.com",
                "x.com",
                "medium.com",  # Often low-signal opinion pieces
            ]
        if self.reputable_domains is None:
            # Trusted sources for engineering/tech decisions
            self.reputable_domains = [
                # Company engineering blogs
                "engineering.fb.com", "engineering.linkedin.com",
                "netflixtechblog.com", "uber.com/blog", "blog.google",
                "aws.amazon.com", "cloud.google.com", "azure.microsoft.com",
                "openai.com", "anthropic.com", "deepmind.com",
                "stripe.com", "shopify.engineering", "github.blog",
                # Reputable tech media
                "techcrunch.com", "wired.com", "arstechnica.com",
                "theverge.com", "thenewstack.io", "infoq.com",
                # Research & analysis
                "arxiv.org", "acm.org", "ieee.org",
                "hbr.org", "mckinsey.com", "gartner.com",
                # Developer-focused
                "stackoverflow.blog", "martinfowler.com", "danluu.com",
            ]
        if self.low_quality_title_patterns is None:
            # Titles that indicate low-signal content
            self.low_quality_title_patterns = [
                "top 10", "top 5", "top 20",
                "beginner's guide", "beginners guide",
                "ultimate guide", "complete guide",
                "everything you need to know",
                "what is", "introduction to",
                "for dummies", "101",
                "you won't believe",
            ]


class TavilySearchClient:
    """
    Tavily API client for AI-focused web search.
    
    Tavily is optimized for AI/LLM applications and returns
    high-quality, relevant results.
    """
    
    BASE_URL = "https://api.tavily.com/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.client = httpx.Client(timeout=30.0)
    
    def is_available(self) -> bool:
        """Check if Tavily API is configured."""
        return bool(self.api_key and self.api_key != "your_tavily_api_key_here")
    
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search using Tavily API.
        
        Returns list of results with: title, url, content, published_date
        """
        if not self.is_available():
            raise ValueError("Tavily API key not configured")
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "search_depth": "basic",
        }
        
        response = self.client.post(self.BASE_URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data.get("results", [])


class SerperSearchClient:
    """
    Serper API client for Google Search results.
    
    Provides Google search results via API.
    """
    
    BASE_URL = "https://google.serper.dev/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.client = httpx.Client(timeout=30.0)
    
    def is_available(self) -> bool:
        """Check if Serper API is configured."""
        return bool(self.api_key and self.api_key != "your_serper_api_key_here")
    
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search using Serper API.
        
        Returns list of results with: title, link, snippet, date
        """
        if not self.is_available():
            raise ValueError("Serper API key not configured")
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        
        payload = {
            "q": query,
            "num": max_results,
        }
        
        response = self.client.post(self.BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        organic = data.get("organic", [])
        
        # Normalize to common format
        results = []
        for item in organic[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "content": item.get("snippet", ""),
                "published_date": item.get("date"),
            })
        
        return results


class MockSearchClient:
    """
    Mock search client for testing without API keys.
    
    Generates unique URLs per query to simulate real search behavior.
    """
    
    # Template data for generating mock results
    MOCK_TEMPLATES = [
        {
            "title_template": "{topic}: A 2024 Production Perspective",
            "domain": "engineering.example.com",
            "content": "Our team deployed AI systems and learned key lessons about reliability, guardrails, and human oversight requirements.",
        },
        {
            "title_template": "Understanding {topic}: Industry Survey",
            "domain": "techblog.example.com",
            "content": "A comprehensive survey of 50 companies reveals common patterns of success and failure in production deployments.",
        },
        {
            "title_template": "{topic}: Technical Deep Dive",
            "domain": "research.example.com",
            "content": "Research paper examining failure modes and proposing robust architectures for production use.",
        },
        {
            "title_template": "Case Study: {topic} in Practice",
            "domain": "casestudies.example.com",
            "content": "Analysis of real-world implementations including challenges, solutions, and measurable outcomes.",
        },
        {
            "title_template": "{topic}: Cost and ROI Analysis",
            "domain": "analyst.example.com",
            "content": "Enterprise adoption shows mixed results. Costs often exceed initial estimates due to monitoring needs.",
        },
    ]
    
    def __init__(self):
        self._query_counter = 0
    
    def is_available(self) -> bool:
        return True
    
    def _extract_topic(self, query: str) -> str:
        """Extract a short topic from the query for title generation."""
        # Remove common words and take first meaningful chunk
        query = query.replace("?", "").strip()
        words = query.split()
        
        # Take first 4-6 words as topic
        topic_words = words[:5]
        topic = " ".join(topic_words)
        
        # Capitalize first letter
        return topic.title() if topic else "AI Agents"
    
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return mock results with unique URLs per query."""
        self._query_counter += 1
        topic = self._extract_topic(query)
        
        results = []
        for i, template in enumerate(self.MOCK_TEMPLATES[:max_results]):
            # Generate unique URL using query counter and result index
            unique_id = f"q{self._query_counter}-r{i}"
            
            results.append({
                "title": template["title_template"].format(topic=topic),
                "url": f"https://{template['domain']}/article-{unique_id}",
                "content": f"{template['content']} [Context: {query[:50]}...]",
                "published_date": f"2024-{(i+6):02d}-{(i*3+10):02d}",
            })
        
        return results


class WebSearcher:
    """
    High-level search interface that:
    - Tries Tavily first, then Serper, then falls back to mock
    - Searches for each sub-question
    - Deduplicates results across questions
    - Filters low-quality domains
    - Enforces source limits
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.tavily = TavilySearchClient()
        self.serper = SerperSearchClient()
        self.mock = MockSearchClient()
        
        # Track which client we're using
        self._active_client = None
        self._using_mock = False
    
    def _get_client(self):
        """Get the best available search client."""
        if self._active_client:
            return self._active_client
        
        if self.tavily.is_available():
            self._active_client = self.tavily
            self._using_mock = False
        elif self.serper.is_available():
            self._active_client = self.serper
            self._using_mock = False
        else:
            self._active_client = self.mock
            self._using_mock = True
        
        return self._active_client
    
    @property
    def is_using_mock(self) -> bool:
        """Check if using mock data."""
        self._get_client()
        return self._using_mock
    
    def _enhance_query(self, sub_question: str) -> str:
        """
        Enhance the query to get better results.
        
        Adds context like "production", "case study" for more relevant results.
        """
        # Add recency hint
        current_year = datetime.now().year
        
        # Keywords to add for better results
        enhancers = ["production", "real-world", str(current_year)]
        
        # Don't add if already present
        query = sub_question
        for enhancer in enhancers:
            if enhancer.lower() not in query.lower():
                # Only add one enhancer to avoid over-stuffing
                query = f"{query} {enhancer}"
                break
        
        return query
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str[:19], fmt)
            except (ValueError, TypeError):
                continue
        
        return None
    
    def _is_valid_source(self, url: str, title: str) -> bool:
        """Check if source passes quality filters."""
        if not url or not title:
            return False
        
        domain = Source._extract_domain(url)
        
        # Check exclude list
        for excluded in self.config.exclude_domains:
            if excluded in domain:
                return False
        
        # Check for low-quality title patterns
        title_lower = title.lower()
        for pattern in self.config.low_quality_title_patterns:
            if pattern in title_lower:
                return False
        
        return True
    
    def _is_reputable_source(self, url: str) -> bool:
        """Check if source is from a reputable domain."""
        domain = Source._extract_domain(url).lower()
        url_lower = url.lower()
        
        for reputable in self.config.reputable_domains:
            if reputable in domain or reputable in url_lower:
                return True
        
        # Also consider engineering blogs as reputable
        if "engineering" in url_lower or "techblog" in url_lower:
            return True
        
        return False
    
    def _detect_source_type(self, domain: str, url: str) -> SourceType:
        """Detect the type of source based on domain."""
        domain_lower = domain.lower()
        url_lower = url.lower()
        
        # News sites
        news_domains = ["nytimes", "bbc", "reuters", "techcrunch", "wired", "theverge", "arstechnica"]
        if any(nd in domain_lower for nd in news_domains):
            return SourceType.NEWS
        
        # Research
        if "arxiv" in domain_lower or "acm.org" in domain_lower or "ieee" in domain_lower:
            return SourceType.RESEARCH
        
        # Official company blogs
        if "engineering" in url_lower or "blog" in url_lower:
            return SourceType.BLOG
        
        # Official docs
        if "docs." in domain_lower or "documentation" in url_lower:
            return SourceType.OFFICIAL
        
        return SourceType.UNKNOWN
    
    def search_single(self, sub_question: str) -> SearchResults:
        """
        Search for a single sub-question.
        
        Returns SearchResults with quality-sorted sources.
        Prioritizes reputable sources while maintaining diversity.
        """
        client = self._get_client()
        enhanced_query = self._enhance_query(sub_question)
        
        raw_results = client.search(enhanced_query, self.config.max_results_per_query)
        
        reputable_sources = []
        other_sources = []
        
        for result in raw_results:
            url = result.get("url", result.get("link", ""))
            title = result.get("title", "")
            snippet = result.get("content", result.get("snippet", ""))
            date_str = result.get("published_date", result.get("date"))
            
            # Validate
            if not self._is_valid_source(url, title):
                continue
            
            source = Source(
                url=url,
                title=title,
                snippet=snippet,
                sub_question=sub_question,
                published_date=self._parse_date(date_str),
                source_type=self._detect_source_type(
                    Source._extract_domain(url), url
                ),
            )
            
            # Separate reputable from other sources
            if self._is_reputable_source(url):
                reputable_sources.append(source)
            else:
                other_sources.append(source)
        
        # Combine: reputable first, then others
        # Ensure at least min_reputable_per_question reputable sources at top
        sources = reputable_sources + other_sources
        
        # Limit to 5 sources per question (original behavior)
        sources = sources[:5]
        
        return SearchResults(sub_question=sub_question, sources=sources)
    
    def search_all(self, sub_questions: list[str]) -> DiscoveryResults:
        """
        Search for all sub-questions with deduplication.
        
        Returns DiscoveryResults with:
        - Results grouped by sub-question
        - All sources deduplicated
        - Total sources capped at max_total_sources
        """
        seen_urls = set()
        results_by_question = {}
        all_sources = []
        
        for sq in sub_questions:
            search_results = self.search_single(sq)
            
            # Deduplicate
            unique_sources = []
            for source in search_results.sources:
                if source.url not in seen_urls:
                    seen_urls.add(source.url)
                    unique_sources.append(source)
                    all_sources.append(source)
            
            search_results.sources = unique_sources
            results_by_question[sq] = search_results
            
            # Check total limit
            if len(all_sources) >= self.config.max_total_sources:
                break
        
        # Trim if over limit
        if len(all_sources) > self.config.max_total_sources:
            all_sources = all_sources[:self.config.max_total_sources]
        
        return DiscoveryResults(
            results_by_question=results_by_question,
            all_sources=all_sources,
        )

