"""
Content Extraction

STEP 4: Content Fetch & Clean Extraction
Fetches web pages and extracts readable text content.

Features:
- Fetches page content via HTTP
- Extracts main article text (removes nav/ads/boilerplate)
- Falls back to search snippet if extraction fails
- Handles timeouts and blocked requests gracefully
"""

import httpx
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# trafilatura is excellent for article extraction
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

from src.models.source import Source


@dataclass
class ExtractionResult:
    """Result of content extraction for a single source."""
    source: Source
    success: bool
    content: str
    content_length: int
    preview: str  # First 300-500 chars
    error: Optional[str] = None
    used_fallback: bool = False  # True if using snippet instead of full content
    
    def __str__(self) -> str:
        status = "✅" if self.success else "⚠️"
        fallback = " (fallback)" if self.used_fallback else ""
        return f"{status} {self.source.title[:40]}... (~{self.content_length:,} chars){fallback}"


@dataclass
class ExtractionSummary:
    """Summary of extraction across all sources."""
    results: list[ExtractionResult]
    total: int
    successful: int
    failed: int
    fallback_count: int
    
    @property
    def success_rate(self) -> float:
        return self.successful / self.total if self.total > 0 else 0.0
    
    def __str__(self) -> str:
        return (
            f"Extracted: {self.successful}/{self.total} "
            f"({self.success_rate:.0%} success, {self.fallback_count} fallbacks)"
        )


class ContentExtractor:
    """
    Extracts readable content from web pages.
    
    Uses trafilatura for high-quality article extraction,
    with fallback to search snippets when extraction fails.
    """
    
    def __init__(
        self,
        timeout: float = 15.0,
        max_content_length: int = 15000,  # Limit content to avoid token overflow
        preview_length: int = 400,
        max_workers: int = 5,  # Parallel fetching
    ):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.preview_length = preview_length
        self.max_workers = max_workers
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0; +research)"
            }
        )
    
    def extract_single(self, source: Source) -> ExtractionResult:
        """
        Extract content from a single source.
        
        Falls back to search snippet if extraction fails.
        """
        try:
            # Fetch the page
            response = self.client.get(source.url)
            response.raise_for_status()
            html = response.text
            
            # Extract main content
            content = self._extract_text(html, source.url)
            
            if content and len(content) > 100:
                # Successful extraction
                content = self._truncate(content)
                preview = self._make_preview(content)
                
                # Update source with content
                source.content = content
                
                return ExtractionResult(
                    source=source,
                    success=True,
                    content=content,
                    content_length=len(content),
                    preview=preview,
                )
            else:
                # Extraction returned empty/too short - use fallback
                return self._fallback_to_snippet(source, "Extraction returned insufficient content")
                
        except httpx.TimeoutException:
            return self._fallback_to_snippet(source, "Request timed out")
        except httpx.HTTPStatusError as e:
            return self._fallback_to_snippet(source, f"HTTP {e.response.status_code}")
        except Exception as e:
            return self._fallback_to_snippet(source, str(e)[:50])
    
    def _extract_text(self, html: str, url: str) -> Optional[str]:
        """Extract main text content from HTML."""
        if TRAFILATURA_AVAILABLE:
            # trafilatura is excellent at extracting article content
            content = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            return content
        else:
            # Basic fallback using BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script/style elements
                for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    tag.decompose()
                
                # Get text
                text = soup.get_text(separator=' ', strip=True)
                
                # Basic cleanup
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                return ' '.join(lines)
            except Exception:
                return None
    
    def _truncate(self, content: str) -> str:
        """Truncate content to max length."""
        if len(content) <= self.max_content_length:
            return content
        
        # Try to truncate at a sentence boundary
        truncated = content[:self.max_content_length]
        last_period = truncated.rfind('.')
        if last_period > self.max_content_length * 0.8:
            return truncated[:last_period + 1]
        return truncated + "..."
    
    def _make_preview(self, content: str) -> str:
        """Create a preview snippet from content."""
        preview = content[:self.preview_length]
        
        # Try to end at a sentence or word boundary
        if len(content) > self.preview_length:
            last_space = preview.rfind(' ')
            if last_space > self.preview_length * 0.7:
                preview = preview[:last_space]
            preview += "..."
        
        return preview
    
    def _fallback_to_snippet(self, source: Source, error: str) -> ExtractionResult:
        """Use search snippet as fallback content."""
        snippet = source.snippet or ""
        
        if snippet:
            # Use snippet as content
            source.content = snippet
            
            return ExtractionResult(
                source=source,
                success=True,  # Partial success
                content=snippet,
                content_length=len(snippet),
                preview=snippet[:self.preview_length],
                error=error,
                used_fallback=True,
            )
        else:
            return ExtractionResult(
                source=source,
                success=False,
                content="",
                content_length=0,
                preview="",
                error=error,
                used_fallback=True,
            )
    
    def extract_all(self, sources: list[Source]) -> ExtractionSummary:
        """
        Extract content from all sources in parallel.
        
        Returns ExtractionSummary with results and statistics.
        """
        results = []
        
        # Use thread pool for parallel fetching
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_source = {
                executor.submit(self.extract_single, source): source
                for source in sources
            }
            
            for future in as_completed(future_to_source):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    source = future_to_source[future]
                    results.append(ExtractionResult(
                        source=source,
                        success=False,
                        content="",
                        content_length=0,
                        preview="",
                        error=str(e)[:50],
                    ))
        
        # Calculate statistics
        successful = sum(1 for r in results if r.success)
        fallback_count = sum(1 for r in results if r.used_fallback)
        
        return ExtractionSummary(
            results=results,
            total=len(results),
            successful=successful,
            failed=len(results) - successful,
            fallback_count=fallback_count,
        )

