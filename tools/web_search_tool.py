"""
Web Search Tool

Provides web search capabilities using DuckDuckGo search engine.
Features:
- Multiple search engine support (DuckDuckGo primary, extensible for others)
- Result caching to reduce redundant searches
- Rate limiting to prevent abuse
- Structured output with titles, snippets, URLs
- Error handling and retry logic
"""

import sys
sys.path.insert(0, '..')

from protocol_ai import Tool, ToolResult
from typing import Dict, Any, List, Optional
import time
import hashlib
from collections import OrderedDict


class WebSearchTool(Tool):
    """
    Web search tool with caching and rate limiting.

    Supports DuckDuckGo search with structured results.
    """

    def __init__(self):
        """Initialize the web search tool."""
        name = "web_search"
        description = "Search the web for information using DuckDuckGo. Returns titles, snippets, and URLs."

        # Define parameter schema
        parameters = {
            "query": {
                "type": "string",
                "description": "Search query to execute",
                "required": True
            },
            "max_results": {
                "type": "number",
                "description": "Maximum number of results to return (1-20)",
                "required": False,
                "default": 5
            },
            "region": {
                "type": "string",
                "description": "Region for search results (e.g., 'us-en', 'uk-en')",
                "required": False,
                "default": "wt-wt"  # No region (worldwide)
            }
        }

        super().__init__(name, description, parameters)

        # Rate limiting: max queries per minute
        self.rate_limit = 10
        self.request_times: List[float] = []

        # Cache: store recent searches (max 100 entries)
        self.cache_max_size = 100
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.cache_ttl = 3600  # Cache TTL: 1 hour

    def _get_cache_key(self, query: str, max_results: int, region: str) -> str:
        """
        Generate cache key for a search query.

        Args:
            query: Search query
            max_results: Number of results
            region: Search region

        Returns:
            Cache key hash
        """
        cache_string = f"{query}:{max_results}:{region}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _check_rate_limit(self) -> tuple[bool, str]:
        """
        Check if request is within rate limit.

        Returns:
            (is_allowed, error_message)
        """
        current_time = time.time()

        # Remove requests older than 1 minute
        self.request_times = [
            t for t in self.request_times
            if current_time - t < 60
        ]

        # Check if at rate limit
        if len(self.request_times) >= self.rate_limit:
            return False, f"Rate limit exceeded: {self.rate_limit} requests per minute"

        # Add current request
        self.request_times.append(current_time)
        return True, ""

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Check if result is in cache and still valid.

        Args:
            cache_key: Cache key to check

        Returns:
            Cached result or None if not found/expired
        """
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            cached_time = cached_entry.get('timestamp', 0)
            current_time = time.time()

            # Check if cache is still valid
            if current_time - cached_time < self.cache_ttl:
                print(f"[WebSearch] Cache hit for key: {cache_key[:8]}...")
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                return cached_entry.get('results')

            # Cache expired, remove it
            del self.cache[cache_key]

        return None

    def _add_to_cache(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """
        Add search results to cache.

        Args:
            cache_key: Cache key
            results: Search results to cache
        """
        # Implement LRU: remove oldest if at max size
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entry (first item)
            self.cache.popitem(last=False)

        # Add new entry
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'results': results
        }

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute web search.

        Args:
            query: Search query string
            max_results: Maximum results to return (default: 5)
            region: Search region (default: worldwide)

        Returns:
            ToolResult with search results or error
        """
        query = kwargs.get('query', '').strip()
        max_results = min(int(kwargs.get('max_results', 5)), 20)  # Cap at 20
        region = kwargs.get('region', 'wt-wt')

        # Validate query
        if not query:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="Search query cannot be empty"
            )

        # Check rate limit
        is_allowed, error_msg = self._check_rate_limit()
        if not is_allowed:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=error_msg
            )

        # Check cache
        cache_key = self._get_cache_key(query, max_results, region)
        cached_results = self._check_cache(cache_key)
        if cached_results is not None:
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=cached_results,
                metadata={
                    "query": query,
                    "from_cache": True,
                    "result_count": len(cached_results)
                }
            )

        # Perform search
        try:
            results = await self._search_duckduckgo(query, max_results, region)

            # Cache results
            self._add_to_cache(cache_key, results)

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=results,
                metadata={
                    "query": query,
                    "from_cache": False,
                    "result_count": len(results),
                    "region": region
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Search failed: {str(e)}"
            )

    async def _search_duckduckgo(self, query: str, max_results: int,
                                  region: str) -> List[Dict[str, Any]]:
        """
        Perform DuckDuckGo search.

        Args:
            query: Search query
            max_results: Maximum results
            region: Search region

        Returns:
            List of search results

        Raises:
            Exception: If search fails
        """
        try:
            # Try new package name first
            from ddgs import DDGS
        except ImportError:
            try:
                # Fallback to old package name
                from duckduckgo_search import DDGS
            except ImportError:
                raise ImportError(
                    "DDGS library not installed. "
                    "Install with: pip install ddgs"
                )

        results = []

        try:
            # Create DDGS instance and perform search
            with DDGS() as ddgs:
                # Use text search
                search_results = ddgs.text(
                    keywords=query,
                    region=region,
                    max_results=max_results
                )

                # Format results
                for result in search_results:
                    results.append({
                        "title": result.get('title', 'No title'),
                        "snippet": result.get('body', 'No description'),
                        "url": result.get('href', ''),
                        "source": "DuckDuckGo"
                    })

            print(f"[WebSearch] Found {len(results)} results for: {query}")
            return results

        except Exception as e:
            # More specific error handling
            error_msg = str(e)
            if "ratelimit" in error_msg.lower():
                raise Exception("DuckDuckGo rate limit reached. Please try again later.")
            elif "timeout" in error_msg.lower():
                raise Exception("Search timed out. Please try again.")
            else:
                raise Exception(f"DuckDuckGo search error: {error_msg}")

    def clear_cache(self) -> None:
        """Clear the search cache."""
        self.cache.clear()
        print(f"[WebSearch] Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_max_size,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit,
            "recent_requests": len(self.request_times)
        }
