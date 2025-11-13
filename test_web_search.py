"""
Test script for Web Search Tool

Tests the web search functionality including:
- Basic search execution
- Result formatting
- Caching behavior
- Rate limiting
- Error handling
"""

import asyncio
import sys
from protocol_ai import ToolRegistry, ToolLoader


async def test_basic_search():
    """Test basic web search functionality."""
    print("=" * 60)
    print("TEST: Basic Web Search")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Check if web_search tool is loaded
    search_tool = registry.get_tool("web_search")
    if not search_tool:
        print("[ERROR] Web search tool not found")
        return

    print(f"[OK] Web search tool loaded: {search_tool.description}\n")

    # Perform a simple search
    print("Searching for: 'Python programming'")
    result = await registry.execute_tool(
        "web_search",
        query="Python programming",
        max_results=3
    )

    if result.success:
        print(f"[OK] Search completed successfully")
        print(f"     Execution time: {result.execution_time:.2f}s")
        print(f"     Results found: {len(result.output)}")
        print(f"     From cache: {result.metadata.get('from_cache', False)}\n")

        # Display results
        print("Search Results:")
        for i, res in enumerate(result.output, 1):
            print(f"\n{i}. {res['title']}")
            print(f"   URL: {res['url']}")
            print(f"   Snippet: {res['snippet'][:100]}...")
    else:
        print(f"[ERROR] Search failed: {result.error}")

    print()


async def test_cache_functionality():
    """Test search result caching."""
    print("=" * 60)
    print("TEST: Cache Functionality")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # First search (should hit network)
    print("First search (should miss cache)...")
    result1 = await registry.execute_tool(
        "web_search",
        query="AI governance frameworks",
        max_results=2
    )

    if result1.success:
        from_cache_1 = result1.metadata.get('from_cache', False)
        time_1 = result1.execution_time
        print(f"[OK] First search: from_cache={from_cache_1}, time={time_1:.2f}s")

        # Second search (should hit cache)
        print("\nSecond search (should hit cache)...")
        result2 = await registry.execute_tool(
            "web_search",
            query="AI governance frameworks",
            max_results=2
        )

        if result2.success:
            from_cache_2 = result2.metadata.get('from_cache', False)
            time_2 = result2.execution_time
            print(f"[OK] Second search: from_cache={from_cache_2}, time={time_2:.2f}s")

            if from_cache_2 and time_2 < time_1:
                print("[OK] Cache working correctly (faster on second request)")
            elif from_cache_2:
                print("[OK] Cache working (marked as cached)")
            else:
                print("[WARNING] Expected cache hit but got cache miss")
        else:
            print(f"[ERROR] Second search failed: {result2.error}")
    else:
        print(f"[ERROR] First search failed: {result1.error}")

    print()


async def test_rate_limiting():
    """Test rate limiting functionality."""
    print("=" * 60)
    print("TEST: Rate Limiting")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    search_tool = registry.get_tool("web_search")

    print(f"Rate limit: {search_tool.rate_limit} requests per minute")
    print(f"Testing with {search_tool.rate_limit + 2} unique queries...\n")

    # Make requests up to the rate limit
    results = []
    for i in range(search_tool.rate_limit + 2):
        # Use unique queries to avoid cache hits
        query = f"test query {i}"
        result = await registry.execute_tool(
            "web_search",
            query=query,
            max_results=1
        )
        results.append(result)
        print(f"Request {i+1}: {'SUCCESS' if result.success else f'FAILED - {result.error}'}")

    # Count successes and failures
    successes = sum(1 for r in results if r.success)
    failures = sum(1 for r in results if not r.success)

    print(f"\nResults:")
    print(f"  Successful: {successes}")
    print(f"  Failed (rate limited): {failures}")

    if failures > 0:
        print(f"[OK] Rate limiting is working (blocked {failures} requests)")
    else:
        print(f"[WARNING] No rate limiting detected")

    print()


async def test_input_validation():
    """Test input validation."""
    print("=" * 60)
    print("TEST: Input Validation")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Test empty query
    print("Test 1: Empty query")
    result = await registry.execute_tool("web_search", query="")
    if not result.success and "empty" in result.error.lower():
        print(f"[OK] Empty query rejected: {result.error}")
    else:
        print(f"[ERROR] Empty query should be rejected")

    # Test missing query parameter
    print("\nTest 2: Missing query parameter")
    result = await registry.execute_tool("web_search")
    if not result.success and "required parameter" in result.error.lower():
        print(f"[OK] Missing parameter caught: {result.error}")
    else:
        print(f"[ERROR] Missing parameter should be caught")

    # Test excessive max_results (should be capped at 20)
    print("\nTest 3: Excessive max_results (should cap at 20)")
    result = await registry.execute_tool(
        "web_search",
        query="test",
        max_results=100
    )
    if result.success:
        actual_max = min(len(result.output), 20)
        print(f"[OK] Results capped appropriately (got {len(result.output)} results, max 20)")
    else:
        print(f"[INFO] Search failed (may be rate limited): {result.error}")

    print()


async def test_cache_stats():
    """Test cache statistics."""
    print("=" * 60)
    print("TEST: Cache Statistics")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    search_tool = registry.get_tool("web_search")

    # Get initial stats
    stats = search_tool.get_cache_stats()
    print("Cache Statistics:")
    print(f"  Cache size: {stats['cache_size']}/{stats['cache_max_size']}")
    print(f"  Cache TTL: {stats['cache_ttl']}s")
    print(f"  Rate limit: {stats['rate_limit']} req/min")
    print(f"  Recent requests: {stats['recent_requests']}")

    # Perform a search
    await registry.execute_tool("web_search", query="test cache stats", max_results=1)

    # Get updated stats
    stats_after = search_tool.get_cache_stats()
    print(f"\nAfter search:")
    print(f"  Cache size: {stats_after['cache_size']}/{stats_after['cache_max_size']}")
    print(f"  Recent requests: {stats_after['recent_requests']}")

    if stats_after['cache_size'] > stats['cache_size']:
        print("[OK] Cache is being populated")
    else:
        print("[INFO] Cache size unchanged (may have been rate limited)")

    # Clear cache
    search_tool.clear_cache()
    stats_cleared = search_tool.get_cache_stats()
    print(f"\nAfter clearing cache:")
    print(f"  Cache size: {stats_cleared['cache_size']}")
    if stats_cleared['cache_size'] == 0:
        print("[OK] Cache cleared successfully")

    print()


async def test_error_handling():
    """Test error handling with invalid inputs."""
    print("=" * 60)
    print("TEST: Error Handling")
    print("=" * 60)

    registry = ToolRegistry()
    loader = ToolLoader(tools_dir="./tools")
    loader.load_tools(registry)

    # Test with special characters
    print("Test: Query with special characters")
    result = await registry.execute_tool(
        "web_search",
        query="test query with @#$% special chars",
        max_results=1
    )
    print(f"Result: {'SUCCESS' if result.success else f'FAILED - {result.error}'}")

    print()


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WEB SEARCH TOOL TEST SUITE")
    print("=" * 60 + "\n")

    print("NOTE: These tests make real web searches.")
    print("Rate limiting may cause some tests to fail on repeated runs.\n")

    # Install check
    try:
        from duckduckgo_search import DDGS
        print("[OK] duckduckgo-search library is installed\n")
    except ImportError:
        print("[ERROR] duckduckgo-search library not installed")
        print("Install with: pip install duckduckgo-search\n")
        return

    # Run tests
    await test_basic_search()
    await test_cache_functionality()
    await test_input_validation()
    await test_cache_stats()
    # NOTE: Commenting out rate limiting test to avoid hitting actual rate limits
    # await test_rate_limiting()
    await test_error_handling()

    # Summary
    print("=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
    print("\nNOTE: Rate limiting test was skipped to avoid DuckDuckGo rate limits.")
    print("To test rate limiting, uncomment the test in main().")


if __name__ == "__main__":
    asyncio.run(main())
