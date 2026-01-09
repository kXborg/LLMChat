"""
Web search utilities.
Handles web search operations using Tavily API.
"""

from tavily import TavilyClient


def perform_web_search(tavily_client: TavilyClient | None, query: str, max_results: int = 5) -> str:
    """
    Perform web search using Tavily and return formatted results.
    Returns empty string if search fails or is not configured.
    
    Args:
        tavily_client: TavilyClient instance (or None if not configured)
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results as a string
    """
    if not tavily_client:
        return ""
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",  # "basic" for faster, "advanced" for more thorough
            max_results=max_results,
            include_answer=True,  # Get a direct answer if available
        )
        
        # Format results for LLM context
        parts = []
        
        # Include direct answer if available
        if response.get("answer"):
            parts.append(f"**Direct Answer:** {response['answer']}")
        
        # Include search results
        results = response.get("results", [])
        if results:
            parts.append("\n**Web Search Results:**")
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:500]  # Limit content length
                parts.append(f"\n{i}. **{title}**\n   URL: {url}\n   {content}")
        
        return "\n".join(parts) if parts else ""
    
    except Exception as e:
        print(f"Web search failed: {e}")
        return ""
