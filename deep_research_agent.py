"""
Deep Research Agent - Multi-source information gathering for structural analysis

Performs comprehensive research by:
1. Generating targeted search queries (critics, controversies, power structures)
2. Gathering information from multiple sources
3. Scoring source reliability
4. Extracting contradictions between claims and behavior
5. Storing findings in RAG system for synthesis
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class ResearchQuery:
    """Represents a research query with metadata"""
    query: str
    category: str  # e.g., "criticism", "controversy", "power_structure", "behavior"
    priority: int = 1  # Higher = more important


@dataclass
class ResearchFinding:
    """Represents a single research finding"""
    content: str
    source_url: str
    source_title: str
    category: str
    reliability_score: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    key_entities: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)


@dataclass
class ResearchReport:
    """Aggregated research findings"""
    target: str
    findings: List[ResearchFinding]
    total_sources: int
    avg_reliability: float
    key_contradictions: List[str]
    power_structures: List[str]
    critics: List[str]
    behavioral_evidence: List[str]


class DeepResearchAgent:
    """
    Multi-stage research agent for gathering structural analysis evidence.

    Generates comprehensive research by:
    - Querying multiple angles (critics, controversies, behavior vs claims)
    - Scoring source reliability
    - Identifying contradictions
    - Building evidence base for analysis
    """

    def __init__(self, web_search_tool, max_queries: int = 10, max_results_per_query: int = 5):
        """
        Initialize deep research agent.

        Args:
            web_search_tool: WebSearchTool instance for executing searches
            max_queries: Maximum number of search queries to generate
            max_results_per_query: Maximum results per search query
        """
        self.web_search_tool = web_search_tool
        self.max_queries = max_queries
        self.max_results_per_query = max_results_per_query

        # Source reliability patterns
        self.high_reliability_patterns = [
            r'\.gov$', r'\.edu$', r'arxiv\.org', r'github\.com',
            r'sec\.gov', r'ftc\.gov'
        ]
        self.medium_reliability_patterns = [
            r'wikipedia\.org', r'reuters\.com', r'apnews\.com',
            r'bloomberg\.com', r'wsj\.com', r'nytimes\.com',
            r'theguardian\.com', r'bbc\.com'
        ]
        # Everything else gets low reliability initially

    def generate_research_queries(self, target: str) -> List[ResearchQuery]:
        """
        Generate targeted research queries for comprehensive analysis.

        Args:
            target: The entity/organization to research

        Returns:
            List of ResearchQuery objects
        """
        queries = [
            # Core information
            ResearchQuery(
                query=f"{target} overview history",
                category="background",
                priority=3
            ),

            # Critics and criticism
            ResearchQuery(
                query=f"{target} criticism controversy",
                category="criticism",
                priority=5
            ),
            ResearchQuery(
                query=f"{target} critics opponents",
                category="criticism",
                priority=4
            ),

            # Behavior vs claims
            ResearchQuery(
                query=f"{target} stated mission vs actual behavior",
                category="behavior",
                priority=5
            ),
            ResearchQuery(
                query=f"{target} promises vs reality",
                category="behavior",
                priority=4
            ),

            # Power structures
            ResearchQuery(
                query=f"{target} board members executives leadership",
                category="power_structure",
                priority=4
            ),
            ResearchQuery(
                query=f"{target} funding sources investors",
                category="power_structure",
                priority=4
            ),

            # Regulatory and legal
            ResearchQuery(
                query=f"{target} lawsuit legal issues",
                category="controversy",
                priority=3
            ),
            ResearchQuery(
                query=f"{target} regulatory investigation",
                category="controversy",
                priority=3
            ),

            # Impact and consequences
            ResearchQuery(
                query=f"{target} impact harm consequences",
                category="behavior",
                priority=4
            ),
        ]

        # Sort by priority and limit
        queries.sort(key=lambda q: q.priority, reverse=True)
        return queries[:self.max_queries]

    def score_source_reliability(self, url: str, title: str) -> float:
        """
        Score source reliability based on URL patterns and title.

        Args:
            url: Source URL
            title: Source title

        Returns:
            Reliability score from 0.0 to 1.0
        """
        url_lower = url.lower()

        # Check high reliability patterns
        for pattern in self.high_reliability_patterns:
            if re.search(pattern, url_lower):
                return 0.9

        # Check medium reliability patterns
        for pattern in self.medium_reliability_patterns:
            if re.search(pattern, url_lower):
                return 0.7

        # Blog posts and opinion pieces - lower reliability
        if any(x in url_lower for x in ['blog', 'opinion', 'editorial']):
            return 0.4

        # Default medium-low
        return 0.5

    def extract_key_entities(self, text: str, target: str) -> List[str]:
        """
        Extract key entities mentioned in text.

        Args:
            text: Text to analyze
            target: Primary target entity

        Returns:
            List of entity names
        """
        entities = []

        # Look for common entity patterns
        # Names with titles (CEO, Director, etc.)
        title_pattern = r'(CEO|CTO|CFO|Director|President|Chair(?:man|woman|person)?|Executive)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        matches = re.findall(title_pattern, text)
        entities.extend([match[1] for match in matches])

        # Organizations (capitalized phrases)
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        matches = re.findall(org_pattern, text)
        entities.extend([m for m in matches if m != target and len(m) > 5])

        return list(set(entities))[:10]  # Limit to 10 unique entities

    def detect_contradictions(self, text: str) -> List[str]:
        """
        Detect potential contradictions in text.

        Args:
            text: Text to analyze

        Returns:
            List of contradiction patterns found
        """
        contradictions = []

        # Contradiction indicators
        indicators = [
            (r'claimed.*but.*actually', 'Claim vs Reality'),
            (r'said.*however.*did', 'Statement vs Action'),
            (r'promised.*yet.*failed', 'Promise vs Outcome'),
            (r'publicly.*privately', 'Public vs Private'),
            (r'stated.*contrary to', 'Statement Contradiction'),
            (r'despite.*continued to', 'Despite Pattern')
        ]

        text_lower = text.lower()
        for pattern, label in indicators:
            if re.search(pattern, text_lower):
                contradictions.append(label)

        return list(set(contradictions))

    async def execute_research_query(self, query: ResearchQuery) -> List[ResearchFinding]:
        """
        Execute a single research query and process results.

        Args:
            query: ResearchQuery to execute

        Returns:
            List of ResearchFinding objects
        """
        findings = []

        try:
            # Execute web search
            result = await self.web_search_tool.execute(
                query=query.query,
                max_results=self.max_results_per_query
            )

            if not result.success or not result.output:
                print(f"[DeepResearch] No results for: {query.query}")
                return findings

            # Process each search result
            for item in result.output:
                # Score reliability
                reliability = self.score_source_reliability(
                    item['url'],
                    item['title']
                )

                # Extract entities
                entities = self.extract_key_entities(
                    item['snippet'],
                    query.query.split()[0]  # Use first word as target
                )

                # Detect contradictions
                contradictions = self.detect_contradictions(item['snippet'])

                # Create finding
                finding = ResearchFinding(
                    content=item['snippet'],
                    source_url=item['url'],
                    source_title=item['title'],
                    category=query.category,
                    reliability_score=reliability,
                    key_entities=entities,
                    contradictions=contradictions
                )

                findings.append(finding)

            print(f"[DeepResearch] Found {len(findings)} results for category: {query.category}")

        except Exception as e:
            print(f"[DeepResearch] Error executing query '{query.query}': {e}")

        return findings

    async def conduct_research(self, target: str) -> ResearchReport:
        """
        Conduct comprehensive research on target.

        Args:
            target: Entity/organization to research

        Returns:
            ResearchReport with aggregated findings
        """
        print(f"\n{'='*60}")
        print(f"DEEP RESEARCH: {target}")
        print(f"{'='*60}\n")

        # Generate research queries
        queries = self.generate_research_queries(target)
        print(f"[DeepResearch] Generated {len(queries)} research queries")

        # Execute queries in parallel
        print(f"[DeepResearch] Executing searches...")
        tasks = [self.execute_research_query(query) for query in queries]
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_findings: List[ResearchFinding] = []
        for finding_list in results:
            all_findings.extend(finding_list)

        print(f"[DeepResearch] Collected {len(all_findings)} total findings")

        # Calculate statistics
        avg_reliability = (
            sum(f.reliability_score for f in all_findings) / len(all_findings)
            if all_findings else 0.0
        )

        # Extract key contradictions (sorted by frequency)
        all_contradictions = []
        for f in all_findings:
            all_contradictions.extend(f.contradictions)
        contradiction_counts = {}
        for c in all_contradictions:
            contradiction_counts[c] = contradiction_counts.get(c, 0) + 1
        key_contradictions = sorted(
            contradiction_counts.keys(),
            key=lambda x: contradiction_counts[x],
            reverse=True
        )[:5]

        # Extract power structures (entities from power_structure category)
        power_structures = []
        for f in all_findings:
            if f.category == "power_structure":
                power_structures.extend(f.key_entities)
        power_structures = list(set(power_structures))[:10]

        # Extract critics (entities from criticism category)
        critics = []
        for f in all_findings:
            if f.category == "criticism":
                critics.extend(f.key_entities)
        critics = list(set(critics))[:10]

        # Extract behavioral evidence (high-reliability behavior findings)
        behavioral_evidence = [
            f.content for f in all_findings
            if f.category == "behavior" and f.reliability_score >= 0.7
        ][:10]

        # Create report
        report = ResearchReport(
            target=target,
            findings=all_findings,
            total_sources=len(set(f.source_url for f in all_findings)),
            avg_reliability=avg_reliability,
            key_contradictions=key_contradictions,
            power_structures=power_structures,
            critics=critics,
            behavioral_evidence=behavioral_evidence
        )

        print(f"\n[DeepResearch] Research complete:")
        print(f"  - Total findings: {len(all_findings)}")
        print(f"  - Unique sources: {report.total_sources}")
        print(f"  - Avg reliability: {report.avg_reliability:.2f}")
        print(f"  - Key contradictions: {len(report.key_contradictions)}")
        print(f"  - Power structures identified: {len(report.power_structures)}")
        print(f"  - Critics identified: {len(report.critics)}")

        return report
