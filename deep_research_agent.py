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
from difflib import SequenceMatcher

# Optional spaCy for NER (graceful fallback if not available)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Optional rank-bm25 for relevance ranking
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Optional NLTK WordNet for query expansion
try:
    from nltk.corpus import wordnet
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False


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

        # Initialize spaCy NER if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("[DeepResearch] spaCy NER initialized")
            except OSError:
                print("[DeepResearch] spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                print("[DeepResearch] Falling back to regex-based entity extraction")

        # Common stop words for query expansion
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'
        }

    def expand_query_with_synonyms(self, query: str, max_expansions: int = 2) -> List[str]:
        """
        Expand query with synonyms using WordNet.

        Args:
            query: Original query string
            max_expansions: Maximum number of synonym expansions per query

        Returns:
            List of expanded query strings (includes original)
        """
        if not WORDNET_AVAILABLE:
            return [query]  # Return original if WordNet not available

        expanded_queries = [query]  # Always include original

        try:
            # Extract keywords (non-stop words)
            words = query.lower().split()
            keywords = [w for w in words if w not in self.stop_words and len(w) > 3]

            # For each keyword, try to find synonyms
            for keyword in keywords[:3]:  # Limit to first 3 keywords to avoid explosion
                synsets = wordnet.synsets(keyword)
                if synsets:
                    # Get synonyms from the first synset (most common meaning)
                    synonyms = []
                    for lemma in synsets[0].lemmas()[:2]:  # Max 2 synonyms per word
                        syn = lemma.name().replace('_', ' ')
                        if syn.lower() != keyword and syn.lower() not in query.lower():
                            synonyms.append(syn)

                    # Create expanded query by replacing keyword with synonym
                    for synonym in synonyms[:max_expansions]:
                        expanded = query.replace(keyword, synonym, 1)
                        if expanded not in expanded_queries:
                            expanded_queries.append(expanded)

            if len(expanded_queries) > 1:
                print(f"[DeepResearch] Expanded query '{query[:50]}...' to {len(expanded_queries)} variants")

        except Exception as e:
            print(f"[DeepResearch] Query expansion error: {e}, using original only")

        return expanded_queries[:max_expansions + 1]  # Original + max_expansions

    def generate_research_queries(self, target: str, user_prompt: Optional[str] = None) -> List[ResearchQuery]:
        """
        Generate targeted research queries for comprehensive analysis.

        Args:
            target: The entity/organization to research
            user_prompt: Optional user's original question for contextual queries

        Returns:
            List of ResearchQuery objects
        """
        queries = []

        # If user prompt is provided, generate contextual queries (highest priority)
        if user_prompt:
            # Extract key themes from user prompt
            user_lower = user_prompt.lower()

            # Add direct contextual query
            queries.append(ResearchQuery(
                query=f"{target} {user_prompt}",
                category="contextual",
                priority=6  # Highest priority
            ))

            # Theme-based contextual queries
            if any(word in user_lower for word in ['manipulat', 'deceiv', 'mislead', 'propaganda']):
                queries.append(ResearchQuery(
                    query=f"{target} public messaging manipulation claims vs reality",
                    category="contextual",
                    priority=6
                ))

            if any(word in user_lower for word in ['safety', 'safe', 'risk', 'danger']):
                queries.append(ResearchQuery(
                    query=f"{target} safety claims actual safety record incidents",
                    category="contextual",
                    priority=6
                ))

            if any(word in user_lower for word in ['ethic', 'moral', 'responsible']):
                queries.append(ResearchQuery(
                    query=f"{target} ethics statements vs ethical violations",
                    category="contextual",
                    priority=6
                ))

            if any(word in user_lower for word in ['open', 'transparent', 'access']):
                queries.append(ResearchQuery(
                    query=f"{target} transparency openness claims vs actual practices",
                    category="contextual",
                    priority=6
                ))

            if any(word in user_lower for word in ['power', 'control', 'dominan']):
                queries.append(ResearchQuery(
                    query=f"{target} power concentration market control monopoly",
                    category="contextual",
                    priority=6
                ))

        # Standard broad queries (for comprehensive coverage)
        queries.extend([
            # Critics and criticism
            ResearchQuery(
                query=f"{target} criticism controversy",
                category="criticism",
                priority=5
            ),
            ResearchQuery(
                query=f"{target} critics opponents whistleblowers",
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
                query=f"{target} promises vs reality outcomes",
                category="behavior",
                priority=4
            ),

            # Power structures
            ResearchQuery(
                query=f"{target} board members executives leadership funding",
                category="power_structure",
                priority=4
            ),

            # Regulatory and legal
            ResearchQuery(
                query=f"{target} lawsuit legal issues violations",
                category="controversy",
                priority=3
            ),
            ResearchQuery(
                query=f"{target} regulatory investigation complaints",
                category="controversy",
                priority=3
            ),

            # Impact and consequences
            ResearchQuery(
                query=f"{target} impact harm consequences victims",
                category="behavior",
                priority=4
            ),

            # Background (lower priority)
            ResearchQuery(
                query=f"{target} overview history founding",
                category="background",
                priority=2
            ),
        ])

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
        Extract key entities mentioned in text using spaCy NER or regex fallback.

        Args:
            text: Text to analyze
            target: Primary target entity

        Returns:
            List of entity names
        """
        entities = []

        # Try spaCy NER first (more accurate)
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    # Extract PERSON, ORG, GPE (geopolitical entities)
                    if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                        entity_text = ent.text.strip()
                        # Filter out target itself and very short entities
                        if entity_text != target and len(entity_text) > 3:
                            entities.append(entity_text)
            except Exception as e:
                # Fall through to regex on error
                print(f"[DeepResearch] spaCy NER error: {e}, falling back to regex")

        # Fallback to regex patterns if spaCy not available or failed
        if not entities:
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

    def is_low_quality_content(self, text: str) -> bool:
        """
        Detect low-quality content like cookie notices, boilerplate, etc.

        Args:
            text: Text to check

        Returns:
            True if content is low quality and should be filtered out
        """
        text_lower = text.lower()

        # Low-quality patterns
        low_quality_patterns = [
            'performance cookies', 'cookies are used', 'this website uses cookies',
            'privacy policy', 'cookie policy', 'terms of service',
            'sign up for newsletter', 'subscribe now', 'subscribe to',
            'newsletter signup', 'email subscription',
            'accept cookies', 'cookie settings', 'manage cookies',
            'all rights reserved', 'copyright ©',
            'this page uses', 'we use cookies',
            'by continuing to use', 'by using this site'
        ]

        # Check if snippet is mostly boilerplate
        for pattern in low_quality_patterns:
            if pattern in text_lower:
                # If pattern takes up significant portion of short text, filter it
                if len(text) < 200:
                    return True

        # Filter very short snippets (likely truncated/useless)
        if len(text.strip()) < 50:
            return True

        return False

    def deduplicate_findings(self, findings: List[ResearchFinding], similarity_threshold: float = 0.85) -> List[ResearchFinding]:
        """
        Remove duplicate or near-duplicate findings based on content similarity.

        Args:
            findings: List of ResearchFinding objects
            similarity_threshold: Similarity ratio above which findings are considered duplicates (0.0-1.0)

        Returns:
            Deduplicated list of findings
        """
        if not findings:
            return findings

        deduplicated = []
        skipped_count = 0

        for finding in findings:
            is_duplicate = False

            # Compare with existing deduplicated findings
            for existing in deduplicated:
                # Calculate similarity ratio
                similarity = SequenceMatcher(None, finding.content, existing.content).ratio()

                if similarity >= similarity_threshold:
                    # This is a duplicate - keep the one with higher reliability
                    if finding.reliability_score > existing.reliability_score:
                        # Replace existing with this one (higher quality)
                        deduplicated.remove(existing)
                        deduplicated.append(finding)
                    # Otherwise skip this finding
                    is_duplicate = True
                    skipped_count += 1
                    break

            if not is_duplicate:
                deduplicated.append(finding)

        if skipped_count > 0:
            print(f"[DeepResearch] Removed {skipped_count} duplicate findings")

        return deduplicated

    def rank_findings_by_relevance(self, findings: List[ResearchFinding], query: str) -> List[ResearchFinding]:
        """
        Rank findings by BM25 relevance to the query.

        Args:
            findings: List of ResearchFinding objects
            query: The search query to rank against

        Returns:
            Findings sorted by relevance (most relevant first)
        """
        if not findings or not BM25_AVAILABLE:
            if not BM25_AVAILABLE and findings:
                print("[DeepResearch] BM25 not available, skipping relevance ranking. Install with: pip install rank-bm25")
            return findings

        try:
            # Tokenize corpus (simple whitespace tokenization)
            corpus = [finding.content.lower().split() for finding in findings]

            # Create BM25 index
            bm25 = BM25Okapi(corpus)

            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores for each document
            scores = bm25.get_scores(tokenized_query)

            # Sort findings by score (descending)
            ranked_findings = [finding for score, finding in sorted(zip(scores, findings), key=lambda x: x[0], reverse=True)]

            print(f"[DeepResearch] Ranked {len(findings)} findings by BM25 relevance")
            return ranked_findings

        except Exception as e:
            print(f"[DeepResearch] BM25 ranking error: {e}, returning unranked")
            return findings

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

            # If no results, try query expansion as fallback
            if not result.success or not result.output:
                print(f"[DeepResearch] No results for: {query.query}")

                # Try expanded queries for high-priority searches
                if query.priority >= 4 and WORDNET_AVAILABLE:
                    expanded_queries = self.expand_query_with_synonyms(query.query, max_expansions=1)
                    for expanded_q in expanded_queries[1:]:  # Skip first (original)
                        print(f"[DeepResearch] Trying expanded query: {expanded_q}")
                        result = await self.web_search_tool.execute(
                            query=expanded_q,
                            max_results=self.max_results_per_query
                        )
                        if result.success and result.output:
                            break  # Found results with expanded query

                # If still no results, return empty
                if not result.success or not result.output:
                    return findings

            # Process each search result
            for item in result.output:
                # Filter out low-quality content (cookie notices, boilerplate, etc.)
                if self.is_low_quality_content(item['snippet']):
                    continue

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

    async def conduct_research(self, target: str, user_prompt: Optional[str] = None) -> ResearchReport:
        """
        Conduct comprehensive research on target.

        Args:
            target: Entity/organization to research
            user_prompt: Optional user's question for contextual search queries

        Returns:
            ResearchReport with aggregated findings
        """
        print(f"\n{'='*60}")
        print(f"DEEP RESEARCH: {target}")
        if user_prompt:
            print(f"CONTEXT: {user_prompt[:100]}...")
        print(f"{'='*60}\n")

        # Generate research queries (contextual if user_prompt provided)
        queries = self.generate_research_queries(target, user_prompt)
        print(f"[DeepResearch] Generated {len(queries)} research queries")
        if user_prompt:
            contextual_count = sum(1 for q in queries if q.category == "contextual")
            print(f"[DeepResearch]   - {contextual_count} contextual (based on your question)")
            print(f"[DeepResearch]   - {len(queries) - contextual_count} broad (comprehensive coverage)")

        # Execute queries in parallel
        print(f"[DeepResearch] Executing searches...")
        tasks = [self.execute_research_query(query) for query in queries]
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_findings: List[ResearchFinding] = []
        for finding_list in results:
            all_findings.extend(finding_list)

        print(f"[DeepResearch] Collected {len(all_findings)} total findings")

        # Deduplicate findings
        all_findings = self.deduplicate_findings(all_findings)
        print(f"[DeepResearch] After deduplication: {len(all_findings)} unique findings")

        # Rank findings by relevance (BM25)
        ranking_query = user_prompt if user_prompt else target
        all_findings = self.rank_findings_by_relevance(all_findings, ranking_query)

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
