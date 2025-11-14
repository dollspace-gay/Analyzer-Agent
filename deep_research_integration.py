"""
Deep Research Integration - Orchestrates multi-stage research and synthesis

Coordinates:
1. Deep research gathering (DeepResearchAgent)
2. Storage in RAG system (ResearchRAG)
3. Retrieval and synthesis for analysis
"""

import asyncio
from typing import Dict, Any, Optional, List
from deep_research_agent import DeepResearchAgent, ResearchReport
from research_rag import ResearchRAG, StoredFinding


class DeepResearchIntegration:
    """
    Integration layer for deep research system.

    Manages the multi-stage process of:
    1. Conducting research
    2. Storing findings
    3. Retrieving context for analysis
    """

    def __init__(self, web_search_tool, rag_storage_dir: str = "./research_storage"):
        """
        Initialize deep research integration.

        Args:
            web_search_tool: WebSearchTool instance
            rag_storage_dir: Directory for RAG storage
        """
        self.research_agent = DeepResearchAgent(
            web_search_tool=web_search_tool,
            max_queries=10,
            max_results_per_query=5
        )
        self.rag = ResearchRAG(storage_dir=rag_storage_dir)

    async def research_and_store(self, target: str, force_refresh: bool = False) -> ResearchReport:
        """
        Conduct research on target and store in RAG.

        Args:
            target: Entity to research
            force_refresh: If True, clear existing data and re-research

        Returns:
            ResearchReport with findings
        """
        # Clear existing if forcing refresh
        if force_refresh:
            self.rag.clear_target(target)
            print(f"[DeepResearch] Cleared existing research for: {target}")

        # Conduct research
        report = await self.research_agent.conduct_research(target)

        # Store findings in RAG
        added = self.rag.add_findings(report.findings)
        print(f"[DeepResearch] Stored {added} new findings in RAG")

        # Save RAG data
        self.rag.save()

        return report

    def retrieve_context(
        self,
        query: str,
        target: str,
        max_findings: int = 15,
        min_reliability: float = 0.5
    ) -> str:
        """
        Retrieve relevant research context for analysis.

        Args:
            query: Analysis query
            target: Target entity
            max_findings: Maximum findings to retrieve
            min_reliability: Minimum reliability score

        Returns:
            Formatted context string for LLM
        """
        # Retrieve relevant findings
        findings = self.rag.retrieve(
            query=f"{target} {query}",
            top_k=max_findings,
            min_reliability=min_reliability
        )

        if not findings:
            return ""

        # Format for prompt
        context = self.rag.format_findings_for_prompt(findings, max_length=3000)

        return context

    def get_category_summary(self, target: str) -> Dict[str, List[str]]:
        """
        Get summary of findings by category for target.

        Args:
            target: Target entity

        Returns:
            Dictionary mapping category to key findings
        """
        categories = ["criticism", "controversy", "behavior", "power_structure"]
        summary = {}

        for category in categories:
            findings = self.rag.get_by_category(category, limit=5)
            # Filter to target-relevant findings
            relevant = [
                f.content for f in findings
                if target.lower() in f.content.lower()
            ]
            summary[category] = relevant[:3]  # Top 3 per category

        return summary

    def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return self.rag.get_statistics()

    async def full_research_cycle(
        self,
        target: str,
        analysis_prompt: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Complete research cycle: gather, store, retrieve, format.

        Args:
            target: Entity to research
            analysis_prompt: Analysis query
            force_refresh: Re-research target

        Returns:
            Dictionary with research report and formatted context
        """
        print(f"\n{'='*60}")
        print(f"FULL RESEARCH CYCLE: {target}")
        print(f"{'='*60}\n")

        # Stage 1: Research
        report = await self.research_and_store(target, force_refresh)

        # Stage 2: Retrieve context
        context = self.retrieve_context(
            query=analysis_prompt,
            target=target,
            max_findings=15,
            min_reliability=0.5
        )

        # Stage 3: Get category summary
        summary = self.get_category_summary(target)

        print(f"\n[DeepResearch] Research cycle complete")
        print(f"  - Total findings: {len(report.findings)}")
        print(f"  - Context length: {len(context)} chars")

        return {
            "report": report,
            "context": context,
            "category_summary": summary,
            "statistics": self.get_rag_statistics()
        }
