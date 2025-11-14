"""
Research RAG System - Storage and retrieval for deep research findings

Stores research findings with:
- Vector embeddings for semantic search
- Metadata (source, reliability, category)
- Efficient retrieval by relevance and reliability
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime


@dataclass
class StoredFinding:
    """Research finding with embedding"""
    id: str
    content: str
    source_url: str
    source_title: str
    category: str
    reliability_score: float
    timestamp: str
    key_entities: List[str]
    contradictions: List[str]
    embedding: Optional[np.ndarray] = None


class ResearchRAG:
    """
    RAG system for research findings with semantic search.

    Uses sentence embeddings for semantic retrieval and metadata
    for filtering by reliability, category, etc.
    """

    def __init__(self, storage_dir: str = "./research_storage"):
        """
        Initialize RAG system.

        Args:
            storage_dir: Directory to store research data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        self.findings: Dict[str, StoredFinding] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_model = None

        # Try to load sentence-transformers
        self._init_embedding_model()

        # Load existing data
        self._load_data()

    def _init_embedding_model(self):
        """Initialize sentence-transformers model if available"""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[RAG] Sentence transformer model loaded")
        except ImportError:
            print("[RAG] Warning: sentence-transformers not installed")
            print("[RAG] Install with: pip install sentence-transformers")
            print("[RAG] Falling back to keyword-based retrieval")
            self.embedding_model = None

    def _generate_id(self, finding_content: str, source_url: str) -> str:
        """Generate unique ID for finding"""
        import hashlib
        unique_str = f"{finding_content[:100]}:{source_url}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        if self.embedding_model is None:
            return None

        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"[RAG] Error generating embedding: {e}")
            return None

    def add_findings(self, findings: List[Any]) -> int:
        """
        Add research findings to RAG system.

        Args:
            findings: List of ResearchFinding objects

        Returns:
            Number of findings added
        """
        added_count = 0

        for finding in findings:
            # Generate ID
            finding_id = self._generate_id(finding.content, finding.source_url)

            # Skip if already exists
            if finding_id in self.findings:
                continue

            # Generate embedding
            embedding = self._embed_text(finding.content)

            # Create stored finding
            stored = StoredFinding(
                id=finding_id,
                content=finding.content,
                source_url=finding.source_url,
                source_title=finding.source_title,
                category=finding.category,
                reliability_score=finding.reliability_score,
                timestamp=finding.timestamp.isoformat(),
                key_entities=finding.key_entities,
                contradictions=finding.contradictions,
                embedding=embedding
            )

            self.findings[finding_id] = stored
            added_count += 1

        # Rebuild embedding matrix
        if added_count > 0:
            self._rebuild_embedding_matrix()
            print(f"[RAG] Added {added_count} new findings (total: {len(self.findings)})")

        return added_count

    def _rebuild_embedding_matrix(self):
        """Rebuild numpy array of all embeddings"""
        if self.embedding_model is None:
            return

        embeddings = []
        for finding in self.findings.values():
            if finding.embedding is not None:
                embeddings.append(finding.embedding)

        if embeddings:
            self.embeddings = np.vstack(embeddings)
        else:
            self.embeddings = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_reliability: float = 0.0,
        category: Optional[str] = None
    ) -> List[StoredFinding]:
        """
        Retrieve relevant findings for query.

        Args:
            query: Search query
            top_k: Number of results to return
            min_reliability: Minimum reliability score
            category: Filter by category (optional)

        Returns:
            List of relevant StoredFinding objects
        """
        if not self.findings:
            return []

        # Filter by metadata
        candidates = [
            f for f in self.findings.values()
            if f.reliability_score >= min_reliability
            and (category is None or f.category == category)
        ]

        if not candidates:
            return []

        # If no embedding model, return by reliability
        if self.embedding_model is None or self.embeddings is None:
            candidates.sort(key=lambda f: f.reliability_score, reverse=True)
            return candidates[:top_k]

        # Semantic search
        try:
            query_embedding = self._embed_text(query)
            if query_embedding is None:
                # Fallback to reliability sorting
                candidates.sort(key=lambda f: f.reliability_score, reverse=True)
                return candidates[:top_k]

            # Calculate cosine similarity
            candidate_embeddings = np.vstack([
                f.embedding for f in candidates if f.embedding is not None
            ])

            if len(candidate_embeddings) == 0:
                candidates.sort(key=lambda f: f.reliability_score, reverse=True)
                return candidates[:top_k]

            # Normalize
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            candidate_norms = candidate_embeddings / np.linalg.norm(
                candidate_embeddings, axis=1, keepdims=True
            )

            # Cosine similarity
            similarities = np.dot(candidate_norms, query_norm)

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Return top findings
            valid_candidates = [f for f in candidates if f.embedding is not None]
            return [valid_candidates[i] for i in top_indices]

        except Exception as e:
            print(f"[RAG] Error during retrieval: {e}")
            # Fallback to reliability sorting
            candidates.sort(key=lambda f: f.reliability_score, reverse=True)
            return candidates[:top_k]

    def get_by_category(self, category: str, limit: int = 20) -> List[StoredFinding]:
        """Get findings by category, sorted by reliability"""
        findings = [
            f for f in self.findings.values()
            if f.category == category
        ]
        findings.sort(key=lambda f: f.reliability_score, reverse=True)
        return findings[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        if not self.findings:
            return {
                "total_findings": 0,
                "categories": {},
                "avg_reliability": 0.0
            }

        category_counts = {}
        for f in self.findings.values():
            category_counts[f.category] = category_counts.get(f.category, 0) + 1

        avg_reliability = sum(f.reliability_score for f in self.findings.values()) / len(self.findings)

        return {
            "total_findings": len(self.findings),
            "categories": category_counts,
            "avg_reliability": avg_reliability,
            "has_embeddings": self.embeddings is not None
        }

    def save(self):
        """Save RAG data to disk"""
        try:
            # Save findings metadata (without embeddings - too large)
            metadata = {}
            for fid, finding in self.findings.items():
                metadata[fid] = {
                    "id": finding.id,
                    "content": finding.content,
                    "source_url": finding.source_url,
                    "source_title": finding.source_title,
                    "category": finding.category,
                    "reliability_score": finding.reliability_score,
                    "timestamp": finding.timestamp,
                    "key_entities": finding.key_entities,
                    "contradictions": finding.contradictions
                }

            metadata_path = self.storage_dir / "findings_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save embeddings separately
            if self.embeddings is not None:
                embeddings_path = self.storage_dir / "embeddings.pkl"
                with open(embeddings_path, 'wb') as f:
                    pickle.dump({
                        'embeddings': self.embeddings,
                        'ids': list(self.findings.keys())
                    }, f)

            print(f"[RAG] Saved {len(self.findings)} findings to {self.storage_dir}")

        except Exception as e:
            print(f"[RAG] Error saving data: {e}")

    def _load_data(self):
        """Load RAG data from disk"""
        try:
            metadata_path = self.storage_dir / "findings_metadata.json"
            if not metadata_path.exists():
                return

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Load embeddings if available
            embeddings_data = None
            embeddings_path = self.storage_dir / "embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, 'rb') as f:
                    embeddings_data = pickle.load(f)

            # Reconstruct findings
            for fid, data in metadata.items():
                embedding = None
                if embeddings_data and fid in embeddings_data['ids']:
                    idx = embeddings_data['ids'].index(fid)
                    embedding = embeddings_data['embeddings'][idx]

                self.findings[fid] = StoredFinding(
                    id=data['id'],
                    content=data['content'],
                    source_url=data['source_url'],
                    source_title=data['source_title'],
                    category=data['category'],
                    reliability_score=data['reliability_score'],
                    timestamp=data['timestamp'],
                    key_entities=data['key_entities'],
                    contradictions=data['contradictions'],
                    embedding=embedding
                )

            # Rebuild embedding matrix
            self._rebuild_embedding_matrix()

            print(f"[RAG] Loaded {len(self.findings)} findings from storage")

        except Exception as e:
            print(f"[RAG] Error loading data: {e}")

    def clear_target(self, target: str):
        """Clear all findings related to a target"""
        to_remove = [
            fid for fid, finding in self.findings.items()
            if target.lower() in finding.content.lower()
        ]

        for fid in to_remove:
            del self.findings[fid]

        self._rebuild_embedding_matrix()
        print(f"[RAG] Removed {len(to_remove)} findings for target: {target}")

    def format_findings_for_prompt(
        self,
        findings: List[StoredFinding],
        max_length: int = 4000
    ) -> str:
        """
        Format findings into prompt context.

        Args:
            findings: List of findings to format
            max_length: Maximum character length

        Returns:
            Formatted string for LLM context
        """
        lines = ["[DEEP RESEARCH FINDINGS]", ""]

        # Group by category
        by_category = {}
        for f in findings:
            if f.category not in by_category:
                by_category[f.category] = []
            by_category[f.category].append(f)

        current_length = 0

        for category, category_findings in sorted(by_category.items()):
            # Sort by reliability within category
            category_findings.sort(key=lambda x: x.reliability_score, reverse=True)

            lines.append(f"\n## {category.upper().replace('_', ' ')}")
            lines.append("")

            for finding in category_findings:
                # Format finding
                finding_text = f"- [{finding.reliability_score:.1f}] {finding.content}"
                finding_text += f"\n  Source: {finding.source_title}"

                if finding.contradictions:
                    finding_text += f"\n  Contradictions: {', '.join(finding.contradictions)}"

                if finding.key_entities:
                    finding_text += f"\n  Entities: {', '.join(finding.key_entities[:5])}"

                finding_text += "\n"

                # Check length
                if current_length + len(finding_text) > max_length:
                    lines.append("\n[... additional findings truncated ...]")
                    break

                lines.append(finding_text)
                current_length += len(finding_text)

            if current_length >= max_length:
                break

        return '\n'.join(lines)
