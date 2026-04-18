#!/usr/bin/env python3
"""
Weighted RAG service with transfer-doc support.
Added explicit indicators for user/custom corpus loading.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _sanitize_path_for_reporting(path: Any) -> str:
    """Return a privacy-safe path string for summaries and logs."""
    try:
        s = str(path or "").strip()
        if not s:
            return ""
        p = Path(s)
        return p.name if p.is_absolute() else s
    except Exception:
        return ""



class DocChunk:
    __slots__ = ("chunk_id", "content", "source", "category", "metadata")

    def __init__(self, chunk_id: str, content: str, source: str = "",
                 category: str = "", metadata: Optional[Dict] = None):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.category = category
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "category": self.category,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocChunk":
        base_meta = d.get("metadata", {})
        if not isinstance(base_meta, dict):
            base_meta = {}
        extra_meta = {
            k: v for k, v in d.items()
            if k not in ("chunk_id", "id", "content", "text", "source", "category", "type", "metadata")
        }
        merged_meta = dict(base_meta)
        merged_meta.update(extra_meta)
        return cls(
            chunk_id=d.get("chunk_id", d.get("id", "")),
            content=d.get("content", d.get("text", "")),
            source=d.get("source", ""),
            category=d.get("category", d.get("type", "")),
            metadata=merged_meta,
        )


def _load_json_corpus(path: Path) -> List[DocChunk]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    chunks: List[DocChunk] = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                c = DocChunk.from_dict(item)
                if not c.chunk_id:
                    c.chunk_id = f"doc_{i}"
                if c.content:
                    chunks.append(c)
            elif isinstance(item, str) and item.strip():
                chunks.append(DocChunk(chunk_id=f"doc_{i}", content=item, source=path.stem))
    elif isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, str) and val.strip():
                chunks.append(DocChunk(chunk_id=str(key), content=val, source=path.stem))
            elif isinstance(val, dict):
                c = DocChunk.from_dict(val)
                c.chunk_id = str(key)
                if c.content:
                    chunks.append(c)
    return chunks


def _load_jsonl_corpus(path: Path) -> List[DocChunk]:
    if not path.is_file():
        return []
    chunks: List[DocChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                c = DocChunk.from_dict(item)
                if not c.chunk_id:
                    c.chunk_id = f"jsonl_{i}"
                if c.content:
                    chunks.append(c)
    return chunks


def _load_jsonl_qa(path: Path) -> List[DocChunk]:
    if not path.is_file():
        return []
    chunks: List[DocChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qa = json.loads(line)
            except Exception:
                continue
            if not isinstance(qa, dict):
                continue
            q = qa.get("question", "")
            a = qa.get("answer", "")
            if not (q and a):
                continue
            refs = qa.get("reference_content", [])
            content = f"Q: {q}\nA: {a}"
            if refs:
                content += "\nRef: " + " ".join(str(r)[:500] for r in refs[:2])
            chunks.append(DocChunk(
                chunk_id=f"qa_{qa.get('id', len(chunks))}",
                content=content,
                source="ORD-QA",
                category=qa.get("type", "qa"),
            ))
    return chunks


def _load_txt_dir(dir_path: Path) -> List[DocChunk]:
    if not dir_path.is_dir():
        return []
    chunks: List[DocChunk] = []
    for f in sorted(dir_path.iterdir()):
        if f.suffix not in (".txt", ".md", ".rst"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                chunks.append(DocChunk(
                    chunk_id=f"file_{f.stem}",
                    content=text,
                    source=f.name,
                    category="custom",
                ))
        except Exception:
            pass
    return chunks


def _load_any_doc_path(path: Path) -> List[DocChunk]:
    if not path.exists():
        return []
    if path.is_dir():
        chunks: List[DocChunk] = []
        for f in sorted(path.iterdir()):
            if f.suffix == ".json":
                chunks.extend(_load_json_corpus(f))
            elif f.suffix == ".jsonl":
                chunks.extend(_load_jsonl_corpus(f))
        chunks.extend(_load_txt_dir(path))
        return chunks
    if path.suffix == ".json":
        return _load_json_corpus(path)
    if path.suffix == ".jsonl":
        return _load_jsonl_corpus(path)
    return []


def _extract_doc_weight(chunk: DocChunk, transfer_weight_boost: float = 1.0) -> float:
    meta = chunk.metadata or {}
    weight = 1.0
    try:
        if "weight" in meta:
            weight = float(meta["weight"])
        elif "recommended_retrieval_weight" in meta:
            weight = float(meta["recommended_retrieval_weight"])
        elif "final_score" in meta:
            weight = float(meta["final_score"])
        elif isinstance(meta.get("retrieval_weight"), dict):
            rw = meta["retrieval_weight"]
            if "recommended_retrieval_weight" in rw:
                weight = float(rw["recommended_retrieval_weight"])
        elif isinstance(meta.get("metadata"), dict):
            inner = meta["metadata"]
            if isinstance(inner.get("retrieval_weight"), dict) and "recommended_retrieval_weight" in inner["retrieval_weight"]:
                weight = float(inner["retrieval_weight"]["recommended_retrieval_weight"])
            elif "final_score" in inner:
                weight = float(inner["final_score"])
    except Exception:
        weight = 1.0

    try:
        if chunk.category == "transfer_rag":
            weight *= float(transfer_weight_boost)
    except Exception:
        pass

    return weight if weight > 0 else 1.0


class _VectorIndex:
    def __init__(self, model_name: str, index_path: Optional[Path], transfer_weight_boost: float = 1.0):
        self.model_name = model_name
        self.index_path = index_path
        self.transfer_weight_boost = float(transfer_weight_boost)
        self._model = None
        self._index = None
        self._chunks: List[DocChunk] = []

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def _encode(self, texts: List[str]):
        import numpy as np
        self._load_model()
        emb = self._model.encode(texts, normalize_embeddings=True, batch_size=64,
                                 show_progress_bar=len(texts) > 200)
        return np.array(emb, dtype=np.float32)

    def build(self, chunks: List[DocChunk]) -> None:
        import faiss
        self._chunks = chunks
        emb = self._encode([c.content for c in chunks])
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
        logger.info(f"[RAG] Built index: {self._index.ntotal} vectors, dim={emb.shape[1]}")
        if self.index_path:
            self._save(emb.shape[1])
        self._release_model()

    def _release_model(self):
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            gc.collect()

    def _save(self, dim: int) -> None:
        import faiss
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        (self.index_path / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self._chunks], ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        (self.index_path / "config.json").write_text(
            json.dumps({"model": self.model_name, "dim": dim, "n": len(self._chunks)}, indent=2) + "\n",
            encoding="utf-8",
        )

    def load(self) -> bool:
        if not self.index_path or not (self.index_path / "index.faiss").is_file():
            return False
        try:
            import faiss
            self._index = faiss.read_index(str(self.index_path / "index.faiss"))
            meta = json.loads((self.index_path / "chunks.json").read_text("utf-8"))
            self._chunks = [DocChunk.from_dict(m) for m in meta]
            logger.info(f"[RAG] Loaded index: {self._index.ntotal} vectors")
            return True
        except Exception as e:
            logger.warning(f"[RAG] Index load failed: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocChunk, float]]:
        if not self._index or self._index.ntotal == 0:
            return []
        q = self._encode([query])
        fetch_k = min(max(top_k * 4, top_k), self._index.ntotal)
        scores, indices = self._index.search(q, fetch_k)
        reranked: List[Tuple[DocChunk, float]] = []
        for s, i in zip(scores[0], indices[0]):
            if 0 <= i < len(self._chunks):
                chunk = self._chunks[i]
                final_score = float(s) * _extract_doc_weight(chunk, self.transfer_weight_boost)
                reranked.append((chunk, final_score))
        reranked.sort(key=lambda x: -x[1])
        return reranked[:top_k]

    def search_multi(self, queries: List[str], top_k: int = 5) -> List[Tuple[DocChunk, float]]:
        seen: Dict[str, float] = {}
        chunk_map: Dict[str, DocChunk] = {}
        for q in queries:
            for chunk, score in self.search(q, top_k):
                old = seen.get(chunk.chunk_id)
                if old is None or score > old:
                    seen[chunk.chunk_id] = score
                    chunk_map[chunk.chunk_id] = chunk
        results = [(chunk_map[k], v) for k, v in seen.items()]
        results.sort(key=lambda x: -x[1])
        return results[:top_k * 2]


_PARAM_QUERIES: Dict[str, str] = {
    "CORE_UTIL": "floorplan core utilization placement density area",
    "ASPECT_RATIO": "floorplan aspect ratio die shape",
    "CORE_MARGIN": "floorplan core margin IO ring spacing",
    "PLACE_CONG_EFFORT": "placement congestion effort optimization",
    "PLACE_TIMING_EFFORT": "placement timing driven optimization effort",
    "PLACE_UNIFORM_DENSITY": "placement uniform density distribution",
    "PLACE_MAX_DENSITY": "placement maximum density target congestion",
    "TD_PLACE": "timing driven placement enable global",
    "PRECTS_SETUP_OPT": "pre-CTS setup time optimization",
    "PRECTS_HOLD_OPT": "pre-CTS hold time optimization fix",
    "DRV_EFFORT": "design rule violation DRV fix effort",
    "CTS_TARGET_SKEW": "clock tree synthesis target skew balance",
    "CTS_POST_OPT": "post-CTS optimization timing",
    "ROUTE_TIMING_DRIVEN": "routing timing driven optimization",
    "ROUTE_SI_DRIVEN": "routing signal integrity crosstalk",
    "ROUTE_EFFORT": "routing effort quality detail",
    "ROUTE_LAYER_EFFORT": "routing layer assignment metal effort",
}

_SYMPTOM_QUERIES: Dict[str, str] = {
    "timing": "fix timing violations setup hold WNS negative slack",
    "congestion": "reduce placement congestion overflow routing",
    "crash": "EDA tool crash common causes fix",
    "timeout": "reduce runtime optimization speed efficiency",
    "drc": "design rule violation DRC fix routing",
    "overflow": "routing overflow congestion global routing",
    "area": "reduce area utilization floorplan placement",
    "power": "reduce power consumption leakage dynamic optimization",
}

_FALLBACK_KNOWLEDGE = """=== EDA Tuning Domain Knowledge (static fallback) ===
- CORE_UTIL: 0.55-0.75 safe. >0.75 congestion risk. Lower = easier timing.
- PLACE_TIMING_EFFORT "high": +WNS, +30% runtime.
- PLACE_CONG_EFFORT "high": -congestion, may hurt timing.
- CTS_TARGET_SKEW 0.05-0.10: typical for medium designs.
- ROUTE_TIMING_DRIVEN true: almost always beneficial for timing.
- ROUTE_SI_DRIVEN true: crosstalk help, +20% route time. Skip for non-critical.
- TD_PLACE=1 + PLACE_TIMING_EFFORT="high": strong timing closure combo.
=== End ==="""


class RAGService:
    def __init__(self, cfg: Dict[str, Any]):
        rc = cfg.get("rag", {}) or {}
        self.enabled = bool(rc.get("enable", False))
        self._corpus_path = rc.get("corpus_path", "")
        self._qa_path = rc.get("qa_path", "")
        self._index_path = Path(rc.get("index_path", "./runs/rag_index")) if rc.get("index_path") else None
        self._model_name = rc.get("embedding_model", "all-MiniLM-L6-v2")
        self._top_k = int(rc.get("top_k", 5))
        self._max_chars = int(rc.get("max_context_chars", 4000))
        self._cadence_docs = rc.get("cadence_docs_path", None)
        self._transfer_docs = rc.get("transfer_docs_path", None)
        self._transfer_weight_boost = float(rc.get("transfer_weight_boost", 1.0))
        self._extra_docs = rc.get("extra_docs", [])
        self._index: Optional[_VectorIndex] = None
        self._ready = False

        self._load_summary: Dict[str, Any] = {
            "used_cached_index": False,
            "corpus_path": _sanitize_path_for_reporting(self._corpus_path),
            "qa_path": _sanitize_path_for_reporting(self._qa_path),
            "cadence_docs_path": _sanitize_path_for_reporting(self._cadence_docs),
            "transfer_docs_path": _sanitize_path_for_reporting(self._transfer_docs),
            "corpus_chunks": 0,
            "qa_chunks": 0,
            "cadence_chunks": 0,
            "transfer_chunks": 0,
            "extra_chunks": 0,
            "total_chunks": 0,
            "has_custom_corpus": False,
            "has_transfer_corpus": False,
        }

    def build_index(self, force: bool = False) -> None:
        if not self.enabled:
            logger.info("[RAG] Disabled in config (rag.enable=false)")
            return

        self._index = _VectorIndex(self._model_name, self._index_path, self._transfer_weight_boost)

        if not force and self._index.load():
            self._ready = True
            self._load_summary["used_cached_index"] = True
            self._load_summary["total_chunks"] = int(getattr(self._index._index, "ntotal", 0))
            logger.info("[RAG] Using cached index from %s", _sanitize_path_for_reporting(self._index_path))
            logger.info(
                "[RAG] Custom corpus mounted status (cached index): has_custom=%s has_transfer=%s total_vectors=%s",
                self._load_summary["has_custom_corpus"],
                self._load_summary["has_transfer_corpus"],
                self._load_summary["total_chunks"],
            )
            return

        all_chunks: List[DocChunk] = []

        if self._corpus_path:
            corpus_chunks = _load_any_doc_path(Path(self._corpus_path))
            all_chunks.extend(corpus_chunks)
            self._load_summary["corpus_chunks"] = len(corpus_chunks)
            logger.info("[RAG] Base corpus loaded: %d chunks from %s", len(corpus_chunks), _sanitize_path_for_reporting(self._corpus_path))

        if self._qa_path:
            qa_chunks = _load_jsonl_qa(Path(self._qa_path))
            all_chunks.extend(qa_chunks)
            self._load_summary["qa_chunks"] = len(qa_chunks)
            logger.info("[RAG] QA corpus loaded: %d chunks from %s", len(qa_chunks), _sanitize_path_for_reporting(self._qa_path))

        if self._cadence_docs:
            cadence_chunks = _load_any_doc_path(Path(self._cadence_docs))
            all_chunks.extend(cadence_chunks)
            self._load_summary["cadence_chunks"] = len(cadence_chunks)
            self._load_summary["has_custom_corpus"] = len(cadence_chunks) > 0
            logger.info("[RAG] Custom local corpus loaded: %d chunks from %s", len(cadence_chunks), _sanitize_path_for_reporting(self._cadence_docs))

        if self._transfer_docs:
            transfer_chunks = _load_any_doc_path(Path(self._transfer_docs))
            for c in transfer_chunks:
                if not c.category:
                    c.category = "transfer_rag"
            all_chunks.extend(transfer_chunks)
            self._load_summary["transfer_chunks"] = len(transfer_chunks)
            self._load_summary["has_transfer_corpus"] = len(transfer_chunks) > 0
            logger.info("[RAG] Transfer/custom RAG loaded: %d chunks from %s", len(transfer_chunks), _sanitize_path_for_reporting(self._transfer_docs))

        extra_count = 0
        for doc in (self._extra_docs or []):
            if isinstance(doc, dict) and "content" in doc:
                all_chunks.append(DocChunk(
                    chunk_id=f"extra_{len(all_chunks)}",
                    content=doc["content"],
                    source=doc.get("title", "extra"),
                    category="extra",
                    metadata={k: v for k, v in doc.items() if k not in ("content", "title")},
                ))
                extra_count += 1
        self._load_summary["extra_chunks"] = extra_count

        if not all_chunks:
            logger.warning("[RAG] No chunks found. RAG inactive.")
            return

        self._load_summary["total_chunks"] = len(all_chunks)
        logger.info(
            "[RAG] Load summary: base=%d qa=%d custom=%d transfer=%d extra=%d total=%d",
            self._load_summary["corpus_chunks"],
            self._load_summary["qa_chunks"],
            self._load_summary["cadence_chunks"],
            self._load_summary["transfer_chunks"],
            self._load_summary["extra_chunks"],
            self._load_summary["total_chunks"],
        )
        logger.info(
            "[RAG] Custom corpus mounted status: has_custom=%s has_transfer=%s",
            self._load_summary["has_custom_corpus"],
            self._load_summary["has_transfer_corpus"],
        )

        self._index.build(all_chunks)
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready and self._index is not None

    @property
    def load_summary(self) -> Dict[str, Any]:
        return dict(self._load_summary)

    @property
    def has_custom_corpus(self) -> bool:
        return bool(self._load_summary.get("has_custom_corpus", False))

    @property
    def has_transfer_corpus(self) -> bool:
        return bool(self._load_summary.get("has_transfer_corpus", False))

    def retrieve(self, query: str, top_k: int = 0, max_chars: int = 0) -> str:
        if not self.is_ready:
            return ""
        top_k = top_k or self._top_k
        max_chars = max_chars or self._max_chars
        return self._format_results(self._index.search(query, top_k), max_chars)

    def retrieve_multi(self, queries: List[str], top_k: int = 0, max_chars: int = 0) -> str:
        if not self.is_ready:
            return ""
        top_k = top_k or self._top_k
        max_chars = max_chars or self._max_chars
        return self._format_results(self._index.search_multi(queries, top_k), max_chars)

    def context_for_candidate_proposal(self, current_params: Dict[str, Any],
                                       failures: Optional[List[str]] = None,
                                       metrics: Optional[Dict[str, Any]] = None) -> str:
        if not self.is_ready:
            return _FALLBACK_KNOWLEDGE
        queries = self._build_queries_from_context(current_params, failures, metrics)
        queries.append("EDA parameter tuning optimization best practices")
        ctx = self.retrieve_multi(queries[:8])
        return ctx if ctx else _FALLBACK_KNOWLEDGE

    def context_for_failure_analysis(self, failure_symptoms: List[str], log_tail: str = "") -> str:
        if not self.is_ready:
            return ""
        queries: List[str] = []
        for sym in failure_symptoms:
            sym_l = str(sym).lower()
            for key, q in _SYMPTOM_QUERIES.items():
                if key in sym_l:
                    queries.append(q)
                    break
            else:
                queries.append(f"EDA {sym} fix debug troubleshoot")
        if log_tail:
            for kw in ("error", "violation", "overflow", "timeout", "fail"):
                if kw in log_tail.lower():
                    queries.append(f"EDA {kw} common causes fix")
                    break
        return self.retrieve_multi(queries[:6]) if queries else ""

    def context_for_sensitivity(self, *param_names: str) -> str:
        if not self.is_ready:
            return ""
        queries = [_PARAM_QUERIES.get(name, f"EDA {name} effect optimization") for name in param_names]
        return self.retrieve_multi(queries[:6])

    def context_for_design_prior(self, current_params: Dict[str, Any],
                                 metrics: Optional[Dict[str, Any]] = None) -> str:
        return self.context_for_candidate_proposal(current_params, failures=None, metrics=metrics)

    def enrich_prompt(self, base_prompt: str, query_hints: Optional[List[str]] = None,
                      max_chars: int = 0) -> str:
        if not self.is_ready or not query_hints:
            return base_prompt
        ctx = self.retrieve_multi(query_hints, max_chars=max_chars or 2000)
        if not ctx:
            return base_prompt
        return f"[Retrieved EDA Knowledge]\\n{ctx}\\n[End of Retrieved Knowledge]\\n\\n{base_prompt}"

    def _build_queries_from_context(self, params: Dict[str, Any],
                                    failures: Optional[List[str]],
                                    metrics: Optional[Dict[str, Any]]) -> List[str]:
        queries: List[str] = []
        if failures:
            for sym in failures:
                sym_l = str(sym).lower()
                for key, q in _SYMPTOM_QUERIES.items():
                    if key in sym_l:
                        queries.append(q)
                        break
                else:
                    queries.append(f"EDA {sym} fix optimization")
        if metrics:
            import math
            wns = metrics.get("wns")
            if isinstance(wns, (int, float)) and math.isfinite(wns) and wns < -0.1:
                queries.append("improve timing WNS negative slack optimization")
            power = metrics.get("power")
            if isinstance(power, (int, float)) and math.isfinite(power) and power > 0:
                queries.append("reduce power consumption EDA optimization")
        for name in list(params.keys())[:5]:
            if name in _PARAM_QUERIES:
                queries.append(_PARAM_QUERIES[name])
        seen = set()
        out: List[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def _format_results(self, results: List[Tuple[DocChunk, float]], max_chars: int) -> str:
        parts: List[str] = []
        total = 0
        for chunk, score in results:
            text = chunk.content.strip()
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    text = text[:remaining] + "..."
                else:
                    break
            tag = chunk.source or chunk.category or ""
            doc_weight = _extract_doc_weight(chunk, self._transfer_weight_boost)
            parts.append(f"[{tag}] (score={score:.3f}, weight={doc_weight:.3f})\\n{text}")
            total += len(text)
        return "\\n\\n".join(parts)


_instance: Optional[RAGService] = None


def get_rag_service(cfg: Optional[Dict[str, Any]] = None) -> RAGService:
    global _instance
    if _instance is None:
        if cfg is None:
            return RAGService({})
        _instance = RAGService(cfg)
        _instance.build_index()
    return _instance


def reset_rag_service() -> None:
    global _instance
    _instance = None
