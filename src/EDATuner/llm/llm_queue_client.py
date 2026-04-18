# src/llm_queue_client.py
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .openai_client import _parse_json_loose


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding) as f:
        f.write(text)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _messages_preview(messages: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for m in messages or []:
        role = str(m.get("role", "")).strip() or "unknown"
        content = m.get("content", "")
        if isinstance(content, (dict, list)):
            try:
                content = json.dumps(content, ensure_ascii=False, indent=2)
            except Exception:
                content = str(content)
        out.append(f"[{role}]\n{str(content)}\n")
    return "\n".join(out).strip() + "\n"


def _stable_hash(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _extract_text_from_response(resp: Any) -> str:
    """Best-effort extraction of assistant text from many possible response shapes."""
    if resp is None:
        return ""

    if isinstance(resp, str):
        return resp

    if isinstance(resp, dict):
        for k in ("content", "text", "assistant", "output_text"):
            v = resp.get(k)
            if isinstance(v, str):
                return v

        msg = resp.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return str(msg.get("content") or "")

        # OpenAI /v1/chat/completions style
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return str(msg.get("content") or "")
                if isinstance(c0.get("text"), str):
                    return str(c0.get("text") or "")

    return str(resp)


def _safe_relative_path(path: Path, root: Path) -> str:
    """Return a privacy-safe path relative to the queue root when possible."""
    try:
        path = Path(path).resolve()
        root = Path(root).resolve()
        return str(path.relative_to(root))
    except Exception:
        return Path(path).name


@dataclass
class _QueuedRequest:
    req_id: str
    kind: str
    inbox_dir: Path
    outbox_dir: Path

    @property
    def request_dir(self) -> Path:
        return self.inbox_dir / self.req_id

    @property
    def response_dir(self) -> Path:
        return self.outbox_dir / self.req_id


class FileQueueLLMClient:
    """
    Offline / no-network LLM client.

    Instead of calling an API, it writes a request package to:
      {queue_dir}/inbox/{request_id}/request.json

    And expects an external process to write a response to:
      {queue_dir}/outbox/{request_id}/response.json

    The request_id is deterministic (hash of the request content),
    so repeated identical requests will reuse existing responses.

    Minimal response format expected (any of these work):
      - {"content": "..."}
      - {"text": "..."}
      - OpenAI chat-completions style JSON (choices[0].message.content)
      - For structured JSON: {"parsed": {...}} (preferred)

    This client will:
      - create the request package if missing
      - wait/poll until response exists (or timeout)
      - return (text, raw_response_dict) / (parsed_json, raw_response_dict)
    """

    def __init__(
        self,
        *,
        queue_dir: Union[str, Path],
        timeout_sec: int = 60,
        poll_interval_sec: float = 0.5,
        nonblocking: bool = False,
    ) -> None:
        self.queue_dir = Path(str(queue_dir)).resolve()
        self.timeout_sec = int(timeout_sec)
        self.poll_interval_sec = float(poll_interval_sec)
        self.nonblocking = bool(nonblocking)

        self.inbox_dir = self.queue_dir / "inbox"
        self.outbox_dir = self.queue_dir / "outbox"
        self.archive_dir = self.queue_dir / "archive"

    def is_ready(self) -> bool:
        try:
            self.inbox_dir.mkdir(parents=True, exist_ok=True)
            self.outbox_dir.mkdir(parents=True, exist_ok=True)
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    def _build_request_id(
        self,
        *,
        kind: str,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_output_tokens: int,
        json_schema: Optional[Dict[str, Any]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        stable = {
            "kind": str(kind),
            "model": str(model),
            "temperature": float(temperature),
            "max_output_tokens": int(max_output_tokens),
            "messages": messages or [],
            "json_schema": (json_schema or None),
            "extra_payload": (extra_payload or None),
        }
        h = _stable_hash(stable)
        return f"{kind}-{h[:12]}"

    def _write_request_package(
        self,
        q: _QueuedRequest,
        *,
        request_json: Dict[str, Any],
        prompt_preview: str,
        json_schema: Optional[Dict[str, Any]],
    ) -> None:
        if q.request_dir.exists():
            return

        # Write into a temp dir and rename atomically so the watcher never sees partial files.
        tmp_dir = q.request_dir.with_name(q.request_dir.name + ".tmp")
        if tmp_dir.exists():
            # stale tmp, remove
            for _ in range(2):
                try:
                    import shutil

                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    break
                except Exception:
                    time.sleep(0.05)

        tmp_dir.mkdir(parents=True, exist_ok=True)

        _atomic_write_json(tmp_dir / "request.json", request_json)
        _atomic_write_text(tmp_dir / "prompt.txt", prompt_preview)
        if json_schema is not None:
            _atomic_write_json(tmp_dir / "schema.json", json_schema)

        _atomic_write_text(tmp_dir / "READY", "ready\n")
        tmp_dir.replace(q.request_dir)

        # Pre-create response dir (optional convenience for the external worker)
        q.response_dir.mkdir(parents=True, exist_ok=True)

    def _find_response_file(self, q: _QueuedRequest) -> Optional[Path]:
        candidates = [
            q.response_dir / "response.json",
            self.outbox_dir / f"{q.req_id}.json",
            q.request_dir / "response.json",
            q.request_dir / "out" / "response.json",
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def _wait_for_response(self, q: _QueuedRequest, timeout_sec: int) -> Path:
        if self.nonblocking:
            raise TimeoutError(f"nonblocking mode: response not ready for {q.req_id}")

        t0 = time.time()
        while True:
            p = self._find_response_file(q)
            if p is not None:
                return p
            if (time.time() - t0) >= float(timeout_sec):
                raise TimeoutError(f"timeout waiting for response: {q.req_id}")
            time.sleep(self.poll_interval_sec)

    def _read_response_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # if worker wrote plain text instead of JSON
            return {"content": path.read_text(encoding="utf-8", errors="ignore")}

    def create_text(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_sec: Optional[int] = None,
        **_: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        mt = int(max_tokens if max_tokens is not None else (max_output_tokens if max_output_tokens is not None else 1200))
        kind = "text"
        req_id = self._build_request_id(
            kind=kind,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=mt,
            json_schema=None,
            extra_payload=extra_payload,
        )
        q = _QueuedRequest(req_id=req_id, kind=kind, inbox_dir=self.inbox_dir, outbox_dir=self.outbox_dir)

        request_json = {
            "id": req_id,
            "kind": kind,
            "created_ts": time.time(),
            "model": model,
            "temperature": float(temperature),
            "max_output_tokens": int(mt),
            "messages": messages or [],
            "json_schema": None,
            "extra_payload": (extra_payload or None),
            "metadata": (metadata or {}),
        }
        self._write_request_package(q, request_json=request_json, prompt_preview=_messages_preview(messages), json_schema=None)

        resp_path = self._wait_for_response(q, timeout_sec=int(timeout_sec if timeout_sec is not None else self.timeout_sec))
        resp = self._read_response_json(resp_path)
        text = _extract_text_from_response(resp).strip()
        raw: Dict[str, Any] = resp if isinstance(resp, dict) else {"content": str(resp)}
        raw.setdefault("_response_path", _safe_relative_path(resp_path, self.queue_dir))
        raw.setdefault("_request_id", req_id)
        return text, raw

    def create_structured_json(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        json_schema: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_sec: Optional[int] = None,
        **_: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        mt = int(max_tokens if max_tokens is not None else (max_output_tokens if max_output_tokens is not None else 1200))
        kind = "structured_json"
        req_id = self._build_request_id(
            kind=kind,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=mt,
            json_schema=json_schema,
            extra_payload=extra_payload,
        )
        q = _QueuedRequest(req_id=req_id, kind=kind, inbox_dir=self.inbox_dir, outbox_dir=self.outbox_dir)

        request_json = {
            "id": req_id,
            "kind": kind,
            "created_ts": time.time(),
            "model": model,
            "temperature": float(temperature),
            "max_output_tokens": int(mt),
            "messages": messages or [],
            "json_schema": (json_schema or {}),
            "extra_payload": (extra_payload or None),
            "metadata": (metadata or {}),
        }
        self._write_request_package(
            q,
            request_json=request_json,
            prompt_preview=_messages_preview(messages),
            json_schema=(json_schema or {}),
        )

        resp_path = self._wait_for_response(q, timeout_sec=int(timeout_sec if timeout_sec is not None else self.timeout_sec))
        resp = self._read_response_json(resp_path)

        raw: Dict[str, Any] = resp if isinstance(resp, dict) else {"content": str(resp)}
        raw.setdefault("_response_path", _safe_relative_path(resp_path, self.queue_dir))
        raw.setdefault("_request_id", req_id)

        # preferred: worker provides parsed
        parsed = raw.get("parsed")
        if isinstance(parsed, dict):
            return parsed, raw

        # otherwise parse from text
        text = _extract_text_from_response(resp)
        parsed2 = _parse_json_loose(text)
        if not isinstance(parsed2, dict):
            raise ValueError("structured output is not an object")
        return parsed2, raw
