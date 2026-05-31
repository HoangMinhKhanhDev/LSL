"""Versioned binary checkpoint utilities for LSLCoreModel."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


FRAME_MAGIC = b"LSLCKPT2"
CHECKPOINT_VERSION = 2


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pack_frame(payload: bytes, header: Dict[str, Any]) -> bytes:
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return FRAME_MAGIC + struct.pack("<QQ", len(header_bytes), len(payload)) + header_bytes + payload


def _iter_frames(data: bytes):
    offset = 0
    while offset < len(data):
        if data[offset: offset + len(FRAME_MAGIC)] != FRAME_MAGIC:
            raise ValueError("Invalid LSL binary checkpoint magic")
        offset += len(FRAME_MAGIC)
        if offset + 16 > len(data):
            raise ValueError("Truncated LSL checkpoint frame")
        header_len, payload_len = struct.unpack("<QQ", data[offset: offset + 16])
        offset += 16
        header_end = offset + int(header_len)
        payload_end = header_end + int(payload_len)
        if payload_end > len(data):
            raise ValueError("Truncated LSL checkpoint payload")
        header = json.loads(data[offset:header_end].decode("utf-8"))
        payload = data[header_end:payload_end]
        yield header, payload
        offset = payload_end


def checkpoint_frame(
    obj: Any,
    *,
    compression_level: int = 9,
    mode: str = "full",
    parent: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(raw, int(compression_level))
    header: Dict[str, Any] = {
        "format": "LSLCoreModelBinary",
        "version": CHECKPOINT_VERSION,
        "mode": str(mode),
        "compression": "zlib",
        "timestamp": _now_utc(),
        "raw_bytes": len(raw),
        "payload_bytes": len(compressed),
        "compression_ratio": len(compressed) / max(1.0, float(len(raw))),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_sha256": hashlib.sha256(compressed).hexdigest(),
        "parent": parent,
    }
    if extra:
        header.update(extra)
    return _pack_frame(compressed, header), header


def save_checkpoint(
    obj: Any,
    path: str,
    *,
    compression_level: int = 9,
    mode: str = "full",
    parent: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    frame, header = checkpoint_frame(
        obj,
        compression_level=compression_level,
        mode=mode,
        parent=parent,
        extra=extra,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(frame)
    header = dict(header)
    header["path"] = str(out)
    header["file_bytes"] = len(frame)
    return header


def append_checkpoint(
    obj: Any,
    path: str,
    *,
    compression_level: int = 9,
    parent: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    frame, header = checkpoint_frame(
        obj,
        compression_level=compression_level,
        mode="incremental_journal_frame",
        parent=parent,
        extra=extra,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "ab") as f:
        f.write(frame)
    header = dict(header)
    header["path"] = str(out)
    header["appended_bytes"] = len(frame)
    return header


def load_checkpoint(path: str) -> Any:
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(FRAME_MAGIC):
        return pickle.loads(data)
    last_header = None
    last_payload = None
    for header, payload in _iter_frames(data):
        last_header = header
        last_payload = payload
    if last_payload is None or last_header is None:
        raise ValueError("Empty LSL checkpoint")
    if hashlib.sha256(last_payload).hexdigest() != last_header.get("payload_sha256"):
        raise ValueError("LSL checkpoint payload hash mismatch")
    if last_header.get("compression") == "zlib":
        raw = zlib.decompress(last_payload)
    else:
        raw = last_payload
    if hashlib.sha256(raw).hexdigest() != last_header.get("raw_sha256"):
        raise ValueError("LSL checkpoint raw hash mismatch")
    return pickle.loads(raw)


def load_legacy_json_checkpoint(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        wrapped = json.load(f)
    if wrapped.get("format") != "LSLCoreModelBase64Pickle":
        raise ValueError("Unsupported LSLCoreModel JSON checkpoint format")
    return pickle.loads(base64.b64decode(wrapped["payload_b64"]))


def migrate_checkpoint(input_path: str, output_path: str, *, compression_level: int = 9) -> Dict[str, Any]:
    source = Path(input_path)
    if source.suffix.lower() == ".json":
        obj = load_legacy_json_checkpoint(str(source))
    else:
        obj = load_checkpoint(str(source))
    return save_checkpoint(
        obj,
        output_path,
        compression_level=compression_level,
        mode="migrated",
        parent=os.path.abspath(str(source)),
    )
