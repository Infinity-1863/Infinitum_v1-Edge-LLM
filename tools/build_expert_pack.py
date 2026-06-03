#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


EXPERT_RE = re.compile(r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<kind>.+)$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a portable external expert pack from GPT-OSS-like HF safetensors.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="")
    ap.add_argument("--pack-name", default="experts.pack.bin")
    ap.add_argument("--manifest-name", default="expert-pack.manifest.json")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    if not index_path.exists():
        raise SystemExit(f"missing safetensors index: {index_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pack_path = out_dir / args.pack_name
    manifest_path = out_dir / args.manifest_name
    if (pack_path.exists() or manifest_path.exists()) and not args.force:
        raise SystemExit(f"output exists; pass --force to overwrite: {out_dir}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = args.model_name or model_dir.name

    expert_tensors = []
    for tensor_name, file_name in sorted(index.get("weight_map", {}).items()):
        match = EXPERT_RE.match(tensor_name)
        if not match:
            continue
        expert_tensors.append(
            {
                "tensor_name": tensor_name,
                "file_name": str(file_name),
                "layer_index": int(match.group("layer")),
                "kind": match.group("kind"),
            }
        )
    if not expert_tensors:
        raise SystemExit("no expert tensors found under model.layers.*.mlp.experts.*")

    entries: list[dict[str, Any]] = []
    total_payload_bytes = 0
    with pack_path.open("wb") as out:
        out.write(b"INFEXP1\0")
        out.write(struct.pack("<Q", 0))
        for item in expert_tensors:
            source_path = model_dir / item["file_name"]
            with safe_open(str(source_path), framework="pt", device="cpu") as reader:
                tensor = reader.get_tensor(item["tensor_name"])
                if tensor.ndim == 0:
                    continue
                expert_count = int(tensor.shape[0])
                for expert_id in range(expert_count):
                    sliced = tensor[expert_id].contiguous().cpu()
                    payload = sliced.view(torch.uint8).numpy().tobytes(order="C")
                    offset = out.tell()
                    out.write(payload)
                    byte_length = len(payload)
                    total_payload_bytes += byte_length
                    entries.append(
                        {
                            "layer_index": int(item["layer_index"]),
                            "expert_id": int(expert_id),
                            "kind": item["kind"],
                            "tensor_name": item["tensor_name"],
                            "source_file": item["file_name"],
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                            "shape": [int(x) for x in sliced.shape],
                            "byte_offset": int(offset),
                            "byte_length": int(byte_length),
                        }
                    )

    manifest = {
        "format": "infinitum_edge_expert_pack_v1",
        "model_name": model_name,
        "source_model": model_dir.name,
        "pack_file": pack_path.name,
        "config": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_local_experts": config.get("num_local_experts"),
        },
        "storage": {
            "mode": "contiguous_binary_slices",
            "header_bytes": 16,
            "endianness": "little",
        },
        "entries": entries,
        "summary": {
            "layer_count": len({entry["layer_index"] for entry in entries}),
            "expert_count_per_layer_seen": _expert_counts(entries),
            "entry_count": len(entries),
            "payload_bytes": total_payload_bytes,
            "pack_bytes": pack_path.stat().st_size,
            "status": "expert_pack_ready" if entries else "empty",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    return 0


def _expert_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    by_layer: dict[int, set[int]] = {}
    for entry in entries:
        by_layer.setdefault(int(entry["layer_index"]), set()).add(int(entry["expert_id"]))
    return {str(layer): len(experts) for layer, experts in sorted(by_layer.items())}


if __name__ == "__main__":
    raise SystemExit(main())
