#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_PACK_NAME = "experts.split.ggml_mxfp4.bin"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rewrite a merged GPT-OSS GGML MXFP4 expert pack as gate/up/down split tensors.")
    ap.add_argument("--package-dir", default="")
    ap.add_argument("--src-pack", default="")
    ap.add_argument("--src-manifest", default="")
    ap.add_argument("--out-pack", default="")
    ap.add_argument("--out-manifest", default="")
    ap.add_argument("--hidden-size", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--block-count", type=int, default=0)
    ap.add_argument("--chunk-mib", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-package-manifest", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    package_dir = Path(args.package_dir).resolve() if args.package_dir else None
    package_manifest = load_package_manifest(package_dir)

    src_pack = resolve_src_pack(args, package_dir, package_manifest)
    src_manifest = resolve_src_manifest(args, package_dir, src_pack)
    source_meta = json.loads(src_manifest.read_text(encoding="utf-8-sig"))

    hidden_size = args.hidden_size or int(source_meta.get("hidden_size") or 0)
    layers = args.layers or int(source_meta.get("packed_layers") or source_meta.get("num_hidden_layers") or 0)
    experts = args.experts or int(source_meta.get("packed_experts") or source_meta.get("num_local_experts") or 0)
    block_count = args.block_count or int(source_meta.get("block_count") or math.ceil(hidden_size / 32))
    if hidden_size <= 0 or layers <= 0 or experts <= 0 or block_count <= 0:
        raise SystemExit("missing pack dimensions; pass --hidden-size, --layers, --experts, and --block-count")

    out_pack = resolve_out_pack(args, package_dir, src_pack)
    if out_pack == src_pack:
        raise SystemExit("out-pack must differ from src-pack")
    out_manifest = Path(args.out_manifest).resolve() if args.out_manifest else default_manifest_path(out_pack)
    if (out_pack.exists() or out_manifest.exists()) and not args.force:
        raise SystemExit(f"output exists; pass --force to overwrite: {out_pack}")

    sizes = compute_sizes(hidden_size, block_count, experts)
    expected_bytes = sizes["layer_bytes"] * layers
    src_size = src_pack.stat().st_size
    if src_size < expected_bytes:
        raise SystemExit(f"source pack is smaller than expected: have {src_size}, need {expected_bytes}")

    out_pack.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_pack = out_pack.with_name(out_pack.name + ".tmp")
    if tmp_pack.exists():
        if not args.force:
            raise SystemExit(f"temporary output exists; pass --force to overwrite: {tmp_pack}")
        tmp_pack.unlink()

    chunk_size = max(1, args.chunk_mib) * 1024 * 1024
    with src_pack.open("rb") as src, tmp_pack.open("wb") as dst:
        for layer_index in range(layers):
            rewrite_layer(src, dst, layer_index, sizes, chunk_size)
            print(f"split-pack layer {layer_index + 1}/{layers}", flush=True)

    os.replace(tmp_pack, out_pack)
    manifest = build_manifest(source_meta, src_manifest, src_pack, out_pack, hidden_size, layers, experts, block_count, sizes)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if package_dir is not None and not args.no_package_manifest:
        update_package_manifest(package_dir, out_pack)

    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    return 0


def load_package_manifest(package_dir: Path | None) -> dict[str, Any]:
    if package_dir is None:
        return {}
    path = package_dir / "package.manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_src_pack(args: argparse.Namespace, package_dir: Path | None, package_manifest: dict[str, Any]) -> Path:
    if args.src_pack:
        return Path(args.src_pack).resolve()
    if package_dir is not None:
        destination = package_manifest.get("files", {}).get("expert_pack", {}).get("destination", "")
        if destination:
            candidate = Path(destination)
            if not candidate.is_absolute():
                candidate = package_dir / candidate
            if candidate.exists():
                return candidate.resolve()
        candidate = package_dir / "moe_ggml_pack" / "experts.ggml_mxfp4.bin"
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit("missing source pack; pass --src-pack or --package-dir")


def resolve_src_manifest(args: argparse.Namespace, package_dir: Path | None, src_pack: Path) -> Path:
    if args.src_manifest:
        return Path(args.src_manifest).resolve()
    candidates = [
        src_pack.with_name("experts.ggml_mxfp4_manifest.json"),
        src_pack.with_name(src_pack.name.replace(".bin", "_manifest.json")),
    ]
    if package_dir is not None:
        candidates.append(package_dir / "moe_ggml_pack" / "experts.ggml_mxfp4_manifest.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit("missing source manifest; pass --src-manifest")


def resolve_out_pack(args: argparse.Namespace, package_dir: Path | None, src_pack: Path) -> Path:
    if args.out_pack:
        return Path(args.out_pack).resolve()
    if package_dir is not None:
        return (package_dir / "moe_ggml_pack" / DEFAULT_SPLIT_PACK_NAME).resolve()
    return src_pack.with_name(DEFAULT_SPLIT_PACK_NAME).resolve()


def default_manifest_path(out_pack: Path) -> Path:
    if out_pack.name.endswith(".bin"):
        return out_pack.with_name(out_pack.name[:-4] + "_manifest.json")
    return out_pack.with_name(out_pack.name + ".manifest.json")


def compute_sizes(hidden_size: int, block_count: int, experts: int) -> dict[str, int]:
    matrix_expert_bytes = hidden_size * block_count * 17
    gate_up_expert_bytes = matrix_expert_bytes * 2
    gate_all_bytes = matrix_expert_bytes * experts
    up_all_bytes = matrix_expert_bytes * experts
    gate_up_all_bytes = gate_up_expert_bytes * experts
    down_all_bytes = matrix_expert_bytes * experts
    gate_up_bias_all_bytes = hidden_size * 2 * 4 * experts
    down_bias_all_bytes = hidden_size * 4 * experts
    return {
        "matrix_expert_bytes": matrix_expert_bytes,
        "gate_up_expert_bytes": gate_up_expert_bytes,
        "gate_all_bytes": gate_all_bytes,
        "up_all_bytes": up_all_bytes,
        "gate_up_all_bytes": gate_up_all_bytes,
        "down_all_bytes": down_all_bytes,
        "gate_up_bias_all_bytes": gate_up_bias_all_bytes,
        "down_bias_all_bytes": down_bias_all_bytes,
        "layer_bytes": gate_up_all_bytes + down_all_bytes + gate_up_bias_all_bytes + down_bias_all_bytes,
    }


def rewrite_layer(src, dst, layer_index: int, sizes: dict[str, int], chunk_size: int) -> None:
    layer_base = layer_index * sizes["layer_bytes"]
    old_gate_up_base = layer_base
    old_down_base = layer_base + sizes["gate_up_all_bytes"]
    expert_bytes = sizes["matrix_expert_bytes"]
    gate_up_expert_bytes = sizes["gate_up_expert_bytes"]
    experts = sizes["gate_all_bytes"] // expert_bytes

    for expert_id in range(experts):
        src.seek(old_gate_up_base + expert_id * gate_up_expert_bytes)
        copy_exact(src, dst, expert_bytes, chunk_size)
    for expert_id in range(experts):
        src.seek(old_gate_up_base + expert_id * gate_up_expert_bytes + expert_bytes)
        copy_exact(src, dst, expert_bytes, chunk_size)

    src.seek(old_down_base)
    copy_exact(
        src,
        dst,
        sizes["down_all_bytes"] + sizes["gate_up_bias_all_bytes"] + sizes["down_bias_all_bytes"],
        chunk_size,
    )


def copy_exact(src, dst, byte_count: int, chunk_size: int) -> None:
    remaining = byte_count
    while remaining > 0:
        data = src.read(min(remaining, chunk_size))
        if not data:
            raise SystemExit("unexpected end of source pack")
        dst.write(data)
        remaining -= len(data)


def build_manifest(
    source_meta: dict[str, Any],
    src_manifest: Path,
    src_pack: Path,
    out_pack: Path,
    hidden_size: int,
    layers: int,
    experts: int,
    block_count: int,
    sizes: dict[str, int],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for layer_index in range(layers):
        layer_base = layer_index * sizes["layer_bytes"]
        gate_offset = layer_base
        up_offset = gate_offset + sizes["gate_all_bytes"]
        down_offset = up_offset + sizes["up_all_bytes"]
        gate_up_bias_offset = down_offset + sizes["down_all_bytes"]
        down_bias_offset = gate_up_bias_offset + sizes["gate_up_bias_all_bytes"]
        entries.extend(
            [
                entry(layer_index, "ffn_gate_exps.weight", "GGML_TYPE_MXFP4", [hidden_size, hidden_size, experts], gate_offset, sizes["gate_all_bytes"]),
                entry(layer_index, "ffn_up_exps.weight", "GGML_TYPE_MXFP4", [hidden_size, hidden_size, experts], up_offset, sizes["up_all_bytes"]),
                entry(layer_index, "ffn_down_exps.weight", "GGML_TYPE_MXFP4", [hidden_size, hidden_size, experts], down_offset, sizes["down_all_bytes"]),
                entry(layer_index, "ffn_gate_up_exps.bias", "GGML_TYPE_F32", [hidden_size * 2, experts], gate_up_bias_offset, sizes["gate_up_bias_all_bytes"]),
                entry(layer_index, "ffn_down_exps.bias", "GGML_TYPE_F32", [hidden_size, experts], down_bias_offset, sizes["down_bias_all_bytes"]),
            ]
        )

    return {
        "format": "infinitum_v2_ggml_split_expert_pack_v0",
        "source_format": source_meta.get("format"),
        "source_manifest": str(src_manifest),
        "source_pack": str(src_pack),
        "pack_file": out_pack.name,
        "hidden_size": hidden_size,
        "num_hidden_layers": layers,
        "num_local_experts": experts,
        "packed_layers": layers,
        "packed_experts": experts,
        "block_count": block_count,
        "layout": "gate_all_up_all_down_all_biases",
        "entries": entries,
        "summary": {
            "status": "expert_split_pack_ready",
            "pack_bytes": out_pack.stat().st_size,
            "matrix_expert_bytes": sizes["matrix_expert_bytes"],
            "gate_all_bytes_per_layer": sizes["gate_all_bytes"],
            "up_all_bytes_per_layer": sizes["up_all_bytes"],
            "down_all_bytes_per_layer": sizes["down_all_bytes"],
            "layer_bytes": sizes["layer_bytes"],
        },
    }


def entry(layer_index: int, suffix: str, dtype: str, shape: list[int], byte_offset: int, byte_length: int) -> dict[str, Any]:
    return {
        "name": f"blk.{layer_index}.{suffix}",
        "dtype": dtype,
        "shape": shape,
        "byte_offset": byte_offset,
        "byte_length": byte_length,
    }


def update_package_manifest(package_dir: Path, out_pack: Path) -> None:
    manifest_path = package_dir / "package.manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = manifest.setdefault("files", {})
    try:
        destination = out_pack.relative_to(package_dir).as_posix()
    except ValueError:
        destination = str(out_pack)
    files["expert_split_pack"] = {"destination": destination}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=4), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
