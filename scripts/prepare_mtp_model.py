"""Fetch missing metadata for the supplied shards; never replace local files."""
import argparse
import json
import struct
import urllib.request
from pathlib import Path

REPO = "jwkirchenbauer/Qwen3-4B-Inst-2507-MTP"
REVISION = "5273e79569b30ca334fe4093d32a225b327fde0a"
FILES = ["config.json", "generation_config.json", "model.safetensors.index.json",
         "tokenizer.json", "tokenizer_config.json", "added_tokens.json",
         "special_tokens_map.json", "chat_template.jinja"]


def fetch(name, destination):
    if not destination.exists():
        data = urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}").read()
        # Exclusive creation prevents accidental replacement if another process
        # created a file while the download was in progress.
        with destination.open("xb") as f:
            f.write(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("vendor/models/qwen3-4b-Inst-2507-MTP"))
    p.add_argument("--reference-code", type=Path)
    args = p.parse_args()
    args.model.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch(name, args.model / name)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = args.model / shard
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        assert max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__") + 8 + header_len == path.stat().st_size, shard
        headers[shard] = header
    for tensor, shard in index["weight_map"].items():
        assert tensor in headers[shard], tensor
    if args.reference_code:
        args.reference_code.mkdir(parents=True, exist_ok=True)
        for name in ["configuration_qwen3.py", "modeling_qwen3.py"]:
            fetch(name, args.reference_code / name)
    provenance = args.model / "mtp-source.json"
    if not provenance.exists():
        provenance.write_text(json.dumps({"repo": REPO, "revision": REVISION,
            "validation": "shard headers, sizes, and index membership; no full-file hash"}, indent=2))
    print(f"Validated {len(index['weight_map'])} indexed tensors in {len(headers)} shards.")


if __name__ == "__main__":
    main()
