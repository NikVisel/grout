"""Fetch pinned MiMo metadata, preserving local files and validating shard headers."""
import argparse
import json
import struct
import urllib.request
from pathlib import Path

REPO = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
REVISION = "2367e865d009c13ac81713a2878291d33ab28177"
FILES = ["config.json", "generation_config.json", "model.safetensors.index.json",
         "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("vendor/models/MiMo-V2.6-Distill-Qwen-9B"))
    args = parser.parse_args()
    args.model.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = args.model / name
        if not path.exists():
            data = urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}").read()
            with path.open("xb") as f:
                f.write(data)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    actual = {}
    for shard in sorted(set(index.values())):
        path = args.model / shard
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if 8 + size + max(v["data_offsets"][1] for v in tensors.values()) != path.stat().st_size:
            raise ValueError(f"Incomplete shard: {shard}")
        actual.update({name: shard for name in tensors})
    if actual != index:
        raise ValueError("Local tensor names/shard assignments differ from the index")
    path = args.model / "model-source.json"
    if not path.exists():
        with path.open("x") as f:
            json.dump({"repo": REPO, "revision": REVISION,
                       "validation": "headers and lengths; not payload hashes"}, f, indent=2)
    print(f"Validated {len(actual)} tensors. Metadata ready.")


if __name__ == "__main__":
    main()
