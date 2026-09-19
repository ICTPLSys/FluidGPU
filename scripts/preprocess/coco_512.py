from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    args = parse_args()
    assert args.images.is_dir(), f"missing image directory: {args.images}"
    assert args.annotations.is_file(), f"missing annotation file: {args.annotations}"
    args.output.mkdir(parents=True, exist_ok=True)
    image_output = args.output / "images"
    image_output.mkdir(exist_ok=True)
    annotations = json.loads(args.annotations.read_text())
    image_rows = annotations.get("images", [])
    assert isinstance(image_rows, list), "COCO annotations must contain an images list"
    if args.limit is not None:
        image_rows = image_rows[: args.limit]

    processed: list[dict[str, str | int]] = []
    for row in image_rows:
        file_name = str(row["file_name"])
        source = args.images / file_name
        assert source.is_file(), f"missing COCO image: {source}"
        output = image_output / file_name
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            resized = ImageOps.fit(image.convert("RGB"), (512, 512), method=Image.Resampling.BICUBIC)
            resized.save(output)
        processed.append(
            {
                "image_id": int(row["id"]),
                "source": str(source),
                "image": str(output),
                "width": 512,
                "height": 512,
            }
        )

    manifest = {
        "status": "ok",
        "images": str(args.images),
        "annotations": str(args.annotations),
        "output": str(args.output),
        "image_count": len(processed),
        "resize": "512x512 center crop/resize",
        "items": processed,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COCO 512x512 AD/AE manifest")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
