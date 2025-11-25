# tools/yolo_to_classification.py
import argparse, json, os, random, shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

def load_class_names(root: Path) -> Dict[int, str]:
    """
    Try to infer class names from data.yaml if present.
    Otherwise return a default mapping {0:'dark_circles', 1:'acne'}.
    """
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        # very lightweight yaml reader to avoid PyYAML dep
        names = []
        with open(yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line.startswith("names:"):
                    # names: [a, b, c]
                    arr = line.split(":", 1)[1].strip()
                    arr = arr.strip("[]")
                    names = [x.strip().strip("'\"") for x in arr.split(",") if x.strip()]
                    break
        return {i: n for i, n in enumerate(names)} if names else {0:"dark_circles", 1:"acne"}
    return {0:"dark_circles", 1:"acne"}

def parse_txt(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes=[]
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: 
                continue
            parts=ln.split()
            if len(parts)!=5:
                continue
            cls=int(float(parts[0]))
            x=float(parts[1]); y=float(parts[2]); w=float(parts[3]); h=float(parts[4])
            boxes.append((cls,x,y,w,h))
    return boxes

def yolo_to_xyxy(img_w, img_h, x, y, w, h, pad=0.08):
    """Convert normalized YOLO (cx,cy,w,h) to pixel [x1,y1,x2,y2] with optional padding."""
    cx=x*img_w; cy=y*img_h; bw=w*img_w; bh=h*img_h
    x1 = cx - bw/2; y1 = cy - bh/2
    x2 = cx + bw/2; y2 = cy + bh/2
    # padding
    px = pad*bw; py = pad*bh
    x1-=px; y1-=py; x2+=px; y2+=py
    # clip
    x1=max(0,int(x1)); y1=max(0,int(y1))
    x2=min(img_w,int(x2)); y2=min(img_h,int(y2))
    if x2<=x1 or y2<=y1:
        return None
    return (x1,y1,x2,y2)

def ensure_dirs(base: Path, splits: List[str], class_names: List[str]):
    for split in splits:
        for name in class_names:
            (base / split / name).mkdir(parents=True, exist_ok=True)

def write_split_json(base: Path, meta: dict):
    with open(base / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    p=argparse.ArgumentParser(description="Convert YOLOv11 boxes to classification crops.")
    p.add_argument("--yolo_root", required=True, help="Path containing images/ and labels/")
    p.add_argument("--out_root",  required=True, help="Output root (e.g., data/ )")
    p.add_argument("--class-map", default="", help="JSON like '{\"0\":\"dark_circles\",\"1\":\"acne\"}' to override names")
    p.add_argument("--include-classes", default="", help="Comma ids to include (e.g. '0' or '0,1'). Empty = all.")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pad", type=float, default=0.08, help="Relative padding added around crops")
    args=p.parse_args()

    random.seed(args.seed)
    root=Path(args.yolo_root)
    images_dir=root/"images"
    labels_dir=root/"labels"
    out_root=Path(args.out_root)

    # class names
    class_names_map = load_class_names(root)
    if args.class_map:
        class_names_map = {int(k):v for k,v in json.loads(args.class_map).items()}
    include = None
    if args.include_classes.strip():
        include = set(int(x) for x in args.include_classes.split(",") if x.strip().isdigit())

    # list all stems that have label files
    label_files = sorted(list(labels_dir.glob("*.txt")))
    stems = [lf.stem for lf in label_files]

    # collect all crops info
    crops = []  # (img_path, class_name, box_index)
    for stem in stems:
        img_path = None
        for ext in [".jpg",".jpeg",".png",".JPG",".PNG",".JPEG"]:
            pth = images_dir/(stem+ext)
            if pth.exists():
                img_path = pth
                break
        if not img_path:
            continue
        boxes = parse_txt(labels_dir/(stem+".txt"))
        if not boxes:
            continue
        # load once
        with Image.open(img_path) as im:
            w,h = im.size
            idx=0
            for cls,x,y,w_rel,h_rel in boxes:
                if include is not None and cls not in include:
                    continue
                xyxy = yolo_to_xyxy(w,h,x,y,w_rel,h_rel, pad=args.pad)
                if not xyxy:
                    continue
                crops.append((img_path, cls, xyxy, idx))
                idx+=1

    # class list present in data
    classes_present = sorted({cls for _,cls,_,_ in crops})
    class_names = [class_names_map.get(c, f"class_{c}") for c in classes_present]

    # split by image-level shuffling so crops of the same image go same split
    stems_set = list({c[0].stem for c in crops})
    random.shuffle(stems_set)
    n_val = int(len(stems_set)*args.val_ratio)
    val_imgs = set(stems_set[:n_val])

    ensure_dirs(out_root, ["train","val"], class_names)

    # write crops
    counts = { "train":{n:0 for n in class_names}, "val":{n:0 for n in class_names} }
    for img_path, cls, (x1,y1,x2,y2), idx in crops:
        split = "val" if img_path.stem in val_imgs else "train"
        cname = class_names_map.get(cls, f"class_{cls}")
        with Image.open(img_path) as im:
            crop = im.crop((x1,y1,x2,y2))
            # normalize orientation
            crop = crop.convert("RGB")
            out_dir = out_root / split / cname
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{img_path.stem}_c{cls}_{idx}.jpg"
            crop.save(out_dir/out_name, quality=95)
            counts[split][cname]+=1

    write_split_json(out_root, {
        "source": str(root.resolve()),
        "classes": class_names_map,
        "include": sorted(list(include)) if include is not None else "all",
        "val_ratio": args.val_ratio,
        "pad": args.pad,
        "counts": counts
    })

    print("Done.")
    print("Counts:", json.dumps(counts, indent=2))
    print("Output root:", out_root.resolve())

if __name__ == "__main__":
    main()
