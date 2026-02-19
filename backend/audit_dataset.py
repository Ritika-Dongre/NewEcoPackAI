import argparse
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def count_images_in_class(class_dir: str) -> int:
    count = 0
    for entry in os.scandir(class_dir):
        if entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                count += 1
    return count


def scan_dataset(dataset_dir: str):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_counts = {}
    for entry in os.scandir(dataset_dir):
        if entry.is_dir():
            class_counts[entry.name] = count_images_in_class(entry.path)

    if not class_counts:
        raise ValueError(f"No class folders found inside: {dataset_dir}")

    return class_counts


def find_normalization_conflicts(class_names):
    normalized_groups = defaultdict(list)
    for name in class_names:
        normalized_groups[normalize_label(name)].append(name)

    conflicts = []
    for norm_key, originals in normalized_groups.items():
        if len(originals) > 1:
            conflicts.append({
                "normalized_key": norm_key,
                "labels": sorted(originals),
                "issue": "Multiple folder names collapse to same normalized form",
            })
    return conflicts


def find_similar_name_pairs(class_names, ratio_threshold=0.82):
    pairs = []
    names = sorted(class_names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left = names[i]
            right = names[j]
            ratio = SequenceMatcher(None, left.lower(), right.lower()).ratio()
            if ratio >= ratio_threshold:
                pairs.append({
                    "label_a": left,
                    "label_b": right,
                    "similarity": round(ratio, 3),
                })
    return pairs


def audit_dataset(dataset_dir: str, low_sample_threshold: int, ratio_threshold: float):
    class_counts = scan_dataset(dataset_dir)
    class_names = sorted(class_counts.keys())

    total_images = sum(class_counts.values())
    num_classes = len(class_names)
    avg_per_class = total_images / max(1, num_classes)

    low_sample_classes = [
        {"class_name": name, "count": class_counts[name]}
        for name in class_names
        if class_counts[name] < low_sample_threshold
    ]

    class_counts_sorted = [
        {"class_name": k, "count": v}
        for k, v in sorted(class_counts.items(), key=lambda item: item[1])
    ]

    normalization_conflicts = find_normalization_conflicts(class_names)
    similar_name_pairs = find_similar_name_pairs(class_names, ratio_threshold=ratio_threshold)

    return {
        "dataset_dir": os.path.abspath(dataset_dir),
        "summary": {
            "num_classes": num_classes,
            "total_images": total_images,
            "avg_images_per_class": round(avg_per_class, 2),
            "min_images_in_a_class": min(class_counts.values()),
            "max_images_in_a_class": max(class_counts.values()),
            "low_sample_threshold": low_sample_threshold,
        },
        "class_counts_sorted": class_counts_sorted,
        "low_sample_classes": low_sample_classes,
        "normalization_conflicts": normalization_conflicts,
        "similar_name_pairs": similar_name_pairs,
    }


def print_report(report):
    summary = report["summary"]
    print("\n=== DATASET AUDIT REPORT ===")
    print(f"Dataset: {report['dataset_dir']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Total images: {summary['total_images']}")
    print(f"Avg/class: {summary['avg_images_per_class']}")
    print(f"Min images in class: {summary['min_images_in_a_class']}")
    print(f"Max images in class: {summary['max_images_in_a_class']}")

    print("\nLowest-count classes:")
    for item in report["class_counts_sorted"][:10]:
        print(f"  {item['class_name']}: {item['count']}")

    print(f"\nLow-sample classes (< {summary['low_sample_threshold']}):")
    if report["low_sample_classes"]:
        for item in report["low_sample_classes"]:
            print(f"  {item['class_name']}: {item['count']}")
    else:
        print("  None")

    print("\nPotential naming conflicts (normalized duplicates):")
    if report["normalization_conflicts"]:
        for conflict in report["normalization_conflicts"]:
            labels = ", ".join(conflict["labels"])
            print(f"  {labels}")
    else:
        print("  None")

    print("\nPotential typo pairs (high similarity):")
    if report["similar_name_pairs"]:
        for pair in report["similar_name_pairs"][:20]:
            print(f"  {pair['label_a']} <-> {pair['label_b']} (score={pair['similarity']})")
        if len(report["similar_name_pairs"]) > 20:
            print(f"  ... and {len(report['similar_name_pairs']) - 20} more")
    else:
        print("  None")


def main():
    parser = argparse.ArgumentParser(description="Audit image dataset quality for class imbalance and label issues.")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset root (default: dataset)")
    parser.add_argument("--low-threshold", type=int, default=120, help="Minimum images per class target")
    parser.add_argument("--similarity-threshold", type=float, default=0.82, help="Threshold for possible typo name pairs")
    parser.add_argument("--out", default=os.path.join("saved_model", "dataset_audit_report.json"), help="Output JSON path")
    parser.add_argument("--no-save", action="store_true", help="Only print report, do not save JSON")
    args = parser.parse_args()

    report = audit_dataset(
        dataset_dir=args.dataset_dir,
        low_sample_threshold=args.low_threshold,
        ratio_threshold=args.similarity_threshold,
    )
    print_report(report)

    if not args.no_save:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved audit JSON: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
