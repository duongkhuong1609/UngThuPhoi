from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


RAW_PREFIX = "data/raw/archive (1)"
LEGACY_SCRIPT_DIR = Path("archive_unused/scripts_nonproduction")


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent] + list(current.parents):
        if (candidate / "dataset").exists() and (candidate / "archive_unused").exists():
            return candidate
    return Path.cwd().resolve()


def run_command(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def resolve_raw_archive_root(raw_root: Path) -> Path:
    raw_root = raw_root.resolve()
    if (raw_root / "LIDC-IDRI-0001-0200").exists() or any(raw_root.glob("LIDC-IDRI-*")):
        return raw_root
    archive_child = raw_root / "archive (1)"
    if archive_child.exists():
        return archive_child.resolve()
    return raw_root


def remap_raw_path(value: str, raw_archive_root: Path) -> str:
    normalized = value.replace("\\", "/").strip()
    if normalized.startswith(RAW_PREFIX):
        suffix = normalized[len(RAW_PREFIX) :].lstrip("/")
        return (raw_archive_root / suffix).as_posix()
    return normalized


def write_kaggle_index(index_csv: Path, raw_root: Path, output_path: Path) -> dict[str, Any]:
    raw_archive_root = resolve_raw_archive_root(raw_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with index_csv.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    missing = 0
    for row in rows:
        for key in ("dicom_folder", "xml_path"):
            row[key] = remap_raw_path(row.get(key, ""), raw_archive_root)
        if not Path(row["dicom_folder"]).exists() or not Path(row["xml_path"]).exists():
            missing += 1

    with output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "source_index": index_csv.as_posix(),
        "rewritten_index": output_path.as_posix(),
        "raw_archive_root": raw_archive_root.as_posix(),
        "rows": len(rows),
        "rows_with_missing_paths": missing,
    }
    (output_path.parent / "kaggle_index_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def ensure_legacy_scripts(project_root: Path) -> dict[str, Path]:
    legacy_dir = project_root / LEGACY_SCRIPT_DIR
    scripts = {
        "build": legacy_dir / "build_yolo_nodule_dataset.py",
        "eval_details": legacy_dir / "evaluate_yolo_detection_details.py",
    }
    missing = [path.as_posix() for path in scripts.values() if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing legacy localization scripts needed to rebuild YOLO data:\n"
            + "\n".join(missing)
        )
    return scripts


def train_and_evaluate(args: argparse.Namespace, project_root: Path, data_yaml: Path) -> dict[str, Any]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str((project_root / ".ultralytics_config").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((project_root / ".matplotlib_config").resolve()))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. On Kaggle run: pip install -r requirements-yolo.txt"
        ) from exc

    run_project = args.run_project.resolve()
    run_project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        plots=True,
        project=str(run_project),
        name=args.run_name,
        exist_ok=True,
        optimizer=args.optimizer,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
    )

    best_weights = run_project / args.run_name / "weights" / "best.pt"
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.nms_iou,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=str(run_project),
        name=f"{args.run_name}_test_eval",
        exist_ok=True,
    )

    test_images = args.output_dir.resolve() / "images" / "test"
    if test_images.exists():
        best_model.predict(
            source=str(test_images),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            device=args.device,
            save=True,
            max_det=args.max_det,
            project=str(run_project),
            name=f"{args.run_name}_test_predictions",
            exist_ok=True,
            verbose=False,
        )

    payload = {
        "model": args.model,
        "best_weights": best_weights.as_posix(),
        "data_yaml": data_yaml.as_posix(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "map50": float(getattr(metrics.box, "map50", 0.0)),
        "map50_95": float(getattr(metrics.box, "map", 0.0)),
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
    }

    metrics_path = run_project / args.run_name / "kaggle_test_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def maybe_zip_artifacts(args: argparse.Namespace) -> str | None:
    if not args.export_zip:
        return None
    run_dir = args.run_project.resolve() / args.run_name
    if not run_dir.exists():
        return None
    zip_base = args.run_project.resolve() / f"{args.run_name}_artifacts"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(run_dir))
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kaggle-friendly pipeline: rewrite raw LIDC paths, build YOLO nodule "
            "dataset from XML edgeMap contours, train on GPU, evaluate, and export artifacts."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True, help="Kaggle raw LIDC root or its archive (1) child.")
    parser.add_argument("--index-csv", type=Path, default=Path("dataset/dataset_index.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/yolo_nodule"))
    parser.add_argument("--run-project", type=Path, default=Path("/kaggle/working/yolo_runs"))
    parser.add_argument("--run-name", default="yolo_nodule_gpu")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--min-box-size", type=int, default=3)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.10)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.20)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=0.20)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--export-zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    scripts = ensure_legacy_scripts(project_root)

    index_csv = args.index_csv if args.index_csv.is_absolute() else project_root / args.index_csv
    output_dir = args.output_dir.resolve()
    rewritten_index = output_dir / "kaggle_dataset_index.csv"
    data_yaml = output_dir / "dataset.yaml"
    manifest = output_dir / "manifest.csv"

    summary: dict[str, Any] = {
        "project_root": project_root.as_posix(),
        "raw_root": args.raw_root.resolve().as_posix(),
        "output_dir": output_dir.as_posix(),
        "run_project": args.run_project.resolve().as_posix(),
        "run_name": args.run_name,
    }

    if not args.skip_build:
        summary["rewritten_index"] = write_kaggle_index(index_csv, args.raw_root, rewritten_index)
        run_command(
            [
                sys.executable,
                str(scripts["build"]),
                "--index-csv",
                str(rewritten_index),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(args.seed),
                "--margin",
                str(args.margin),
                "--min-box-size",
                str(args.min_box_size),
                "--merge-iou",
                str(args.merge_iou),
                "--max-examples",
                str(args.max_examples),
            ]
            + ([] if args.max_patients is None else ["--max-patients", str(args.max_patients)]),
            cwd=project_root,
        )

    if not data_yaml.exists():
        raise SystemExit(f"Missing dataset.yaml: {data_yaml}")

    if not args.skip_train:
        summary["ultralytics_metrics"] = train_and_evaluate(args, project_root, data_yaml)

    if manifest.exists():
        details_output = args.run_project.resolve() / args.run_name / f"detection_details_conf{int(args.conf * 100):03d}.json"
        weights = args.run_project.resolve() / args.run_name / "weights" / "best.pt"
        if weights.exists():
            run_command(
                [
                    sys.executable,
                    str(scripts["eval_details"]),
                    "--weights",
                    str(weights),
                    "--manifest",
                    str(manifest),
                    "--split",
                    "test",
                    "--imgsz",
                    str(args.imgsz),
                    "--conf",
                    str(args.conf),
                    "--nms-iou",
                    str(args.nms_iou),
                    "--match-iou",
                    str(args.match_iou),
                    "--device",
                    str(args.device),
                    "--output",
                    str(details_output),
                ],
                cwd=project_root,
            )
            summary["detection_details"] = details_output.as_posix()

    zip_path = maybe_zip_artifacts(args)
    if zip_path:
        summary["artifact_zip"] = zip_path

    summary_path = args.run_project.resolve() / args.run_name / "kaggle_yolo_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nKaggle YOLO pipeline complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
