from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
from ultralytics import YOLO


# ============================================================
# CASE 2 - K3 PPE COMPLIANCE MONITORING USING COMPUTER VISION
# ------------------------------------------------------------
# Fitur yang dicakup:
# 1) Deteksi pekerja dari CCTV/video/webcam
# 2) Identifikasi PPE: helmet, vest, gloves, safety_shoes, goggles
# 3) Penilaian kepatuhan sesuai policy/standar per zona industri
# 4) Notifikasi pelanggaran real-time (overlay, log, optional webhook)
# 5) Pembuatan laporan otomatis (CSV, XLSX, HTML, evidences)
# 6) Preprocessing untuk low light / debu / smoke ringan
# 7) Tracking untuk mengurangi notifikasi berulang
#
# Catatan penting:
# - Agar seluruh PPE bisa terdeteksi, Anda tetap membutuhkan MODEL PPE
#   hasil training custom dengan class yang sesuai.
# - Script ini siap dipakai untuk training + inference + reporting.
# - Jika Anda memakai model umum COCO (mis. yolo11n.pt), biasanya hanya
#   class 'person' yang tersedia. Maka sistem tetap jalan untuk deteksi
#   pekerja, tetapi penilaian PPE penuh baru optimal setelah custom training.
# ============================================================


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: float, digits: int = 4) -> float:
    return float(round(float(x), digits))


def beep_windows() -> None:
    try:
        import winsound

        winsound.Beep(2200, 300)
    except Exception:
        pass


# ------------------------------------------------------------
# Preprocessing untuk CCTV yang gelap / kontras rendah / berkabut ringan
# ------------------------------------------------------------
def apply_gamma(frame: np.ndarray, gamma: float = 1.3) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)


def apply_clahe_bgr(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def simple_dehaze_like(frame: np.ndarray) -> np.ndarray:
    # Bukan dehazing ilmiah penuh, tetapi cukup membantu untuk kabut/debu ringan
    blurred = cv2.GaussianBlur(frame, (0, 0), 5)
    sharpened = cv2.addWeighted(frame, 1.35, blurred, -0.35, 0)
    return sharpened


def preprocess_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return frame
    out = frame.copy()
    if mode in {"gamma", "all"}:
        out = apply_gamma(out, gamma=1.35)
    if mode in {"clahe", "all"}:
        out = apply_clahe_bgr(out)
    if mode in {"dehaze", "all"}:
        out = simple_dehaze_like(out)
    return out


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------
@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class PersonState:
    track_id: int
    first_seen_at: str
    last_seen_at: str
    frame_last_seen: int
    violations_consecutive: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    notifications_sent: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_compliance: str = "UNKNOWN"


# ------------------------------------------------------------
# Class aliasing agar lebih fleksibel terhadap nama class dataset
# ------------------------------------------------------------
ALIASES = {
    "person": {"person", "worker", "human", "operator", "employee"},
    "helmet": {"helmet", "hardhat", "hard_hat", "safety_helmet"},
    "vest": {"vest", "safety_vest", "reflective_vest", "hi_vis_vest", "jacket"},
    "gloves": {"gloves", "glove", "hand_gloves", "safety_gloves"},
    "safety_shoes": {"safety_shoes", "safety_shoe", "shoes", "shoe", "boots", "boot"},
    "goggles": {"goggles", "goggle", "safety_goggles", "glasses", "eyewear"},
}


def canonical_class_name(raw_name: str) -> str:
    raw = raw_name.strip().lower()
    for canonical, names in ALIASES.items():
        if raw in names:
            return canonical
    return raw


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def point_in_box(point: Tuple[float, float], box: Tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_intersection(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return float((x_right - x_left) * (y_bottom - y_top))


def intersection_over_area(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> float:
    inter = box_intersection(inner, outer)
    inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / inner_area


# ------------------------------------------------------------
# Policy helpers
# ------------------------------------------------------------
def load_policy(policy_path: str, zone_name: str) -> dict:
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    zones = policy.get("zones", {})
    if zone_name not in zones:
        raise ValueError(
            f"zone_name '{zone_name}' tidak ditemukan di policy. Pilihan tersedia: {list(zones.keys())}"
        )
    return zones[zone_name]


# ------------------------------------------------------------
# Reporting and notification
# ------------------------------------------------------------
class Reporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.report_dir = output_dir / "reports"
        self.evidence_dir = output_dir / "evidences"
        self.log_dir = output_dir / "logs"
        ensure_dir(self.report_dir)
        ensure_dir(self.evidence_dir)
        ensure_dir(self.log_dir)

        self.events_csv = self.report_dir / "violation_events.csv"
        self.events_jsonl = self.log_dir / "violation_events.jsonl"
        self.summary_xlsx = self.report_dir / "k3_summary.xlsx"
        self.summary_html = self.report_dir / "k3_summary.html"

        if not self.events_csv.exists():
            with open(self.events_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "camera_name",
                        "zone_name",
                        "track_id",
                        "violation_type",
                        "missing_ppe",
                        "detected_ppe",
                        "required_ppe",
                        "confidence_mean",
                        "evidence_path",
                    ]
                )

    def save_event(self, event: dict) -> None:
        with open(self.events_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    event["timestamp"],
                    event["camera_name"],
                    event["zone_name"],
                    event["track_id"],
                    event["violation_type"],
                    ", ".join(event["missing_ppe"]),
                    ", ".join(event["detected_ppe"]),
                    ", ".join(event["required_ppe"]),
                    safe_float(event["confidence_mean"]),
                    str(event["evidence_path"]),
                ]
            )

        with open(self.events_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def save_evidence(self, frame: np.ndarray, filename: str) -> Path:
        path = self.evidence_dir / filename
        cv2.imwrite(str(path), frame)
        return path

    def finalize_summary(self) -> None:
        if not self.events_csv.exists():
            return

        df = pd.read_csv(self.events_csv)
        if df.empty:
            html = """
            <html><head><meta charset='utf-8'><title>K3 Summary</title></head>
            <body><h1>K3 PPE Compliance Report</h1><p>Tidak ada pelanggaran yang tercatat.</p></body></html>
            """
            self.summary_html.write_text(html, encoding="utf-8")
            with pd.ExcelWriter(self.summary_xlsx, engine="openpyxl") as writer:
                pd.DataFrame([{"info": "Tidak ada pelanggaran yang tercatat."}]).to_excel(
                    writer, sheet_name="summary", index=False
                )
            return

        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date.astype(str)
        by_type = df.groupby("violation_type").size().reset_index(name="count")
        by_date = df.groupby("date").size().reset_index(name="count")
        by_camera = df.groupby("camera_name").size().reset_index(name="count")
        by_zone = df.groupby("zone_name").size().reset_index(name="count")
        by_track = df.groupby("track_id").size().reset_index(name="count")

        with pd.ExcelWriter(self.summary_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="events", index=False)
            by_type.to_excel(writer, sheet_name="by_type", index=False)
            by_date.to_excel(writer, sheet_name="by_date", index=False)
            by_camera.to_excel(writer, sheet_name="by_camera", index=False)
            by_zone.to_excel(writer, sheet_name="by_zone", index=False)
            by_track.to_excel(writer, sheet_name="by_track", index=False)

        html = f"""
        <html>
        <head>
            <meta charset='utf-8'>
            <title>K3 PPE Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; }}
                h1, h2 {{ color: #1b365d; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
                th, td {{ border: 1px solid #d9d9d9; padding: 8px; text-align: left; }}
                th {{ background: #f2f4f7; }}
                .badge {{ display: inline-block; padding: 4px 8px; border-radius: 999px; background: #ffe8e8; color: #9f1239; }}
            </style>
        </head>
        <body>
            <h1>K3 PPE Compliance Report</h1>
            <p>Dibuat otomatis pada: <strong>{now_str()}</strong></p>
            <p><span class='badge'>Total pelanggaran: {len(df)}</span></p>
            <h2>Rekap per Jenis Pelanggaran</h2>
            {by_type.to_html(index=False)}
            <h2>Rekap per Tanggal</h2>
            {by_date.to_html(index=False)}
            <h2>Rekap per Kamera</h2>
            {by_camera.to_html(index=False)}
            <h2>Rekap per Zona</h2>
            {by_zone.to_html(index=False)}
            <h2>Seluruh Event</h2>
            {df.to_html(index=False)}
        </body>
        </html>
        """
        self.summary_html.write_text(html, encoding="utf-8")


class Notifier:
    def __init__(self, webhook_url: Optional[str], cooldown_seconds: int = 10, enable_beep: bool = True):
        self.webhook_url = webhook_url
        self.cooldown_seconds = cooldown_seconds
        self.enable_beep = enable_beep
        self.last_sent_at: Dict[str, float] = {}

    def can_send(self, key: str) -> bool:
        last = self.last_sent_at.get(key, 0.0)
        return (time.time() - last) >= self.cooldown_seconds

    def mark_sent(self, key: str) -> None:
        self.last_sent_at[key] = time.time()

    def notify(self, event: dict, evidence_image_path: Optional[Path] = None) -> None:
        notif_key = f"{event['track_id']}::{event['violation_type']}"
        if not self.can_send(notif_key):
            return

        print(
            f"[{event['timestamp']}] VIOLATION | camera={event['camera_name']} | track={event['track_id']} | "
            f"missing={event['missing_ppe']}"
        )

        if self.enable_beep:
            beep_windows()

        if self.webhook_url:
            payload = {
                "text": (
                    f"[K3 ALERT] camera={event['camera_name']} zone={event['zone_name']} "
                    f"track={event['track_id']} missing={', '.join(event['missing_ppe'])} "
                    f"time={event['timestamp']}"
                ),
                "event": event,
            }
            try:
                if evidence_image_path and evidence_image_path.exists():
                    with open(evidence_image_path, "rb") as img_f:
                        payload["evidence_base64"] = base64.b64encode(img_f.read()).decode("utf-8")
                requests.post(self.webhook_url, json=payload, timeout=5)
            except Exception as exc:
                print(f"[WARN] Gagal mengirim webhook: {exc}")

        self.mark_sent(notif_key)


# ------------------------------------------------------------
# Main engine
# ------------------------------------------------------------
class PPEComplianceMonitor:
    def __init__(
        self,
        weights_path: str,
        policy_path: str,
        zone_name: str,
        camera_name: str,
        output_dir: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        preprocess_mode: str = "all",
        violation_frames_threshold: int = 5,
        enable_beep: bool = True,
        webhook_url: Optional[str] = None,
        save_video: bool = True,
    ) -> None:
        self.model = YOLO(weights_path)
        self.names = self.model.names
        self.policy_zone = load_policy(policy_path, zone_name)
        self.zone_name = zone_name
        self.camera_name = camera_name
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        self.reporter = Reporter(self.output_dir)
        self.notifier = Notifier(webhook_url=webhook_url, enable_beep=enable_beep)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.preprocess_mode = preprocess_mode
        self.violation_frames_threshold = violation_frames_threshold
        self.person_states: Dict[int, PersonState] = {}
        self.frame_index = 0
        self.save_video = save_video
        self.policy_grace_frames = violation_frames_threshold

        self.required_ppe: List[str] = [canonical_class_name(x) for x in self.policy_zone.get("required_ppe", [])]
        self.allowed_missing_grace_seconds: int = int(self.policy_zone.get("allowed_missing_grace_seconds", 0))

        available_classes = {canonical_class_name(str(v)) for _, v in self.names.items()}
        print(f"[INFO] Model classes tersedia: {sorted(available_classes)}")
        print(f"[INFO] Required PPE untuk zone '{zone_name}': {self.required_ppe}")
        if "person" not in available_classes:
            raise ValueError("Model weights Anda tidak memiliki class 'person'.")
        missing_from_model = [c for c in self.required_ppe if c not in available_classes]
        if missing_from_model:
            print(
                "[WARNING] Class PPE berikut tidak ditemukan di model weights: "
                f"{missing_from_model}. Sistem tetap jalan, tetapi deteksi PPE tersebut tidak akan akurat/tersedia."
            )

    def _extract_detections(self, result, frame_shape: Tuple[int, int, int]) -> List[Detection]:
        detections: List[Detection] = []
        h, w = frame_shape[:2]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy().astype(float)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([-1] * len(xyxy))

        for i in range(len(xyxy)):
            class_name = canonical_class_name(str(self.names[int(cls[i])]))
            x1, y1, x2, y2 = clamp_bbox(*xyxy[i], w=w, h=h)
            track_id = None if int(ids[i]) < 0 else int(ids[i])
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(conf[i]),
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                )
            )
        return detections

    def _is_valid_ppe_for_person(self, ppe: Detection, person: Detection) -> bool:
        # Kombinasi rule spatial agar asosiasi PPE ke person lebih stabil
        cx, cy = ppe.center
        x1, y1, x2, y2 = person.bbox
        pw = max(1, x2 - x1)
        ph = max(1, y2 - y1)

        if not point_in_box((cx, cy), person.bbox):
            if intersection_over_area(ppe.bbox, person.bbox) < 0.35:
                return False

        rel_y = (cy - y1) / ph

        if ppe.class_name == "helmet":
            return 0.00 <= rel_y <= 0.30
        if ppe.class_name == "goggles":
            return 0.02 <= rel_y <= 0.35
        if ppe.class_name == "vest":
            return 0.20 <= rel_y <= 0.75
        if ppe.class_name == "gloves":
            return 0.20 <= rel_y <= 0.90
        if ppe.class_name == "safety_shoes":
            return 0.60 <= rel_y <= 1.00
        return True

    def _assign_ppe_to_persons(
        self,
        persons: List[Detection],
        ppes: List[Detection],
    ) -> Dict[int, Dict[str, List[Detection]]]:
        person_to_ppe: Dict[int, Dict[str, List[Detection]]] = defaultdict(lambda: defaultdict(list))

        for ppe in ppes:
            best_person_id = None
            best_score = -1.0
            for person in persons:
                if person.track_id is None:
                    continue
                if not self._is_valid_ppe_for_person(ppe, person):
                    continue
                score = intersection_over_area(ppe.bbox, person.bbox)
                if point_in_box(ppe.center, person.bbox):
                    score += 0.5
                if score > best_score:
                    best_score = score
                    best_person_id = person.track_id
            if best_person_id is not None:
                person_to_ppe[best_person_id][ppe.class_name].append(ppe)

        return person_to_ppe

    def _person_state(self, track_id: int, bbox: Tuple[int, int, int, int]) -> PersonState:
        if track_id not in self.person_states:
            self.person_states[track_id] = PersonState(
                track_id=track_id,
                first_seen_at=now_str(),
                last_seen_at=now_str(),
                frame_last_seen=self.frame_index,
                last_bbox=bbox,
            )
        st = self.person_states[track_id]
        st.last_seen_at = now_str()
        st.frame_last_seen = self.frame_index
        st.last_bbox = bbox
        return st

    def _cleanup_old_states(self, stale_after_frames: int = 180) -> None:
        stale_ids = [
            track_id
            for track_id, st in self.person_states.items()
            if (self.frame_index - st.frame_last_seen) > stale_after_frames
        ]
        for track_id in stale_ids:
            del self.person_states[track_id]

    def _build_event(
        self,
        person: Detection,
        detected_ppe: List[str],
        missing_ppe: List[str],
        evidence_path: Path,
        confidence_mean: float,
    ) -> dict:
        violation_type = "missing_" + "_and_".join(missing_ppe)
        return {
            "timestamp": now_str(),
            "camera_name": self.camera_name,
            "zone_name": self.zone_name,
            "track_id": int(person.track_id if person.track_id is not None else -1),
            "violation_type": violation_type,
            "missing_ppe": missing_ppe,
            "detected_ppe": detected_ppe,
            "required_ppe": self.required_ppe,
            "confidence_mean": safe_float(confidence_mean),
            "evidence_path": str(evidence_path),
        }

    def _draw_overlay(
        self,
        frame: np.ndarray,
        persons: List[Detection],
        person_to_ppe: Dict[int, Dict[str, List[Detection]]],
        summary: Dict[str, int],
    ) -> np.ndarray:
        out = frame.copy()
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            track_id = person.track_id if person.track_id is not None else -1
            ppe_map = person_to_ppe.get(track_id, {})
            detected = sorted(set(ppe_map.keys()))
            missing = [x for x in self.required_ppe if x not in detected]
            compliant = len(missing) == 0
            color = (0, 200, 0) if compliant else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id} | {'COMPLIANT' if compliant else 'VIOLATION'}"
            cv2.putText(out, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(
                out,
                f"PPE: {', '.join(detected) if detected else '-'}",
                (x1, min(out.shape[0] - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
            )
            if missing:
                cv2.putText(
                    out,
                    f"Missing: {', '.join(missing)}",
                    (x1, min(out.shape[0] - 10, y2 + 42)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 255),
                    2,
                )

        # Status box kiri atas
        panel_h = 140
        cv2.rectangle(out, (10, 10), (450, panel_h), (30, 30, 30), -1)
        cv2.putText(out, f"Camera: {self.camera_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(out, f"Zone: {self.zone_name}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(
            out,
            f"Workers: {summary['workers']} | Violations: {summary['violations']} | Compliant: {summary['compliant']}",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            out,
            f"Required PPE: {', '.join(self.required_ppe)}",
            (20, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
        return out

    def process_source(self, source: str) -> None:
        cap = cv2.VideoCapture(0 if source == "0" else source)
        if not cap.isOpened():
            raise RuntimeError(f"Gagal membuka source video: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        writer = None
        if self.save_video:
            ensure_dir(self.output_dir / "videos")
            out_path = self.output_dir / "videos" / f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_w, frame_h),
            )
            print(f"[INFO] Annotated video akan disimpan ke: {out_path}")

        self.policy_grace_frames = max(self.violation_frames_threshold, int(round(self.allowed_missing_grace_seconds * fps)))
        print(f"[INFO] Grace frames aktif: {self.policy_grace_frames}")
        print("[INFO] Tekan tombol 'q' untuk keluar.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream selesai atau frame tidak terbaca.")
                break

            self.frame_index += 1
            processed = preprocess_frame(frame, self.preprocess_mode)

            results = self.model.track(
                processed,
                persist=True,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                tracker="bytetrack.yaml",
            )
            result = results[0]
            detections = self._extract_detections(result, processed.shape)

            persons = [d for d in detections if d.class_name == "person"]
            ppes = [d for d in detections if d.class_name in {"helmet", "vest", "gloves", "safety_shoes", "goggles"}]
            person_to_ppe = self._assign_ppe_to_persons(persons, ppes)

            violation_count = 0
            compliant_count = 0

            for person in persons:
                if person.track_id is None:
                    continue
                st = self._person_state(person.track_id, person.bbox)
                ppe_map = person_to_ppe.get(person.track_id, {})
                detected_ppe = sorted(set(ppe_map.keys()))
                missing_ppe = [item for item in self.required_ppe if item not in detected_ppe]
                compliant = len(missing_ppe) == 0
                st.last_compliance = "COMPLIANT" if compliant else "VIOLATION"

                if compliant:
                    compliant_count += 1
                    for req in self.required_ppe:
                        st.violations_consecutive[req] = 0
                else:
                    violation_count += 1
                    conf_values = [x.confidence for detections_for_type in ppe_map.values() for x in detections_for_type]
                    confidence_mean = float(np.mean(conf_values)) if conf_values else 0.0

                    # counter bertahap per jenis PPE yang hilang
                    for req in self.required_ppe:
                        if req in missing_ppe:
                            st.violations_consecutive[req] += 1
                        else:
                            st.violations_consecutive[req] = 0

                    confirmed_missing = [
                        req for req in missing_ppe if st.violations_consecutive[req] >= self.policy_grace_frames
                    ]
                    if confirmed_missing:
                        event_key = "missing::" + "__".join(sorted(confirmed_missing))
                        last_event_frame = st.notifications_sent.get(event_key, -10**9)
                        if (self.frame_index - last_event_frame) >= self.policy_grace_frames:
                            x1, y1, x2, y2 = person.bbox
                            pad = 20
                            ex1 = max(0, x1 - pad)
                            ey1 = max(0, y1 - pad)
                            ex2 = min(frame.shape[1] - 1, x2 + pad)
                            ey2 = min(frame.shape[0] - 1, y2 + pad)
                            evidence = frame[ey1:ey2, ex1:ex2].copy()
                            filename = (
                                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cam-{self.camera_name}_"
                                f"id-{person.track_id}_missing-{'-'.join(confirmed_missing)}.jpg"
                            )
                            evidence_path = self.reporter.save_evidence(evidence, filename)
                            event = self._build_event(
                                person=person,
                                detected_ppe=detected_ppe,
                                missing_ppe=confirmed_missing,
                                evidence_path=evidence_path,
                                confidence_mean=confidence_mean,
                            )
                            self.reporter.save_event(event)
                            self.notifier.notify(event, evidence_image_path=evidence_path)
                            st.notifications_sent[event_key] = self.frame_index

            summary = {
                "workers": len(persons),
                "violations": violation_count,
                "compliant": compliant_count,
            }
            annotated = self._draw_overlay(frame, persons, person_to_ppe, summary)

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("K3 PPE Compliance Monitoring", annotated)
            self._cleanup_old_states()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        self.reporter.finalize_summary()
        print(f"[INFO] Laporan selesai. Cek folder output: {self.output_dir.resolve()}")


# ------------------------------------------------------------
# Training helper
# ------------------------------------------------------------
def train_model(
    data_yaml: str,
    model_name: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 960,
    batch: int = 8,
    project: str = "runs/ppe_train",
    run_name: str = "exp",
    device: str = "auto",
) -> None:
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=run_name,
        device=device,
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K3 PPE Compliance Monitoring Using Computer Vision")
    sub = parser.add_subparsers(dest="mode", required=True)

    mon = sub.add_parser("monitor", help="Jalankan monitoring realtime/video")
    mon.add_argument("--source", required=True, help="Sumber video: 0 untuk webcam, atau path video")
    mon.add_argument("--weights", required=True, help="Path weights YOLO custom PPE")
    mon.add_argument("--policy", required=True, help="Path file JSON policy PPE")
    mon.add_argument("--zone-name", default="general_factory", help="Nama zona di policy JSON")
    mon.add_argument("--camera-name", default="camera_1", help="Nama kamera/cctv")
    mon.add_argument("--output-dir", default="output", help="Folder output hasil laporan")
    mon.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    mon.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    mon.add_argument(
        "--preprocess",
        default="all",
        choices=["none", "gamma", "clahe", "dehaze", "all"],
        help="Preprocessing frame",
    )
    mon.add_argument(
        "--violation-frames-threshold",
        type=int,
        default=5,
        help="Minimal frame berturut-turut untuk mengkonfirmasi pelanggaran",
    )
    mon.add_argument("--webhook-url", default=None, help="Webhook URL opsional untuk notifikasi")
    mon.add_argument("--disable-beep", action="store_true", help="Matikan bunyi beep notifikasi")
    mon.add_argument("--no-save-video", action="store_true", help="Jangan simpan video hasil anotasi")

    tr = sub.add_parser("train", help="Training model PPE custom")
    tr.add_argument("--data", required=True, help="Path dataset YAML")
    tr.add_argument("--model", default="yolo11n.pt", help="Base model YOLO")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--imgsz", type=int, default=960)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--project", default="runs/ppe_train")
    tr.add_argument("--name", default="exp")
    tr.add_argument("--device", default="auto")

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            run_name=args.name,
            device=args.device,
        )
    elif args.mode == "monitor":
        monitor = PPEComplianceMonitor(
            weights_path=args.weights,
            policy_path=args.policy,
            zone_name=args.zone_name,
            camera_name=args.camera_name,
            output_dir=args.output_dir,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            preprocess_mode=args.preprocess,
            violation_frames_threshold=args.violation_frames_threshold,
            enable_beep=not args.disable_beep,
            webhook_url=args.webhook_url,
            save_video=not args.no_save_video,
        )
        monitor.process_source(args.source)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
