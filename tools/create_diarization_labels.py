#!/usr/bin/env python3
"""
Script to create detailed multi-class segment labels (.npy) for Spoof Diarization
from vad_20sil.tar.gz (PartialSpoof v1.3).

Outputs:
  {out_dir}/{split}_seglab_{resolution}.npy
  where each file contains a dict: {utt_id: np.ndarray([frame_labels], dtype=np.int16)}

Label Encoding (Preserving exact PartialSpoof v1.3 / label2num_all semantics):
  0:         nonspeech (silence)
  1:         bonafide (authentic speech)
  2 .. 7:    A01 .. A06 (known spoofing techniques in train/dev)
  8 .. 20:   A07 .. A19 (unseen spoofing techniques in eval)
  100:       nonmix (concatenated boundary part / ConP)
  101:       nonbona (nonspeech pause from bonafide)
  102..120:  nonA01 .. nonA19 (nonspeech pause from spoof attack)
"""

import os
import sys
import tarfile
import argparse
from typing import Dict, List
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Diarization Segment Labels (.npy)")
    parser.add_argument(
        "--tar_path",
        type=str,
        default="/GuestShare_NAS/WorkingSpace/Personal/nghiadq/dataset/vad_20sil.tar.gz",
        help="Path to vad_20sil.tar.gz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/GuestShare_NAS/WorkingSpace/Personal/nghiadq/dataset/database/segment_labels_diarization",
        help="Directory where output .npy files will be saved",
    )
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[0.16, 0.02],
        help="Frame resolutions in seconds (e.g., 0.16 for WavLM model, 0.02 for paper baseline)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.tar_path):
        raise FileNotFoundError(f"Archive not found: {args.tar_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 60)
    print("=== Generating Diarization Segment Labels ===")
    print(f"[*] Input archive : {args.tar_path}")
    print(f"[*] Output dir    : {args.out_dir}")
    print(f"[*] Resolutions   : {args.resolutions}")
    print("=" * 60 + "\n")

    # Data structure: {split: {res: {utt_id: np.ndarray}}}
    splits = ["train", "dev", "eval"]
    data: Dict[str, Dict[float, Dict[str, np.ndarray]]] = {
        s: {r: {} for r in args.resolutions} for s in splits
    }

    print("[*] Reading and parsing vad_20sil.tar.gz...")
    with tarfile.open(args.tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".vad")]
        print(f"[*] Found {len(members)} .vad files in archive.")

        for member in tqdm(members, desc="Processing .vad files"):
            # member.name format: vad_20sil/{train,dev,eval}/{utt_id}.vad
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            split = parts[1]
            if split not in splits:
                continue
            utt_id = os.path.splitext(parts[2])[0]

            f = tar.extractfile(member)
            if f is None:
                continue

            lines = f.read().decode("utf-8").strip().splitlines()
            if not lines:
                continue

            # Parse intervals: (st, et, lab)
            intervals = []
            max_et = 0.0
            for line in lines:
                row = line.strip().split()
                if len(row) >= 3:
                    st = float(row[0])
                    et = float(row[1])
                    lab = int(row[2])
                    intervals.append((st, et, lab))
                    if et > max_et:
                        max_et = et

            if max_et <= 0.0 or not intervals:
                continue

            # Generate labels for each requested resolution
            for r in args.resolutions:
                n_frames = int(np.ceil(max_et / r))
                arr = np.zeros(n_frames, dtype=np.int16)

                for st, et, lab in intervals:
                    sf = int(round(st / r))
                    ef = int(round(et / r))
                    if sf < ef and sf < n_frames:
                        arr[sf : min(ef, n_frames)] = lab

                data[split][r][utt_id] = arr

    # Save to .npy files
    print("\n[*] Saving results to .npy files...")
    for split in splits:
        for r in args.resolutions:
            out_file = os.path.join(args.out_dir, f"{split}_seglab_{r}.npy")
            dict_to_save = data[split][r]
            np.save(out_file, dict_to_save)
            print(f"  [✓] Saved {len(dict_to_save):>6} samples -> {out_file}")

    print("\n[✓] All diarization segment labels generated successfully!")


if __name__ == "__main__":
    main()
