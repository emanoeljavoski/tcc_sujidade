#!/usr/bin/env python3
"""
Varre /Volumes/Z Slim/dataset_final/{train,val,test} e move imagens corrompidas para _quarantine.
Não deleta nada.
"""
import os
from pathlib import Path
from PIL import Image
import cv2
import shutil

DATASET_ROOT = Path("/Volumes/Z Slim/dataset_final")
SPLITS = ["train", "val", "test"]
QUARANTINE = DATASET_ROOT / "_quarantine"

ok = 0
bad = 0
moved = []

for split in SPLITS:
    for cls in ["limpo", "sujo"]:
        dirp = DATASET_ROOT / split / cls
        if not dirp.exists():
            continue
        for p in dirp.rglob("*.*"):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            is_bad = False
            try:
                # Tentativa com PIL
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                is_bad = True
            if not is_bad:
                try:
                    # Tentativa com OpenCV
                    img = cv2.imread(str(p))
                    if img is None or img.size == 0:
                        is_bad = True
                except Exception:
                    is_bad = True
            if is_bad:
                bad += 1
                rel = p.relative_to(DATASET_ROOT)
                target = QUARANTINE / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(p), str(target))
                    moved.append(str(rel))
                except Exception as e:
                    print(f"WARN: falha ao mover {p}: {e}")
            else:
                ok += 1

print("=== LIMPEZA CONCLUÍDA ===")
print(f"OK: {ok}")
print(f"Corrompidas movidas: {bad}")
if moved:
    print("Lista (primeiras 20):")
    for m in moved[:20]:
        print(f" - {m}")
