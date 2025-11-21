#!/usr/bin/env python3
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from aplicacao.modelos.detector_modulos import DetectorModulos

import cv2
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("orthomosaic")

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

DJI_RE = re.compile(r"DJI_(\d{8})(\d{6})?_(\d{4})_D\.[jJpP][pPnN][gG]")


def list_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.glob("**/*")) if p.is_file() and p.suffix in SUPPORTED and not p.name.startswith("._")]


def parse_key(p: Path) -> Tuple[str, int]:
    m = DJI_RE.match(p.name)
    if not m:
        # fallback: whole name as date key, index 0
        return (p.name[:8], 0)
    date = m.group(1)  # YYYYMMDD
    idx = int(m.group(3))  # 4-digit index
    return (date, idx)


def cluster_by_index(paths: List[Path], gap_threshold: int = 10) -> Dict[str, List[List[Path]]]:
    by_date: Dict[str, List[Tuple[int, Path]]] = {}
    for p in paths:
        date, idx = parse_key(p)
        by_date.setdefault(date, []).append((idx, p))
    clusters: Dict[str, List[List[Path]]] = {}
    for date, items in by_date.items():
        items.sort(key=lambda x: x[0])
        groups: List[List[Path]] = []
        cur: List[Path] = []
        prev_idx = None
        for idx, p in items:
            if prev_idx is None or (idx - prev_idx) <= gap_threshold:
                cur.append(p)
            else:
                if len(cur) >= 3:
                    groups.append(cur)
                cur = [p]
            prev_idx = idx
        if len(cur) >= 3:
            groups.append(cur)
        if groups:
            clusters[date] = groups
    return clusters


def stitch_images(images: List[Path], work_max_w: int = 1600) -> Image.Image | None:
    pil_imgs = []
    for p in images:
        try:
            im = Image.open(p).convert('RGB')
            # downscale for stitching to save memory
            w, h = im.size
            if w > work_max_w:
                scale = work_max_w / float(w)
                im = im.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            pil_imgs.append(im)
        except Exception:
            continue
    if len(pil_imgs) < 2:
        return None
    # Re-read using cv2 to keep consistent sizes
    mats = []
    for im in pil_imgs:
        mats.append(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    try:
        stitcher = cv2.Stitcher_create() if hasattr(cv2, 'Stitcher_create') else cv2.Stitcher.create()
        status, pano = stitcher.stitch(mats)
        if status != cv2.Stitcher_OK:
            logger.warning(f"Stitch falhou com status={status}")
            return None
        # Convert to PIL
        pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        return Image.fromarray(pano_rgb)
    except Exception as e:
        logger.error(f"Erro no stitch: {e}")
        return None


def save_image(pil_img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(path, quality=95)


def run(root: Path, out_root: Path, conf: float = 0.30, modelo_size: str = 's', work_max_w: int = 1600, fallback_confs: List[float] | None = None):
    lavado = root / 'lavado'
    sujo = root / 'sujo'
    imgs = []
    if lavado.exists():
        imgs += list_images(lavado)
    if sujo.exists():
        imgs += list_images(sujo)
    if not imgs:
        logger.error(f"Sem imagens em {root}")
        return
    clusters = cluster_by_index(imgs, gap_threshold=10)
    logger.info(f"Datas detectadas: {list(clusters.keys())}")

    detector = DetectorModulos(caminho_modelo=None, modelo_size=modelo_size)

    summary_rows = []
    for date, groups in clusters.items():
        for gi, group in enumerate(groups, start=1):
            plant_id = f"{date}_g{gi:02d}"
            logger.info(f"\n🧩 Montando mosaico {plant_id} com {len(group)} imagens")
            pano = stitch_images(group, work_max_w=work_max_w)
            if pano is None:
                logger.warning(f"Mosaico falhou: {plant_id}")
                # ainda assim salvar lista de imagens e seguir para contagem por imagem
                out_dir = out_root / plant_id
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / 'images.txt', 'w') as f:
                    for p in group:
                        f.write(str(p) + "\n")
                # contagem por imagem (fallback)
                per_counts = []
                for img in group:
                    detc = []
                    for ctry in ([conf] + (fallback_confs or [])):
                        detc = detector.detectar(str(img), confianca_min=ctry)
                        if len(detc) > 0:
                            break
                    per_counts.append((str(img), len(detc)))
                import csv
                with open(out_dir / 'per_image_counts.csv', 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['image','count'])
                    for row in per_counts:
                        w.writerow(row)
                per_max = max((c for _, c in per_counts), default=0)
                summary_rows.append((plant_id, len(group), 0, per_max, ''))
                continue
            out_dir = out_root / plant_id
            mosaic_path = out_dir / 'mosaic.jpg'
            save_image(pano, mosaic_path)

            # Salvar e contar módulos no mosaico
            tmp_path = out_dir / 'mosaic_tmp.jpg'
            save_image(pano, tmp_path)
            confs = [conf]
            if fallback_confs:
                confs.extend([c for c in fallback_confs if c is not None])
            dets = []
            for ctry in confs:
                dets = detector.detectar(str(tmp_path), confianca_min=ctry)
                if len(dets) > 0:
                    break
            count = len(dets)
            logger.info(f"{plant_id}: módulos detectados no mosaico = {count}")

            # Desenhar boxes
            ann_path = out_dir / 'mosaic_annotated.jpg'
            try:
                detector.desenhar_deteccoes(str(tmp_path), dets, salvar_path=str(ann_path))
            except Exception:
                pass
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            # Registrar
            with open(out_dir / 'images.txt', 'w') as f:
                for p in group:
                    f.write(str(p) + "\n")
            with open(out_dir / 'count.txt', 'w') as f:
                f.write(str(count))
            # se count==0, computar per-image counts como fallback
            per_max = 0
            if count == 0:
                per_counts = []
                for img in group:
                    detc = []
                    for ctry in ([conf] + (fallback_confs or [])):
                        detc = detector.detectar(str(img), confianca_min=ctry)
                        if len(detc) > 0:
                            break
                    per_counts.append((str(img), len(detc)))
                import csv
                with open(out_dir / 'per_image_counts.csv', 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['image','count'])
                    for row in per_counts:
                        w.writerow(row)
                per_max = max((c for _, c in per_counts), default=0)
            summary_rows.append((plant_id, len(group), count, per_max, str(mosaic_path)))

    # Salvar resumo
    import csv
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / 'summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['plant_id', 'num_images', 'mosaic_count', 'per_image_max', 'mosaic_path'])
        for r in summary_rows:
            w.writerow(r)
    logger.info(f"✅ Ortomosaicos concluídos. Resumo: {out_root/'summary.csv'}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-root', default=str(Path('dados')/ 'plantas_completas' / 'imagens'))
    ap.add_argument('--out', default='/Volumes/Z Slim/ortomosaicos')
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--model-size', choices=['n','s'], default='s')
    ap.add_argument('--max-width', type=int, default=1600)
    ap.add_argument('--fallback-conf1', type=float, default=0.20)
    ap.add_argument('--fallback-conf2', type=float, default=0.15)
    args = ap.parse_args()

    fallbacks = []
    if args.fallback_conf1 is not None:
        fallbacks.append(args.fallback_conf1)
    if args.fallback_conf2 is not None:
        fallbacks.append(args.fallback_conf2)
    run(Path(args.images_root), Path(args.out), conf=args.conf, modelo_size=args.model_size, work_max_w=args.max_width, fallback_confs=fallbacks)
