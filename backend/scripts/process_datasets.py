"""
Processa e integra datasets públicos + dataset local em um dataset final de classificação binária (limpo/sujo).

Funcionalidades:
- DeepSolarEye: filtra imagens com soiling >= --min-soiling e rotula como 'sujo'
- Roboflow Solar Panel Dust Detection (PyTorch export): considera imagens de 'train' como 'sujo'
- (Opcional) Duke UAV: ignorado na classificação de sujidade por não conter rótulos de sujeira
- Integra com dataset local (padrão: dados/modulos_individuais/limpo, sujo)
- Redimensiona tudo para --img-size (padrão 640)
- Data augmentation nos dados públicos (rotação ±30°, flip, brilho/contraste, crop)
- Split: dados locais (70/15/15); dados públicos apenas no train
- Balanceamento no train (50/50) via downsample/oversample
- Gera dataset_final com estrutura: train/val/test/{limpo,sujo} e CSVs (dataset_statistics.csv, images.csv)

Uso:
    python scripts/process_datasets.py \
        --public-root /home/claude/datasets_publicos \
        --local-root dados/modulos_individuais \
        --final-root /home/claude/dataset_final \
        --min-soiling 20 \
        --img-size 640 \
        --augment-multiplier 1 \
        --balance downsample
"""
from __future__ import annotations
import argparse
import csv
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Augmentations
import albumentations as A

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("process_datasets")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
CLASSES_BIN = ["limpo", "sujo"]


@dataclass
class Sample:
    path: Path
    label: str  # 'limpo' or 'sujo'
    source: str  # 'local' | 'deepsolareye' | 'roboflow'


def is_image(p: Path) -> bool:
    # Ignorar arquivos AppleDouble/metadata e similares
    if p.name.startswith("._"):
        return False
    try:
        if not p.exists():
            return False
        if not p.is_file():
            return False
    except OSError:
        # Ex.: Stale NFS file handle
        return False
    return p.suffix.lower() in SUPPORTED_EXTS


def load_local_dataset(local_root: Path) -> Dict[str, List[Sample]]:
    """Carrega dataset local em dados/modulos_individuais/{limpo,sujo}."""
    samples_by_class: Dict[str, List[Sample]] = {"limpo": [], "sujo": []}
    for c in CLASSES_BIN:
        d = local_root / c
        if not d.exists():
            logger.warning(f"Classe local ausente: {d}")
            continue
        for p in d.rglob('*'):
            if p.is_file() and is_image(p):
                samples_by_class[c].append(Sample(p, c, "local"))
    logger.info(f"Local: limpo={len(samples_by_class['limpo'])} sujo={len(samples_by_class['sujo'])}")
    return samples_by_class


def find_first_csv(root: Path) -> Optional[Path]:
    for p in root.rglob('*.csv'):
        return p
    return None


def load_deepsolareye(public_root: Path, min_soiling: float) -> List[Sample]:
    """Tenta carregar amostras do DeepSolarEye com soiling >= min_soiling como 'sujo'."""
    base = public_root / "DeepSolarEye"
    if not base.exists():
        logger.info("DeepSolarEye não encontrado, pulando.")
        return []

    # Encontrar CSV com rótulos de sujeira
    csv_path = find_first_csv(base)
    if not csv_path:
        logger.warning("CSV de rótulos não encontrado no DeepSolarEye. Tentando fallback por nome de arquivo (L_<loss>).")
        # Fallback: parsear nomes de arquivo com padrão ..._L_<float>_I_<float>.jpg
        import re
        pat = re.compile(r"_L_([0-9]*\.?[0-9]+)")
        samples: List[Sample] = []
        for p in base.rglob('*'):
            if p.is_file() and is_image(p):
                m = pat.search(p.name)
                if not m:
                    continue
                try:
                    loss = float(m.group(1))
                except Exception:
                    continue
                # Converter 0..1 para 0..100
                if 0.0 <= loss <= 1.0:
                    loss *= 100.0
                if loss >= min_soiling:
                    samples.append(Sample(p, 'sujo', 'deepsolareye'))
        logger.info(f"DeepSolarEye (fallback filename): {len(samples)} amostras 'sujo' (>= {min_soiling}%)")
        return samples

    logger.info(f"Lendo CSV do DeepSolarEye: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Erro lendo CSV DeepSolarEye: {e}")
        return []

    # Detectar colunas
    cols = {c.lower(): c for c in df.columns}
    # candidatos de coluna de imagem
    img_cols = [c for c in cols if c in {"filename", "image", "img", "path"}]
    if not img_cols:
        logger.warning("Coluna de caminho de imagem não detectada no CSV DeepSolarEye. Pulando.")
        return []
    img_col = cols[img_cols[0]]

    # candidatos de coluna de sujeira
    soiling_cols = [c for c in cols if any(k in c for k in ["soil", "soiling", "dust", "loss", "power"]) ]
    if not soiling_cols:
        logger.warning("Coluna de soiling/perda não detectada no CSV DeepSolarEye. Pulando.")
        return []
    # Selecionar a primeira coluna numérica
    soiling_col = None
    for c in soiling_cols:
        try:
            pd.to_numeric(df[cols[c]].head(5))  # teste
            soiling_col = cols[c]
            break
        except Exception:
            continue
    if soiling_col is None:
        logger.warning("Nenhuma coluna numérica de soiling encontrada no CSV. Pulando.")
        return []

    samples: List[Sample] = []
    base_dir = base

    for _, row in df.iterrows():
        img_rel = str(row[img_col])
        val = row[soiling_col]
        try:
            valf = float(val)
        except Exception:
            continue
        # Converter valores de 0..1 para 0..100 se necessário
        if 0.0 <= valf <= 1.0:
            valf *= 100.0
        if valf < min_soiling:
            continue
        # Encontrar caminho da imagem
        candidate = base_dir / img_rel
        if not candidate.exists():
            # tentar apenas o nome do arquivo
            candidate = next(base_dir.rglob(Path(img_rel).name), None)
            if not candidate:
                continue
        if candidate.is_file() and is_image(candidate):
            samples.append(Sample(candidate, "sujo", "deepsolareye"))
    logger.info(f"DeepSolarEye: {len(samples)} amostras 'sujo' (>= {min_soiling}%)")
    return samples


def load_deepsolareye_clean(public_root: Path, max_soiling: float) -> List[Sample]:
    """Carrega amostras do DeepSolarEye com soiling < max_soiling como 'limpo'."""
    base = public_root / "DeepSolarEye"
    if not base.exists():
        return []

    csv_path = find_first_csv(base)
    if not csv_path:
        logger.warning("CSV de rótulos não encontrado no DeepSolarEye para 'limpo'. Tentando fallback por nome de arquivo (L_<loss>).")
        import re
        pat = re.compile(r"_L_([0-9]*\.?[0-9]+)")
        samples: List[Sample] = []
        for p in base.rglob('*'):
            if p.is_file() and is_image(p):
                m = pat.search(p.name)
                if not m:
                    continue
                try:
                    loss = float(m.group(1))
                except Exception:
                    continue
                if 0.0 <= loss <= 1.0:
                    loss *= 100.0
                if loss < max_soiling:
                    samples.append(Sample(p, 'limpo', 'deepsolareye'))
        logger.info(f"DeepSolarEye (fallback filename): {len(samples)} amostras 'limpo' (< {max_soiling}%)")
        return samples

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Erro lendo CSV DeepSolarEye (limpo): {e}")
        return []

    cols = {c.lower(): c for c in df.columns}
    img_cols = [c for c in cols if c in {"filename", "image", "img", "path"}]
    if not img_cols:
        logger.warning("Coluna de imagem não detectada no CSV DeepSolarEye para 'limpo'. Pulando.")
        return []
    img_col = cols[img_cols[0]]

    soiling_cols = [c for c in cols if any(k in c for k in ["soil", "soiling", "dust", "loss", "power"])]
    if not soiling_cols:
        logger.warning("Coluna de soiling/perda não detectada no CSV para 'limpo'. Pulando.")
        return []
    soiling_col = None
    for c in soiling_cols:
        try:
            pd.to_numeric(df[cols[c]].head(5))
            soiling_col = cols[c]
            break
        except Exception:
            continue
    if soiling_col is None:
        return []

    samples: List[Sample] = []
    base_dir = base
    for _, row in df.iterrows():
        img_rel = str(row[img_col])
        val = row[soiling_col]
        try:
            valf = float(val)
        except Exception:
            continue
        if 0.0 <= valf <= 1.0:
            valf *= 100.0
        if valf >= max_soiling:
            continue
        candidate = base_dir / img_rel
        if not candidate.exists():
            cand2 = None
            try:
                cand2 = next(base_dir.rglob(Path(img_rel).name))
            except StopIteration:
                cand2 = None
            candidate = cand2 if cand2 else candidate
        if candidate.exists() and candidate.is_file() and is_image(candidate):
            samples.append(Sample(candidate, "limpo", "deepsolareye"))
    logger.info(f"DeepSolarEye: {len(samples)} amostras 'limpo' (< {max_soiling}%)")
    return samples


def load_roboflow_pytorch(public_root: Path) -> List[Sample]:
    base = public_root / "Roboflow"
    if not base.exists():
        logger.info("Roboflow não encontrado, pulando.")
        return []
    # Estrutura típica: Roboflow/<proj>/ <version>/ {train,valid,test}/
    # Considerar apenas train/* como 'sujo'
    samples: List[Sample] = []
    for p in base.rglob('train'):
        if p.is_dir():
            img_dir = p / 'images'
            if not img_dir.exists():
                # algumas exports não usam subpasta images
                img_dir = p
            for img in img_dir.rglob('*'):
                if img.is_file() and is_image(img):
                    samples.append(Sample(img, 'sujo', 'roboflow'))
    logger.info(f"Roboflow: {len(samples)} amostras 'sujo' (train)")
    return samples


def load_duke_uav_clean(public_root: Path, limit: Optional[int] = None) -> List[Sample]:
    """Carrega imagens do Duke UAV como classe 'limpo' (sem rótulo de sujeira)."""
    base = public_root / "Duke_UAV"
    if not base.exists():
        logger.info("Duke UAV não encontrado, pulando.")
        return []
    samples: List[Sample] = []
    count = 0
    for img in base.rglob('*'):
        if img.is_file() and is_image(img):
            samples.append(Sample(img, 'limpo', 'duke_uav'))
            count += 1
            if limit is not None and count >= limit:
                break
    logger.info(f"Duke UAV: {len(samples)} amostras 'limpo' (não rotuladas)")
    return samples


def extract_duke_uav_frames(public_root: Path, every_n_seconds: float = 2.0, limit_frames: Optional[int] = 2000) -> List[Sample]:
    """Extrai frames dos vídeos de Duke UAV como 'limpo'."""
    base = public_root / "Duke_UAV" / "video"
    out_dir = public_root / "Duke_UAV" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        logger.info("Duke UAV/videos não encontrado, pulando extração de frames.")
        return []
    samples: List[Sample] = []
    total = 0
    vids = sorted([p for p in base.iterdir() if p.suffix.lower() in {'.mp4', '.mov', '.m4v', '.avi', '.mpg', '.mpeg'}])
    for v in vids:
        try:
            cap = cv2.VideoCapture(str(v))
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(fps * every_n_seconds))
            frame_idx = 0
            saved_this = 0
            while True:
                ret = cap.grab()
                if not ret:
                    break
                if frame_idx % step == 0:
                    ret2, frame = cap.retrieve()
                    if not ret2 or frame is None:
                        frame_idx += 1
                        continue
                    name = f"{v.stem}_f{frame_idx:07d}.jpg"
                    out_path = out_dir / name
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    samples.append(Sample(out_path, 'limpo', 'duke_uav_frame'))
                    total += 1
                    saved_this += 1
                    if limit_frames is not None and total >= limit_frames:
                        break
                frame_idx += 1
            cap.release()
            if limit_frames is not None and total >= limit_frames:
                break
        except Exception:
            continue
    logger.info(f"Duke UAV frames extraídos: {len(samples)} 'limpo' (every {every_n_seconds}s)")
    return samples


def build_aug_pipeline(img_size: int) -> A.Compose:
    return A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_REFLECT101, p=0.9),
            A.Affine(scale=(0.9, 1.1), rotate=(-30, 30), shear=(-5, 5), p=0.6),
        ], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
    ])


def load_and_resize(path: Path, img_size: int) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def save_rgb_image(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def split_local(samples_by_class: Dict[str, List[Sample]], seed: int = 42) -> Dict[str, Dict[str, List[Sample]]]:
    random.seed(seed)
    splits = {s: {c: [] for c in CLASSES_BIN} for s in ["train", "val", "test"]}
    for c, items in samples_by_class.items():
        items = items.copy()
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        splits['train'][c] = items[:n_train]
        splits['val'][c] = items[n_train:n_train+n_val]
        splits['test'][c] = items[n_train+n_val:]
    return splits


def balance_train_list(train_list: List[Sample], mode: str = 'downsample', seed: int = 42) -> List[Sample]:
    random.seed(seed)
    limpo = [s for s in train_list if s.label == 'limpo']
    sujo = [s for s in train_list if s.label == 'sujo']
    if not limpo or not sujo:
        return train_list
    if mode == 'downsample':
        n = min(len(limpo), len(sujo))
        limpo = random.sample(limpo, n)
        sujo = random.sample(sujo, n)
        return limpo + sujo
    elif mode == 'oversample':
        if len(limpo) < len(sujo):
            limpo += random.choices(limpo, k=len(sujo) - len(limpo))
        else:
            sujo += random.choices(sujo, k=len(limpo) - len(sujo))
        return limpo + sujo
    return train_list


def main():
    ap = argparse.ArgumentParser(description="Processar e integrar datasets de sujidade (binário)")
    ap.add_argument('--public-root', type=str, default=str(Path('dados') / 'datasets_publicos'))
    ap.add_argument('--local-root', type=str, default=str(Path('dados') / 'modulos_individuais'))
    ap.add_argument('--final-root', type=str, default=str(Path('dados') / 'dataset_final'))
    ap.add_argument('--min-soiling', type=float, default=20.0)
    ap.add_argument('--img-size', type=int, default=640)
    ap.add_argument('--augment-multiplier', type=int, default=1)
    ap.add_argument('--balance', type=str, choices=['downsample', 'oversample', 'none'], default='downsample')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--duke-frames-every', type=float, default=2.0)
    ap.add_argument('--duke-frames-limit', type=int, default=2000)
    args = ap.parse_args()

    public_root = Path(args.public_root)
    local_root = Path(args.local_root)
    final_root = Path(args.final_root)

    final_train = final_root / 'train'
    final_val = final_root / 'val'
    final_test = final_root / 'test'

    logger.info("=" * 80)
    logger.info("🚀 PROCESSAMENTO DE DATASETS (binário limpo/sujo)")
    logger.info("=" * 80)

    # 1) Carregar datasets
    local_by_class = load_local_dataset(local_root)
    dse_sujo = load_deepsolareye(public_root, args.min_soiling)
    dse_limpo = load_deepsolareye_clean(public_root, args.min_soiling)
    rf_sujo = load_roboflow_pytorch(public_root)
    duke_limpo = load_duke_uav_clean(public_root)
    if len(duke_limpo) == 0:
        duke_limpo = extract_duke_uav_frames(public_root, every_n_seconds=args.duke_frames_every, limit_frames=args.duke_frames_limit)

    # 2) Combinar TODOS os dados (locais + públicos) antes do split
    all_samples_by_class = {'limpo': [], 'sujo': []}
    
    # Adicionar dados locais
    for c in CLASSES_BIN:
        if c in local_by_class:
            all_samples_by_class[c] += local_by_class[c]
    
    # Adicionar dados públicos
    all_samples_by_class['sujo'] += dse_sujo
    all_samples_by_class['sujo'] += rf_sujo
    all_samples_by_class['limpo'] += dse_limpo
    all_samples_by_class['limpo'] += duke_limpo
    
    logger.info(f"Total de amostras antes do split:")
    logger.info(f"   Limpo: {len(all_samples_by_class['limpo'])}")
    logger.info(f"   Sujo: {len(all_samples_by_class['sujo'])}")
    
    # 3) Fazer split 70/15/15 de TODOS os dados
    random.seed(args.seed)
    splits = {s: {c: [] for c in CLASSES_BIN} for s in ["train", "val", "test"]}
    for c, items in all_samples_by_class.items():
        items = items.copy()
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        splits['train'][c] = items[:n_train]
        splits['val'][c] = items[n_train:n_train+n_val]
        splits['test'][c] = items[n_train+n_val:]
    
    # 4) Montar listas alvo
    train_list: List[Sample] = []
    val_list: List[Sample] = []
    test_list: List[Sample] = []

    for c in CLASSES_BIN:
        train_list += splits['train'][c]
        val_list += splits['val'][c]
        test_list += splits['test'][c]
    
    logger.info(f"Splits antes do balanceamento:")
    logger.info(f"   Train: limpo={len(splits['train']['limpo'])}, sujo={len(splits['train']['sujo'])}")
    logger.info(f"   Val: limpo={len(splits['val']['limpo'])}, sujo={len(splits['val']['sujo'])}")
    logger.info(f"   Test: limpo={len(splits['test']['limpo'])}, sujo={len(splits['test']['sujo'])}")

    # 5) Balancear apenas o conjunto de treino
    if args.balance != 'none':
        train_list = balance_train_list(train_list, mode=args.balance, seed=args.seed)

    # 6) Augment nos públicos (apenas train)
    aug_pipeline = build_aug_pipeline(args.img_size)
    augmented_additions: List[Tuple[np.ndarray, str, str]] = []  # (img_arr, label, source)
    if args.augment_multiplier > 0:
        for s in tqdm(train_list, desc='Augment (train)'):
            if s.source in {"deepsolareye", "roboflow"}:  # apenas públicos
                arr = load_and_resize(s.path, args.img_size)
                if arr is None:
                    continue
                for _ in range(args.augment_multiplier):
                    aug = aug_pipeline(image=arr)
                    augmented_additions.append((aug['image'], s.label, s.source))

    # 7) Copiar/Salvar imagens para dataset_final com resize
    stats = []  # para dataset_statistics.csv
    rows = []   # para images.csv

    def save_list(lst: List[Sample], split: str):
        for s in tqdm(lst, desc=f'Salvando {split}'):
            arr = load_and_resize(s.path, args.img_size)
            if arr is None:
                continue
            out_dir = (final_train if split == 'train' else final_val if split == 'val' else final_test) / s.label
            out_path = out_dir / f"{s.path.stem}_{hash(str(s.path)) & 0xffffffff:08x}.jpg"
            save_rgb_image(arr, out_path)
            rows.append({
                'split': split,
                'filename': str(out_path),
                'label': s.label,
                'source': s.source,
                'original_path': str(s.path)
            })

    save_list(train_list, 'train')
    save_list(val_list, 'val')
    save_list(test_list, 'test')

    # Salvar augmentations adicionais no train
    for arr, label, source in tqdm(augmented_additions, desc='Salvando augmentações'):
        out_dir = final_train / label
        out_name = f"aug_{source}_{random.randint(0, 1_000_000):06d}.jpg"
        out_path = out_dir / out_name
        save_rgb_image(arr, out_path)
        rows.append({
            'split': 'train',
            'filename': str(out_path),
            'label': label,
            'source': f'{source}_aug',
            'original_path': ''
        })

    # 8) Estatísticas
    # contagem por split/classe
    counts: Dict[Tuple[str, str], int] = {}
    for r in rows:
        key = (r['split'], r['label'])
        counts[key] = counts.get(key, 0) + 1

    stats_rows = []
    for split in ['train', 'val', 'test']:
        for c in CLASSES_BIN:
            stats_rows.append({
                'split': split,
                'class': c,
                'count': counts.get((split, c), 0)
            })

    final_root.mkdir(parents=True, exist_ok=True)
    stats_csv = final_root / 'dataset_statistics.csv'
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'class', 'count'])
        writer.writeheader()
        writer.writerows(stats_rows)

    images_csv = final_root / 'images.csv'
    with open(images_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'filename', 'label', 'source', 'original_path'])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("=" * 80)
    logger.info(f"✅ Dataset final criado em: {final_root}")
    logger.info(f"📄 Estatísticas: {stats_csv}")
    logger.info(f"📄 Índice de imagens: {images_csv}")


if __name__ == '__main__':
    main()
