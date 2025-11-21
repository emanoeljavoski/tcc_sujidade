#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple
import random
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))
from aplicacao.modelos.treinamento_classificador import TreinadorClassificador

CLASSES = ['limpo', 'sujo']


def read_pool(images_csv: Path, use_splits=('train','val')) -> List[Tuple[str,int]]:
    pool = []
    with open(images_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row['split'] in use_splits:
                label = 0 if row['label'] == 'limpo' else 1
                pool.append((row['filename'], label))
    return pool

def read_pool_from_modulos(local_root: Path) -> List[Tuple[str,int]]:
    pairs: List[Tuple[str,int]] = []
    for cls, lbl in (('limpo',0), ('sujo',1)):
        d = local_root / cls
        if not d.exists():
            continue
        for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.JPEG','*.PNG'):
            for p in d.glob(ext):
                pairs.append((str(p), lbl))
    return pairs


def make_fold_dirs(base_dir: Path, fold_idx: int) -> Tuple[Path, Path, Path]:
    fold_root = base_dir / f"cv_fold_{fold_idx}"
    train_root = fold_root / 'train'
    val_root = fold_root / 'val'
    test_root = fold_root / 'test'
    for root in (train_root, val_root, test_root):
        for c in CLASSES:
            (root / c).mkdir(parents=True, exist_ok=True)
    return train_root, val_root, test_root


def populate_split(pairs: List[Tuple[str,int]], out_root: Path):
    for img_path, label in pairs:
        cls = CLASSES[label]
        dst = out_root / cls / f"{Path(img_path).stem}_{abs(hash(img_path)) & 0xffffffff:08x}.jpg"
        try:
            # symlink if same filesystem, else copy
            os.link(img_path, dst)
        except OSError:
            shutil.copy2(img_path, dst)


def kfold_split(pool: List[Tuple[str,int]], k: int, seed: int = 42):
    random.seed(seed)
    # Stratify by label
    by_label = {0: [], 1: []}
    for item in pool:
        by_label[item[1]].append(item)
    for lbl in by_label:
        random.shuffle(by_label[lbl])
    folds = [([], []) for _ in range(k)]  # (train, val)
    for lbl in (0,1):
        items = by_label[lbl]
        fold_size = max(1, len(items) // k)
        parts = [items[i*fold_size:(i+1)*fold_size] for i in range(k-1)]
        parts.append(items[(k-1)*fold_size:])
        for i in range(k):
            val_part = parts[i]
            train_part = [x for j, p in enumerate(parts) if j != i for x in p]
            folds[i][0].extend(train_part)
            folds[i][1].extend(val_part)
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True, help='Raiz do dataset_final_misto (com images.csv)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out', default='/Volumes/Z Slim/modelos_salvos/cv_resnet50')
    args = ap.parse_args()

    root = Path(args.dataset_root)
    images_csv = root / 'images.csv'
    if images_csv.exists():
        pool = read_pool(images_csv, use_splits=('train','val'))
    elif (root / 'limpo').exists() and (root / 'sujo').exists():
        pool = read_pool_from_modulos(root)
    elif (root / 'train').exists() and (root / 'val').exists():
        # montar pool a partir de train+val pré-separados
        pool = []
        for split in ('train','val'):
            for cls, lbl in (('limpo',0), ('sujo',1)):
                d = root / split / cls
                if not d.exists():
                    continue
                for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.JPEG','*.PNG'):
                    for p in d.glob(ext):
                        pool.append((str(p), lbl))
    else:
        raise SystemExit(f"Não foi possível localizar images.csv nem pastas limpo/sujo em {root}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = kfold_split(pool, k=args.folds, seed=42)
    fold_results = []

    tmp_base = out_dir / '_tmp'
    if tmp_base.exists():
        shutil.rmtree(tmp_base)
    tmp_base.mkdir(parents=True, exist_ok=True)

    for i, (train_pairs, val_pairs) in enumerate(folds, start=1):
        print(f"\n=== FOLD {i}/{args.folds} ===")
        train_root, val_root, test_root = make_fold_dirs(tmp_base, i)
        populate_split(train_pairs, train_root)
        populate_split(val_pairs, val_root)
        # Para evitar erros no Treinador (exige test não vazio), duplicar val em test
        populate_split(val_pairs, test_root)

        # Treinar neste fold (sem test; avaliação no val)
        treinador = TreinadorClassificador(str(tmp_base / f"cv_fold_{i}"), modelo_base='resnet50')
        resultado = treinador.treinar(
            epocas=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch,
            diretorio_saida=str(out_dir / f"fold_{i}"),
            use_focal=True,
            unfreeze_epoch=2,
            unfreeze_lr_factor=0.1
        )
        val_acc = resultado['metricas_finais']['val_acc']
        print(f"Fold {i} val_acc={val_acc:.4f}")
        fold_results.append(val_acc)

    import numpy as np
    mean_acc = float(np.mean(fold_results))
    std_acc = float(np.std(fold_results))
    with open(out_dir / 'cv_summary.txt','w') as f:
        f.write(f"folds={args.folds}\n")
        f.write("val_accs=" + ",".join(f"{x:.4f}" for x in fold_results) + "\n")
        f.write(f"mean={mean_acc:.4f}\nstd={std_acc:.4f}\n")
    print(f"\nCV concluído: mean={mean_acc:.4f} std={std_acc:.4f}")

if __name__ == '__main__':
    main()
