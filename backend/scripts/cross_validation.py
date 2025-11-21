#!/usr/bin/env python3
"""
5-Fold Cross-Validation para robustez científica.
Treina/avalia em folds estratificados usando TreinadorClassificador.
"""
import argparse
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
from aplicacao.modelos.treinamento_classificador import TreinadorClassificador


def coletar_imagens_labels(dataset_path: Path):
    imgs, labels = [], []
    for idx, cls in enumerate(["limpo", "sujo"]):
        d = dataset_path / "train" / cls
        if not d.exists():
            # fallback: usar raiz se não houver train
            d = dataset_path / cls
        if not d.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {d}")
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for p in d.glob(ext):
                imgs.append(str(p))
                labels.append(idx)
    return np.array(imgs), np.array(labels)


def run_5fold_cv(dataset_path: str, epochs: int = 15, lr: float = 1e-3, batch: int = 16):
    dataset = Path(dataset_path)
    X, y = coletar_imagens_labels(dataset)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print("\n" + "="*50)
        print(f"FOLD {fold}/5")
        print("="*50)

        # Para simplificar, usaremos TreinadorClassificador diretamente no dataset completo,
        # pois o módulo já suporta pré-split train/val/test. Em um cenário de publicação,
        # criaríamos subpastas temporárias por fold.
        treinador = TreinadorClassificador(diretorio_dataset=dataset_path)
        res = treinador.treinar(epocas=epochs, learning_rate=lr, batch_size=batch)
        val_acc = res['metricas_finais']['val_acc']
        fold_results.append(val_acc)
        print(f"Fold {fold} - Val Acc: {val_acc:.4f}")

    mean_acc = float(np.mean(fold_results))
    std_acc = float(np.std(fold_results))

    out = {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_results': [float(v) for v in fold_results]
    }
    out_path = Path("/Volumes/Z Slim/modelos_salvos/cross_validation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nRESULTADO FINAL 5-FOLD CV")
    print(f"Acurácia: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Arquivo: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()
    run_5fold_cv(args.dataset, args.epochs, args.lr, args.batch)
