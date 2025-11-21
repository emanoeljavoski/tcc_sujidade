#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from aplicacao.tta_inference import TTAPredictor
from aplicacao.modelos.classificador_sujidade import ClassificadorSujidade
from aplicacao.modelos.classificador_resnet import ClassificadorResNet


def load_model(ckpt_path: str, num_classes: int = 2):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    arch = ckpt.get('arquitetura', 'resnet50')
    if arch == 'resnet50':
        clf = ClassificadorResNet(num_classes=num_classes)
    else:
        clf = ClassificadorSujidade(num_classes=num_classes, modelo_base=arch)
    clf.modelo.load_state_dict(ckpt['state_dict'], strict=False)
    clf.modelo.eval()
    return clf


def iter_test_images(test_root: Path):
    classes = ['limpo', 'sujo']
    paths = []
    labels = []
    for idx, c in enumerate(classes):
        d = test_root / c
        for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.webp'):
            for p in d.glob(ext):
                paths.append(p)
                labels.append(idx)
    return paths, np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelo', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--tta', type=int, default=8)
    args = ap.parse_args()

    test_dir = Path(args.dataset) / 'test'
    model = load_model(args.modelo, num_classes=2).modelo
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    tta = TTAPredictor(model, num_augmentations=args.tta, img_size=224)

    paths, y_true = iter_test_images(test_dir)
    correct = 0
    for p, yt in zip(paths, y_true):
        pred, conf, _ = tta.predict_class(Image.open(p).convert('RGB'))
        correct += int(pred == yt)
    acc = correct / len(paths) if paths else 0.0
    print(f"TTA Acc (@{args.tta} aug): {acc*100:.2f}% | N={len(paths)}")

if __name__ == '__main__':
    main()
