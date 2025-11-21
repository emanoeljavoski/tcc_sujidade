#!/usr/bin/env python3
"""Pipeline completo 2 classes (limpo/sujo) para rodar no Dell G15 (ou Mac).

Este script:
1) Monta o dataset binário 2 classes (limpo/sujo) a partir de:
   - datasets públicos (DeepSolarEye, Duke_UAV, Roboflow)
   - seus dados de drone (meus_dados_drone/limpo e /sujo)
2) Salva o dataset já dividido em train/val/test.
3) Dispara o treinamento EfficientNet-B4 2 classes usando CUDA/MPS/CPU.

Uso típico no Dell (exemplo):

    python scripts/pipeline_2classes_dell.py \
        --datasets-publicos-dir E:/datasets_publicos \
        --meus-dados-dir E:/meus_dados_drone \
        --dataset-dir C:/datasets/dataset_2classes_final \
        --output-dir C:/modelos/classificador_2classes \
        --max-limpo 4000 --max-sujo 12000 \
        --epochs 20

Para retomar de um checkpoint copiado do Mac (por exemplo epoch 2):

    python scripts/pipeline_2classes_dell.py \
        --datasets-publicos-dir E:/datasets_publicos \
        --meus-dados-dir E:/meus_dados_drone \
        --dataset-dir C:/datasets/dataset_2classes_final \
        --output-dir C:/modelos/classificador_2classes \
        --max-limpo 4000 --max-sujo 12000 \
        --epochs 20 \
        --resume-from C:/modelos/checkpoint_epoch_2.pth

Observação:
- Este script NÃO cria ambiente Python nem instala dependências; isso deve ser
  feito antes (venv + PyTorch com CUDA no Dell).
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import sys

# Reaproveitar lógica de treino do script local
from treinar_efficientnet_2classes_local import treinar_local  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline 2 classes (dataset + treino EfficientNet-B4).",
    )
    parser.add_argument(
        "--datasets-publicos-dir",
        type=str,
        required=True,
        help="Diretório raiz dos datasets públicos (contendo DeepSolarEye, Duke_UAV, Roboflow)",
    )
    parser.add_argument(
        "--meus-dados-dir",
        type=str,
        required=True,
        help="Diretório raiz dos seus dados de drone (meus_dados_drone)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Diretório onde o dataset 2 classes final (train/val/test) será criado",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Diretório para salvar modelos, checkpoints e relatórios",
    )
    parser.add_argument(
        "--max-limpo",
        type=int,
        default=4000,
        help="Máximo de imagens limpas a usar (0 = sem limite)",
    )
    parser.add_argument(
        "--max-sujo",
        type=int,
        default=12000,
        help="Máximo de imagens sujas a usar (0 = sem limite)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Número máximo de épocas de treino (early stopping pode parar antes)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Opcional: caminho para checkpoint .pth para retomar treino",
    )
    return parser.parse_args()


def coletar_imagens(datasets_pub: Path, meus_dados: Path):
    """Coleta listas de caminhos para imagens limpas e sujas.

    Heurísticas de classificação por nome de arquivo/caminho como no Mac.
    """

    limpo: list[Path] = []
    sujo: list[Path] = []

    # DeepSolarEye
    dse = datasets_pub / "DeepSolarEye"
    if dse.exists():
        for root, _, files in os.walk(dse):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                p = Path(root) / f
                ps = str(p).lower()
                if any(k in ps for k in ["clean", "limpo", "no_dust", "good"]):
                    limpo.append(p)
                else:
                    sujo.append(p)

    # Duke UAV
    duke = datasets_pub / "Duke_UAV"
    if duke.exists():
        for root, _, files in os.walk(duke):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                p = Path(root) / f
                if "clean" in str(p).lower():
                    limpo.append(p)
                else:
                    sujo.append(p)

    # Roboflow
    robo = datasets_pub / "Roboflow"
    if robo.exists():
        for root, _, files in os.walk(robo):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                p = Path(root) / f
                ps = str(p).lower()
                if "clean" in ps or "good" in ps:
                    limpo.append(p)
                else:
                    sujo.append(p)

    # Meus dados de drone
    limpo_root = meus_dados / "limpo"
    sujo_root = meus_dados / "sujo"
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        if limpo_root.exists():
            limpo.extend(list(limpo_root.rglob(ext)))
        if sujo_root.exists():
            sujo.extend(list(sujo_root.rglob(ext)))

    print(f"📊 Coletados (antes do corte): {len(limpo)} limpo, {len(sujo)} sujo")
    return limpo, sujo


def montar_dataset(limpo, sujo, out_dir: Path, max_limpo: int, max_sujo: int):
    """Cria estrutura train/val/test/limpo,sujo com amostragem e split 70/20/10."""

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "val", "test"]:
        for cls in ["limpo", "sujo"]:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    random.shuffle(limpo)
    random.shuffle(sujo)

    if max_limpo > 0:
        limpo = limpo[:max_limpo]
    if max_sujo > 0:
        sujo = sujo[:max_sujo]

    print(f"🎲 Após corte: {len(limpo)} limpo, {len(sujo)} sujo")

    for cls_name, imgs in [("limpo", limpo), ("sujo", sujo)]:
        n_total = len(imgs)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.2)

        for i, img in enumerate(imgs):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            try:
                dest = out_dir / split / cls_name / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
            except Exception:
                # Se alguma imagem estiver corrompida, apenas pula
                continue

    # Estatísticas finais
    stats: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        stats[split] = {}
        for cls in ["limpo", "sujo"]:
            count = len(list((out_dir / split / cls).glob("*")))
            stats[split][cls] = count
            print(f"   {split}/{cls}: {count}")

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        import json

        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dataset 2 classes criado em: {out_dir}")


def main() -> None:
    args = parse_args()

    datasets_pub = Path(args.datasets_publicos_dir).expanduser().resolve()
    meus_dados = Path(args.meus_dados_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    resume_from = Path(args.resume_from).expanduser().resolve() if args.resume_from else None

    print("=" * 80)
    print("🚀 PIPELINE 2 CLASSES (DATASET + TREINO EFFICIENTNET-B4)")
    print("=" * 80)
    print()

    print("[1/2] Preparando dataset binário 2 classes...")
    limpo, sujo = coletar_imagens(datasets_pub, meus_dados)
    montar_dataset(limpo, sujo, dataset_dir, args.max_limpo, args.max_sujo)

    print("\n[2/2] Iniciando treinamento EfficientNet-B4 2 classes...")
    treinar_local(dataset_dir, output_dir, resume_from, args.epochs)

    print("\n" + "=" * 80)
    print("✅ PIPELINE CONCLUÍDO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
