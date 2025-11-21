"""Avaliação do classificador EfficientNet-B4 binário (limpo/sujo).

Carrega o melhor checkpoint da execução no Dell (run_20251114_1620),
roda no conjunto de teste do dataset_final (binário) e salva:

- matriz_confusao_b4_2classes.json
- matriz_confusao_b4_2classes.png

Uso:
    python backend/avaliar_classificador_2classes.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")  # backend não interativo para salvar figuras
import matplotlib.pyplot as plt
import seaborn as sns

from aplicacao.modelos.treinamento_classificador import TreinadorClassificador
from aplicacao.modelos.classificador_sujidade import ClassificadorSujidade


def avaliar_classificador_2classes(
    dataset_dir: Path,
    checkpoint_path: Path,
    saida_dir: Path,
    batch_size: int = 16,
    split: str = "test",
) -> None:
    """Avalia o EfficientNet-B4 binário no conjunto de teste e gera matriz de confusão.

    Args:
        dataset_dir: Diretório raiz do dataset binário com subpastas train/val/test.
        checkpoint_path: Caminho para o checkpoint .pth do modelo treinado.
        saida_dir: Diretório onde salvar JSON e PNG da matriz de confusão.
        batch_size: Tamanho do batch para avaliação.
    """

    dataset_dir = dataset_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    saida_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    # Inicializar treinador apenas para reaproveitar lógica de DataLoader e classes
    treinador = TreinadorClassificador(str(dataset_dir), modelo_base="efficientnet_b4")

    # Precisamos dos loaders e dos pesos de classe
    train_loader, val_loader, test_loader, class_weights = treinador.preparar_datasets(batch_size=batch_size)
    classes: List[str] = list(treinador.classes_detectadas)

    dispositivo = treinador.dispositivo

    # Criar modelo EfficientNet-B4 binário e carregar pesos do checkpoint
    modelo = ClassificadorSujidade(num_classes=len(classes), modelo_base="efficientnet_b4")
    ckpt = torch.load(checkpoint_path, map_location=dispositivo)
    state_dict = ckpt.get("state_dict", ckpt)
    modelo.modelo.load_state_dict(state_dict, strict=False)
    modelo.modelo.to(dispositivo)
    modelo.modelo.eval()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(dispositivo))

    # Escolher split de avaliação
    if split == "val":
        eval_loader = val_loader
        split_name = "val"
    else:
        eval_loader = test_loader
        split_name = "test"

    all_targets: List[int] = []
    all_predictions: List[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for dados, targets in eval_loader:
            dados = dados.to(dispositivo)
            targets = targets.to(dispositivo)

            # Avaliação sempre em precisão padrão para evitar erros Half/Float
            saidas = modelo.modelo(dados)
            loss = criterion(saidas, targets)

            total_loss += loss.item()
            _, predicted = torch.max(saidas.data, 1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(predicted.cpu().numpy().tolist())

    all_targets_np = np.array(all_targets)
    all_predictions_np = np.array(all_predictions)

    # Acurácia simples: proporção de acertos
    accuracy = float(np.mean(all_targets_np == all_predictions_np))
    test_loss = float(total_loss / max(1, len(eval_loader)))

    # Matriz de confusão (ordem das classes conforme ImageFolder)
    cm = confusion_matrix(all_targets_np, all_predictions_np, labels=list(range(len(classes))))
    cm_list = cm.tolist()

    # Normalização por linha (classe real)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    # Evitar divisão por zero
    row_sums[row_sums == 0.0] = 1.0
    cm_norm = (cm_norm / row_sums).tolist()

    # Classification report detalhado
    report = classification_report(
        all_targets_np,
        all_predictions_np,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    # Salvar JSON
    json_path = saida_dir / "matriz_confusao_b4_2classes.json"
    payload = {
        "classes": classes,
        "confusion_matrix": cm_list,
        "confusion_matrix_normalized": cm_norm,
        "accuracy": accuracy,
        "loss": test_loss,
        "split": split_name,
        "classification_report": report,
        "dataset_dir": str(dataset_dir),
        "checkpoint_path": str(checkpoint_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Gerar heatmap normalizado
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        np.array(cm_norm),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - EfficientNet-B4 (2 classes)")
    plt.tight_layout()

    png_path = saida_dir / "matriz_confusao_b4_2classes.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nAvaliação concluída.")
    print(f"   - Split avaliado: {split_name}")
    print(f"   - Acurácia {split_name}: {accuracy:.4f}")
    print(f"   - Loss {split_name}: {test_loss:.4f}")
    print(f"   - JSON salvo em: {json_path}")
    print(f"   - Figura salva em: {png_path}")


def main() -> None:
    # Dataset binário final com estrutura train/val/test/{limpo,sujo}
    # Neste ambiente, o dataset consolidado de 2 classes está em F:\\dataset_2classes_final
    dataset_dir = Path(r"F:\\dataset_2classes_final")

    # Modelo continuado na GPU (experimento final do classificador)
    checkpoint_path = Path(r"f:\\modelos_salvos\\classificador_2classes_continuado_gpu\\checkpoint_epoch_4.pth")

    # Usar o diretório do experimento continuado para salvar matriz de confusão
    saida_dir = checkpoint_path.parent

    # Avaliar no conjunto de teste para obter a matriz de confusão final
    avaliar_classificador_2classes(dataset_dir, checkpoint_path, saida_dir, split="test")


if __name__ == "__main__":
    main()
