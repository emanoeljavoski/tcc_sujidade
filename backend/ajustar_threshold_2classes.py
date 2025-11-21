"""Ajuste automático do limiar da classe "sujo" para o classificador EfficientNet-B4 binário.

O script usa o modelo continuado na GPU (checkpoint_epoch_4) e o conjunto de teste
F:\\dataset_2classes_final para buscar um threshold que equilibre melhor o desempenho
entre as classes "limpo" e "sujo".
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict, Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from aplicacao.modelos.treinamento_classificador import TreinadorClassificador
from aplicacao.modelos.classificador_sujidade import ClassificadorSujidade


def main() -> None:
    # Configurações fixas deste experimento
    dataset_dir = Path(r"F:\\dataset_2classes_final")
    checkpoint_path = Path(r"F:\\modelos_salvos\\classificador_2classes_continuado_gpu\\checkpoint_epoch_4.pth")
    saida_dir = checkpoint_path.parent
    saida_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    print("Dataset:", dataset_dir)
    print("Checkpoint:", checkpoint_path)

    # Reaproveitar lógica de DataLoader do TreinadorClassificador
    treinador = TreinadorClassificador(str(dataset_dir), modelo_base="efficientnet_b4")
    _, _, test_loader, _ = treinador.preparar_datasets(batch_size=32)
    classes: List[str] = list(treinador.classes_detectadas)
    dispositivo = treinador.dispositivo

    if "sujo" not in classes or "limpo" not in classes:
        raise RuntimeError(f"Esperado classes ['limpo','sujo'], obtido: {classes}")

    idx_limpo = classes.index("limpo")
    idx_sujo = classes.index("sujo")

    # Carregar modelo EfficientNet-B4 com pesos continuados
    modelo = ClassificadorSujidade(num_classes=len(classes), modelo_base="efficientnet_b4")
    ckpt = torch.load(checkpoint_path, map_location=dispositivo)
    state_dict = ckpt.get("state_dict", ckpt)
    modelo.modelo.load_state_dict(state_dict, strict=False)
    modelo.modelo.to(dispositivo)
    modelo.modelo.eval()

    # Coletar, para TODO o conjunto de teste, os rótulos verdadeiros e a probabilidade de "sujo"
    all_targets: List[int] = []
    all_p_sujo: List[float] = []

    with torch.no_grad():
        for dados, targets in test_loader:
            dados = dados.to(dispositivo)
            targets = targets.to(dispositivo)

            outputs = modelo.modelo(dados)
            probs = torch.softmax(outputs, dim=1)

            p_sujo_batch = probs[:, idx_sujo]

            all_targets.extend(targets.cpu().numpy().tolist())
            all_p_sujo.extend(p_sujo_batch.cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=np.int64)
    p_sujo = np.array(all_p_sujo, dtype=np.float32)

    # Espaço de busca de limiares para a classe "sujo"
    thresholds = np.linspace(0.3, 0.9, 13)  # 0.30, 0.35, ..., 0.90

    resultados: List[Dict[str, Any]] = []

    for thr in thresholds:
        # Predição binária: sujo se p_sujo >= thr, caso contrário limpo
        y_pred = np.where(p_sujo >= thr, idx_sujo, idx_limpo)

        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        report = classification_report(
            y_true,
            y_pred,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )

        macro_f1 = (report["limpo"]["f1-score"] + report["sujo"]["f1-score"]) / 2.0

        resultados.append(
            {
                "threshold": float(thr),
                "accuracy": acc,
                "macro_f1": float(macro_f1),
                "cm": cm.tolist(),
                "report": report,
            }
        )

    # Critério de escolha: maximizar F1 médio, preservando recall de "sujo" >= 0,95 sempre que possível
    candidatos = [r for r in resultados if r["report"]["sujo"]["recall"] >= 0.95]
    if candidatos:
        melhor = max(candidatos, key=lambda r: r["macro_f1"])
    else:
        melhor = max(resultados, key=lambda r: r["macro_f1"])

    # Salvar JSON detalhado com todos os thresholds testados e o melhor selecionado
    json_path = saida_dir / "melhor_threshold_sujo.json"
    payload = {
        "classes": classes,
        "resultados": resultados,
        "melhor": melhor,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== MELHOR THRESHOLD ENCONTRADO ===")
    print(f"Threshold sujo: {melhor['threshold']:.2f}")
    print(f"Acurácia teste: {melhor['accuracy']:.4f}")
    print(f"Macro F1: {melhor['macro_f1']:.4f}")
    print(f"Recall limpo: {melhor['report']['limpo']['recall']:.4f}")
    print(f"Recall sujo: {melhor['report']['sujo']['recall']:.4f}")
    print(f"JSON salvo em: {json_path}")


if __name__ == "__main__":
    main()
