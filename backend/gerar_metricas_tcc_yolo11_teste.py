from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from ultralytics import YOLO


DATASET_YAML = Path("F:/dataset_yolo_completo/dataset.yaml")
PROJECT_DIR = Path("F:/modelos_salvos/detector_yolo11_gpu_completo/treinamento")
BEST_WEIGHTS = PROJECT_DIR / "weights" / "best.pt"


def main() -> None:
    if not DATASET_YAML.exists():
        raise SystemExit(f"Dataset YAML não encontrado em {DATASET_YAML}")
    if not BEST_WEIGHTS.exists():
        raise SystemExit(f"Pesos best.pt não encontrados em {BEST_WEIGHTS}")

    print(f"Carregando modelo de {BEST_WEIGHTS}...")
    model = YOLO(str(BEST_WEIGHTS))

    if not torch.cuda.is_available():
        device = "cpu"
        print("CUDA não disponível, validando em CPU (mais lento)...")
    else:
        device = "cuda"
        print("CUDA disponível, validando em GPU...")

    print("Executando validação no split de teste...")
    metrics = model.val(data=str(DATASET_YAML), split="test", device=device, workers=0)

    # Extrair métricas principais
    try:
        p = metrics.box.p
        r = metrics.box.r
        try:
            import numpy as np  # type: ignore

            precision = float(np.mean(p))  # type: ignore[arg-type]
            recall = float(np.mean(r))  # type: ignore[arg-type]
        except Exception:  # pylint: disable=broad-except
            precision = float(p if not hasattr(p, "mean") else p.mean())  # type: ignore[arg-type]
            recall = float(r if not hasattr(r, "mean") else r.mean())  # type: ignore[arg-type]

        mAP50 = float(metrics.box.map50)
        mAP50_95 = float(metrics.box.map)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[AVISO] Falha ao extrair métricas de validação: {exc}")
        precision = recall = mAP50 = mAP50_95 = 0.0

    metricas_tcc: Dict[str, Any] = {
        "dataset": {
            "path": str(DATASET_YAML),
        },
        "treinamento": {
            "modelo": str(BEST_WEIGHTS),
        },
        "metricas_finais": {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "precision": precision,
            "recall": recall,
        },
    }

    output_path = PROJECT_DIR / "metricas_tcc.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metricas_tcc, f, indent=2, ensure_ascii=False)

    print(f"[OK] Métricas de teste salvas em: {output_path}")
    print(json.dumps(metricas_tcc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
