"""Script de treino YOLO11 otimizado para GPU CUDA.

Este script:
- Usa o dataset consolidado em F:/dataset_yolo_completo/dataset.yaml
- Verifica a disponibilidade de CUDA
- Aplica um monkeypatch em OpenCV para contornar constantes/funções
  ausentes em instalações headless
- Treina YOLO11 (n ou s) com configuração adequada para GPU
- Executa validação no split de teste
- Gera um arquivo metricas_tcc.json com os valores consolidados de métricas

Uso recomendado (PowerShell, com o venv ativado):

    cd F:\tccemanoel\sistema-paineis-solares\backend
    ..\.venv\Scripts\python.exe treinar_yolo11_gpu_otimizado.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def corrigir_opencv() -> None:
    """Corrige atributos ausentes em cv2 para evitar erros na Ultralytics.

    Não instala nada; apenas adiciona constantes e funções dummy se
    necessário. Deve ser chamado ANTES de importar ultralytics.YOLO.
    """

    try:
        import cv2  # type: ignore

        if not hasattr(cv2, "IMREAD_COLOR"):
            cv2.IMREAD_COLOR = 1
        if not hasattr(cv2, "IMREAD_GRAYSCALE"):
            cv2.IMREAD_GRAYSCALE = 0
        if not hasattr(cv2, "IMREAD_UNCHANGED"):
            cv2.IMREAD_UNCHANGED = -1

        if not hasattr(cv2, "imshow"):
            cv2.imshow = lambda *args, **kwargs: None  # type: ignore[assignment]
        if not hasattr(cv2, "waitKey"):
            cv2.waitKey = lambda *args, **kwargs: -1  # type: ignore[assignment]
        if not hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows = lambda: None  # type: ignore[assignment]

        print("[OK] OpenCV corrigido")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[AVISO] Erro ao corrigir OpenCV: {exc}")


# Executar monkeypatch antes de importar YOLO
corrigir_opencv()

import torch
from ultralytics import YOLO


DATASET_ROOT = Path("F:/dataset_yolo_completo")
DATASET_YAML = DATASET_ROOT / "dataset.yaml"
RELATORIO_CONSOLIDACAO = DATASET_ROOT / "relatorio_consolidacao.json"


def configurar_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def verificar_cuda() -> None:
    """Garante que CUDA está disponível, ou aborta com erro claro."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível! Verifique drivers e instalação do PyTorch com suporte a CUDA.")

    nome = torch.cuda.get_device_name(0)
    versao_cuda = torch.version.cuda
    logging.info("GPU detectada: %s", nome)
    logging.info("Versão CUDA: %s", versao_cuda)


def carregar_resumo_dataset() -> Dict[str, Any]:
    """Carrega o relatório de consolidação, se existir, ou gera um resumo leve.

    O consolidar_datasets_yolo.py gera relatorio_consolidacao.json com
    total_datasets e total_imagens. Caso esse arquivo não exista, este
    método estima o total de imagens contando arquivos em train/val/test.
    """

    if RELATORIO_CONSOLIDACAO.exists():
        try:
            with RELATORIO_CONSOLIDACAO.open("r", encoding="utf-8") as f:
                dados = json.load(f)
            return {
                "total_datasets": dados.get("total_datasets"),
                "total_imagens": dados.get("total_imagens"),
            }
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Falha ao ler %s: %s", RELATORIO_CONSOLIDACAO, exc)

    # Fallback: contar imagens
    total_imagens = 0
    for split in ("train", "val", "test"):
        dir_imgs = DATASET_ROOT / split / "images"
        if dir_imgs.exists():
            total_imagens += sum(1 for _ in dir_imgs.glob("*.*"))

    return {
        "total_datasets": None,
        "total_imagens": total_imagens,
    }


def extrair_metricas_de_results_csv(project_dir: Path) -> Dict[str, float]:
    """Lê a última linha de results.csv para obter losses de validação.

    Retorna um dicionário com box_loss, cls_loss e dfl_loss (0.0 se não
    forem encontrados).
    """

    import csv

    results_csv = project_dir / "results.csv"
    if not results_csv.exists():
        logging.warning("results.csv não encontrado em %s", results_csv)
        return {"box_loss": 0.0, "cls_loss": 0.0, "dfl_loss": 0.0}

    last_row: Dict[str, str] | None = None
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    if not last_row:
        logging.warning("results.csv em %s está vazio.", results_csv)
        return {"box_loss": 0.0, "cls_loss": 0.0, "dfl_loss": 0.0}

    def _parse(name: str) -> float:
        try:
            return float(last_row.get(name, 0.0))
        except Exception:  # pylint: disable=broad-except
            return 0.0

    return {
        "box_loss": _parse("val/box_loss"),
        "cls_loss": _parse("val/cls_loss"),
        "dfl_loss": _parse("val/dfl_loss"),
    }


def criar_config() -> Dict[str, Any]:
    """Cria o dicionário de configuração de treino para a Ultralytics."""

    return {
        # Dataset
        "data": str(DATASET_YAML),

        # Modelo
        "model": "yolo11n.pt",  # trocar para yolo11s.pt se a GPU tiver VRAM suficiente

        # Treinamento
        "epochs": 200,
        "batch": -1,  # auto-batch (máximo que a GPU suportar)
        "imgsz": 640,

        # Otimizador
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,

        # Scheduler
        "cos_lr": True,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # Loss
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,

        # Hardware
        "device": "cuda",
        "workers": 0,
        "amp": True,

        # Data augmentation / regularização
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,

        # Validação / salvamento
        "patience": 50,
        "save_period": 10,

        # Saída
        "project": "F:/modelos_salvos/detector_yolo11_gpu_completo",
        "name": "treinamento",
        "exist_ok": True,
        "verbose": True,
        "plots": True,
    }


def treinar_com_fallback(model: YOLO, config: Dict[str, Any]):
    tentativa = 0
    max_tentativas = 4

    while tentativa < max_tentativas:
        try:
            return model.train(**config)
        except RuntimeError as exc:  # pylint: disable=broad-except
            msg = str(exc).lower()
            if "out of memory" not in msg and "cuda error" not in msg:
                raise

            logging.warning("Falha de CUDA OOM no treino com batch=%s: %s", config.get("batch"), exc)
            tentativa += 1

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:  # pylint: disable=broad-except
                    pass

            atual = config.get("batch", -1)
            if isinstance(atual, int):
                if atual == -1:
                    novo_batch = 16
                elif atual > 1:
                    novo_batch = max(1, atual // 2)
                else:
                    novo_batch = 1
            else:
                novo_batch = 4

            if novo_batch == atual or novo_batch < 1:
                logging.error("Não foi possível ajustar o batch size para evitar OOM.")
                raise

            logging.info("Reduzindo batch de %s para %s e tentando novamente...", atual, novo_batch)
            config["batch"] = novo_batch

    raise RuntimeError("Treino YOLO11 falhou após múltiplas tentativas de reduzir o batch por OOM.")


def main() -> int:
    """Ponto de entrada principal para treino YOLO11 otimizado em GPU."""

    configurar_logging()

    if not DATASET_YAML.exists():
        logging.error("Dataset consolidado não encontrado em %s. Execute primeiro consolidar_datasets_yolo.py.", DATASET_YAML)
        return 1

    verificar_cuda()

    config = criar_config()

    # Criar modelo
    model = YOLO(config["model"])

    project_dir = Path(config["project"]) / config["name"]

    print("\n" + "=" * 80)
    print("INICIANDO TREINO YOLO11 - GPU OTIMIZADO")
    print("=" * 80)
    print(f"\nDataset: {config['data']}")
    print(f"Modelo: {config['model']}")
    print(f"Épocas: {config['epochs']}")
    print(f"Device: {config['device']}")
    print(f"Mixed Precision: {config['amp']}")
    print(f"Batch inicial: {config['batch']}")
    print("\n" + "=" * 80)

    # Treino
    results = treinar_com_fallback(model, config)

    # Validação no conjunto de teste
    print("\nValidando no conjunto de teste...")
    metrics = model.val(data=config["data"], split="test")

    # Resumo de dataset
    resumo_dataset = carregar_resumo_dataset()

    # Métricas finais de losses a partir do CSV
    losses = extrair_metricas_de_results_csv(project_dir)

    # Construir estrutura com as métricas consolidadas
    try:
        # Ultralytics retorna arrays/tensores para p e r; usar média se necessário
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
        logging.warning("Falha ao extrair métricas de validação: %s", exc)
        precision = recall = mAP50 = mAP50_95 = 0.0

    metricas_tcc: Dict[str, Any] = {
        "dataset": {
            "path": str(DATASET_YAML),
            "total_datasets": resumo_dataset.get("total_datasets"),
            "total_imagens": resumo_dataset.get("total_imagens"),
        },
        "treinamento": {
            "modelo": config["model"],
            "epocas_total": config["epochs"],
            # Como o objeto results pode variar entre versões da Ultralytics,
            # usamos simplesmente o valor configurado de épocas.
            "epocas_treinadas": config["epochs"],
            "batch_size": config["batch"],
            "device": config["device"],
            "mixed_precision": config["amp"],
        },
        "metricas_finais": {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "precision": precision,
            "recall": recall,
            "box_loss": losses["box_loss"],
            "cls_loss": losses["cls_loss"],
            "dfl_loss": losses["dfl_loss"],
        },
    }

    output_path = project_dir / "metricas_tcc.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metricas_tcc, f, indent=2, ensure_ascii=False)

    print(f"[OK] Métricas salvas em: {output_path}")

    # Exibir resumo das principais métricas do detector
    print("\n" + "=" * 80)
    print("MÉTRICAS DO TREINAMENTO DO DETECTOR")
    print("=" * 80)
    print(f"\nmAP@0.5:        {metricas_tcc['metricas_finais']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95:   {metricas_tcc['metricas_finais']['mAP50_95']:.4f}")
    print(f"Precision:      {metricas_tcc['metricas_finais']['precision']:.4f}")
    print(f"Recall:         {metricas_tcc['metricas_finais']['recall']:.4f}")
    print(f"Box Loss:       {metricas_tcc['metricas_finais']['box_loss']:.4f}")
    print("=" * 80)

    print("\nTREINO CONCLUÍDO (verifique logs e gráficos gerados pela Ultralytics).")
    print(f"\nModelo: {project_dir / 'weights' / 'best.pt'}")
    print(f"Gráficos: {project_dir / 'results.png'}")
    print(f"CSV: {project_dir / 'results.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
