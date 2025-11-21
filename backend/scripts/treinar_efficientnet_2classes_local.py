#!/usr/bin/env python3
"""Treinar EfficientNet-B4 binário (limpo/sujo) em máquina local (CUDA/MPS/CPU).

Uso típico (no Dell, com CUDA):
    python scripts/treinar_efficientnet_2classes_local.py \
        --dataset-dir /CAMINHO/para/dataset_2classes_final \
        --output-dir /CAMINHO/para/modelos_classificador_2classes \
        --resume-from /CAMINHO/opcional/para/checkpoint_epoch_2.pth

- Detecta automaticamente CUDA, MPS ou CPU.
- Usa batch_size maior em CUDA (treino bem mais rápido no Dell G15).
- Salva status em JSON para monitoramento e um resultado_final.json ao final.
"""

import argparse
import sys
import json
import time
import logging
from pathlib import Path

import torch

# Localizar raiz do backend (este script está em backend/scripts)
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_ROOT))

from aplicacao.modelos.treinamento_classificador import (  # type: ignore
    TreinadorClassificador,
    obter_status_treinamento_classificador,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino EfficientNet-B4 2 classes (limpo/sujo) com dataset misto."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Diretório com dataset_2classes_final (contendo train/val/test/limpo,sujo)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Diretório para salvar modelos e relatórios (será criado se não existir)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="(Opcional) Caminho para checkpoint .pth para retomar treinamento (por exemplo checkpoint_epoch_2.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Número máximo de épocas (early stopping pode parar antes)",
    )
    return parser.parse_args()


def escolher_batch_size() -> int:
    """Escolhe batch_size de forma adaptativa conforme hardware."""
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            mem_total = getattr(props, "total_memory", 0)
        except Exception:
            mem_total = 0
        if mem_total >= 8 * 1024**3:
            logger.info("💻 CUDA disponível (>=8GB) - usando batch_size=32")
            return 32
        else:
            logger.info("💻 CUDA disponível (<8GB) - usando batch_size=8")
            return 8
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("🍏 MPS disponível - usando batch_size=6 (seguro no Mac)")
        return 6
    logger.info("⚙️  Sem CUDA/MPS - usando CPU com batch_size=6")
    return 6


def treinar_local(dataset_dir: Path, output_dir: Path, resume_from: Path | None, epocas: int) -> dict:
    logger.info("🚀 Iniciando treinamento EfficientNet-B4 (2 classes: limpo/sujo)")
    logger.info(f"📂 Dataset: {dataset_dir}")
    logger.info(f"📁 Output:  {output_dir}")

    if not dataset_dir.exists():
        logger.error(f"❌ Dataset não encontrado em {dataset_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status_treinamento.json"

    treinador = TreinadorClassificador(
        diretorio_dataset=str(dataset_dir),
        modelo_base="efficientnet_b0",
    )

    def callback_progresso(epoca_atual: int, total_epocas: int, metricas: dict) -> None:
        """Salva status de treinamento em JSON para monitoramento externo."""
        try:
            estado = obter_status_treinamento_classificador()
            estado.update(
                {
                    "treinando": True,
                    "epoca_atual": epoca_atual,
                    "total_epocas": total_epocas,
                    "progresso": int((epoca_atual / max(1, total_epocas)) * 100),
                    "metricas": metricas,
                }
            )
            if "tempo_restante_sec" not in estado and "tempo_restante_seg" in estado:
                estado["tempo_restante_sec"] = estado.get("tempo_restante_seg")
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(estado, f, indent=2, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Erro ao salvar status: {e}")

    inicio = time.time()

    batch_size = escolher_batch_size()

    resultado = treinador.treinar(
        epocas=epocas,
        learning_rate=1e-4,
        weight_decay=1e-4,
        patience=5,
        batch_size=batch_size,
        callback_progresso=callback_progresso,
        diretorio_saida=str(output_dir),
        use_focal=True,
        unfreeze_epoch=0,
        unfreeze_lr_factor=1.0,
        resume_weights_path=str(resume_from) if resume_from is not None else None,
    )

    tempo_total = time.time() - inicio

    logger.info("\n✅ Treinamento concluído")
    logger.info(f"⏱️  Tempo total: {tempo_total / 3600:.2f} h")
    logger.info(
        f"📊 Acurácia validação final: {resultado['metricas_finais'].get('val_acc', 0):.4f}"
    )

    with open(output_dir / "resultado_final.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tempo_total_horas": tempo_total / 3600,
                "metricas": resultado.get("metricas_finais", {}),
                "historico": resultado.get("historico", {}),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return resultado


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    resume_from = Path(args.resume_from).expanduser().resolve() if args.resume_from else None

    print("=" * 70)
    print("🎯 TREINAMENTO EFFICIENTNET-B4 - 2 CLASSES (LIMPO / SUJO) - LOCAL")
    print("=" * 70)
    print()

    treinar_local(dataset_dir, output_dir, resume_from, args.epochs)

    print("\n" + "=" * 70)
    print("✅ CONCLUÍDO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
