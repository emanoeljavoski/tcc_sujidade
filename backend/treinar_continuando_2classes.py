"""Treino continuado do classificador EfficientNet-B4 binário (limpo/sujo).

Usa como ponto de partida o checkpoint_epoch_3 da run 20251114_1620
com o dataset binário consolidado em F:\\dataset_2classes_final.
"""

from pathlib import Path
import logging

from aplicacao.modelos.treinamento_classificador import TreinadorClassificador


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("treino_continuado_2classes")

    dataset_dir = Path(r"F:\\dataset_2classes_final")
    checkpoint_path = Path(
        r"F:\\modelos_salvos\\classificador_2classes\\run_20251114_1620\\checkpoint_epoch_3.pth"
    )
    # Novo diretório de saída para o treino continuado rodando na GPU
    saida_dir = Path(r"F:\\modelos_salvos\\classificador_2classes_continuado_gpu")
    saida_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    logger.info("Iniciando treino continuado do EfficientNet-B4 (2 classes)...")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Checkpoint inicial: {checkpoint_path}")
    logger.info(f"Saída: {saida_dir}")

    treinador = TreinadorClassificador(str(dataset_dir), modelo_base="efficientnet_b4")

    resultado = treinador.treinar(
        epocas=5,
        learning_rate=5e-4,
        batch_size=16,
        diretorio_saida=str(saida_dir),
        resume_weights_path=str(checkpoint_path),
        patience=10,
    )

    logger.info("Treino continuado concluído.")
    logger.info(f"Métricas finais: {resultado.get('metricas_finais')}")


if __name__ == "__main__":
    main()
