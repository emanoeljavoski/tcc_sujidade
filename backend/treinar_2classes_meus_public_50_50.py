"""Treino do classificador EfficientNet-B4 binário (limpo/sujo)
utilizando o dataset balanceado 50/50 em
F:\\dataset_2classes_meus_public_50_50.

Os pesos são inicializados do ImageNet (sem continuar do checkpoint antigo),
para o modelo se adaptar ao novo dataset balanceado.
"""

from pathlib import Path
import logging

from aplicacao.modelos.treinamento_classificador import TreinadorClassificador


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("treino_2classes_meus_public_50_50")

    dataset_dir = Path(r"F:\dataset_2classes_meus_public_50_50")
    saida_dir = Path(r"F:\modelos_salvos\classificador_2classes_meus_public_50_50")
    saida_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")

    logger.info("Iniciando treino do EfficientNet-B4 (2 classes) com dataset 50/50 (meus + públicos)...")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Saída: {saida_dir}")

    treinador = TreinadorClassificador(str(dataset_dir), modelo_base="efficientnet_b4")

    resultado = treinador.treinar(
        epocas=20,
        learning_rate=5e-4,
        batch_size=8,
        diretorio_saida=str(saida_dir),
        resume_weights_path=None,
        patience=5,
    )

    logger.info("Treino concluído.")
    logger.info(f"Métricas finais: {resultado.get('metricas_finais')}")


if __name__ == "__main__":
    main()
