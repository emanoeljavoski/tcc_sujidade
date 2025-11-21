"""Avalia o classificador EfficientNet-B4 binário (limpo/sujo)
treinado no dataset balanceado 50/50 em
F:\\dataset_2classes_meus_public_50_50.

Gera matriz de confusão e métricas no split de teste.
"""

from pathlib import Path

from avaliar_classificador_2classes import avaliar_classificador_2classes


def main() -> None:
    dataset_dir = Path(r"F:\dataset_2classes_meus_public_50_50")
    checkpoint_path = Path(r"F:\modelos_salvos\classificador_2classes_meus_public_50_50\checkpoint_epoch_2.pth")
    saida_dir = checkpoint_path.parent

    avaliar_classificador_2classes(dataset_dir, checkpoint_path, saida_dir, split="test")


if __name__ == "__main__":
    main()
