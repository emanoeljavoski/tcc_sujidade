import os
import numpy as np
import matplotlib.pyplot as plt


def gerar_matriz_confusao_esquematica_png(saida: str) -> None:
    """Gera a figura 3.2 (matriz de confusão esquemática 4x4) como PNG.

    A imagem é conceitual: mostra uma matriz 4x4 com as classes
    [Limpo, Pouco sujo, Sujo, Muito sujo] nos eixos Real/Predito
    e marca a diagonal principal com "checks" (✔) para indicar acertos.
    """
    classes = ["Limpo", "Pouco sujo", "Sujo", "Muito sujo"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Eixos e ticks
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_yticklabels(classes)

    # Desenhar grade 4x4
    for i in range(4):  # linhas (Real)
        for j in range(4):  # colunas (Predito)
            rect = plt.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#999",
                linewidth=1.0,
            )
            ax.add_patch(rect)

    # Diagonal principal (acertos) com "✔"
    for i in range(4):
        ax.text(
            i,
            i,
            "✔",
            ha="center",
            va="center",
            fontsize=14,
            color="#333",
        )

    # Ajustar limites e orientação (linha 0 em cima)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)

    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

    plt.tight_layout()

    # Garantir diretório de saída
    os.makedirs(os.path.dirname(saida), exist_ok=True)

    plt.savefig(saida, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Caminho padrão usado no TCC_METODOLOGIA_RESULTADOS.html
    caminho_saida = os.path.join("figuras", "figura3_2_matriz_confusao_esquematica.png")
    gerar_matriz_confusao_esquematica_png(caminho_saida)
    print(f"Figura gerada em: {caminho_saida}")
