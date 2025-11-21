"""Monta um dataset binário 2-classes balanceado 50/50 (limpo/sujo)
priorizando os dados locais (casaemanoel, gustavo1, setor2) e
complementando com dados públicos do dataset_2classes_final.

Saída: F:\\dataset_2classes_meus_public_50_50\\{train,val,test}\\{limpo,sujo}
"""

from __future__ import annotations

from pathlib import Path
import logging
import random
import shutil
from typing import List, Tuple


def listar_imagens(pasta: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted([p for p in pasta.rglob("*") if p.is_file() and p.suffix in exts])


def montar_conjunto_balanceado(
    meus: List[Path],
    publicos: List[Path],
    target: int,
    seed: int = 42,
) -> List[Path]:
    """Retorna uma lista de paths de tamanho target, priorizando meus dados
    e completando com públicos.
    """
    random.seed(seed)

    if len(meus) >= target:
        return random.sample(meus, target)

    restantes = target - len(meus)
    if restantes > len(publicos):
        restantes = len(publicos)
        target = len(meus) + restantes

    escolhidos_publicos = random.sample(publicos, restantes)
    return meus + escolhidos_publicos


def split_train_val_test(
    arquivos: List[Path], proporcoes: Tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    random.seed(seed)
    arquivos_baralhados = arquivos[:]
    random.shuffle(arquivos_baralhados)

    n = len(arquivos_baralhados)
    n_train = int(n * proporcoes[0])
    n_val = int(n * proporcoes[1])
    n_test = n - n_train - n_val

    train = arquivos_baralhados[:n_train]
    val = arquivos_baralhados[n_train : n_train + n_val]
    test = arquivos_baralhados[n_train + n_val :]
    return train, val, test


def copiar_arquivos(arquivos: List[Path], destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for src in arquivos:
        dst = destino / src.name
        # Evitar overwrite: se já existir, adicionar sufixo
        if dst.exists():
            base = dst.stem
            ext = dst.suffix
            contador = 1
            while True:
                cand = destino / f"{base}_{contador}{ext}"
                if not cand.exists():
                    dst = cand
                    break
                contador += 1
        shutil.copy2(src, dst)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("montar_dataset_meus_public_50_50")

    raiz_meus = Path(r"F:\dataset_2classes_meus")
    raiz_publico = Path(r"F:\dataset_2classes_final")
    raiz_dest = Path(r"F:\dataset_2classes_meus_public_50_50")

    if not raiz_meus.exists():
        raise FileNotFoundError(f"Dataset 'meus' não encontrado: {raiz_meus}")
    if not raiz_publico.exists():
        raise FileNotFoundError(f"Dataset público não encontrado: {raiz_publico}")

    # Listar meus dados
    meus_limpo = listar_imagens(raiz_meus / "limpo")
    meus_sujo = listar_imagens(raiz_meus / "sujo")

    logger.info("Meus dados - limpo: %d, sujo: %d", len(meus_limpo), len(meus_sujo))

    # Dados públicos do split de treino atual
    pub_train_limpo = listar_imagens(raiz_publico / "train" / "limpo")
    pub_train_sujo = listar_imagens(raiz_publico / "train" / "sujo")

    logger.info("Públicos treino - limpo: %d, sujo: %d", len(pub_train_limpo), len(pub_train_sujo))

    # Definir tamanho alvo por classe (train) de forma equilibrada, usando
    # todos os dados disponíveis (meus + públicos) sem limite artificial.
    max_limpo = len(meus_limpo) + len(pub_train_limpo)
    max_sujo = len(meus_sujo) + len(pub_train_sujo)
    target_por_classe = min(max_limpo, max_sujo)

    logger.info("Target por classe (train): %d", target_por_classe)

    escolhidos_limpo = montar_conjunto_balanceado(meus_limpo, pub_train_limpo, target_por_classe)
    escolhidos_sujo = montar_conjunto_balanceado(meus_sujo, pub_train_sujo, target_por_classe)

    logger.info("Escolhidos para treino - limpo: %d, sujo: %d", len(escolhidos_limpo), len(escolhidos_sujo))

    # Gerar splits train/val/test por classe
    train_l, val_l, test_l = split_train_val_test(escolhidos_limpo)
    train_s, val_s, test_s = split_train_val_test(escolhidos_sujo)

    logger.info(
        "Splits limpo - train: %d, val: %d, test: %d", len(train_l), len(val_l), len(test_l)
    )
    logger.info(
        "Splits sujo - train: %d, val: %d, test: %d", len(train_s), len(val_s), len(test_s)
    )

    # Copiar arquivos para a nova estrutura
    for subset_name, arquivos_l, arquivos_s in [
        ("train", train_l, train_s),
        ("val", val_l, val_s),
        ("test", test_l, test_s),
    ]:
        dest_limpo = raiz_dest / subset_name / "limpo"
        dest_sujo = raiz_dest / subset_name / "sujo"
        logger.info("Copiando subset %s: limpo=%d, sujo=%d", subset_name, len(arquivos_l), len(arquivos_s))
        copiar_arquivos(arquivos_l, dest_limpo)
        copiar_arquivos(arquivos_s, dest_sujo)

    logger.info("Dataset 50/50 montado em %s", raiz_dest)


if __name__ == "__main__":
    main()
