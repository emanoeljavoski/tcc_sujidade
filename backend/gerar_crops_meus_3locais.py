"""Gera recortes de módulos LIMPOS e SUJOS a partir das imagens de drone em
F:\\meus_dados_drone\\{limpo,sujo} para os conjuntos casaemanoel, gustavo1 e setor2.

Os recortes são salvos em F:\\dataset_2classes_meus\\{limpo,sujo}, para
posterior montagem de um dataset balanceado 50/50 (meus + públicos).
"""

from pathlib import Path
import logging
from typing import List

from aplicacao.modelos.detector_modulos import DetectorModulos

SUBDIRS_FOCO = ["casaemanoel", "gustavo1", "setor2"]


def listar_imagens(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    arquivos: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            arquivos.append(p)
    return sorted(arquivos)


def gerar_crops_por_classe(detector: DetectorModulos, origem_root: Path, destino_root: Path, classe: str) -> int:
    logger = logging.getLogger("gerar_crops_meus_3locais")

    total_recortes = 0
    for sub in SUBDIRS_FOCO:
        pasta = origem_root / sub
        if not pasta.exists():
            logger.warning("Pasta %s não encontrada para classe %s", pasta, classe)
            continue

        logger.info("Listando imagens em %s", pasta)
        imagens = listar_imagens(pasta)
        logger.info("Encontradas %d imagens em %s", len(imagens), pasta.name)

        for idx, img_path in enumerate(imagens, start=1):
            logger.info("[%s] (%d/%d) Detectando módulos em %s", classe.upper(), idx, len(imagens), img_path.name)
            # Usar limiar mais baixo para LIMPO para aumentar recall de módulos
            conf_min = 0.10 if classe == "limpo" else 0.25
            deteccoes = detector.detectar(str(img_path), confianca_min=conf_min, imgsz=1280)
            if not deteccoes:
                logger.info("  Nenhum módulo detectado em %s", img_path.name)
                continue

            logger.info("  %d módulos detectados, recortando...", len(deteccoes))
            recortes = detector.recortar_modulos(str(img_path), deteccoes, salvar_dir=str(destino_root))
            logger.info("  %d recortes salvos em %s", len(recortes), destino_root)
            total_recortes += len(recortes)

    return total_recortes


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("gerar_crops_meus_3locais")

    origem_limpo = Path(r"F:\meus_dados_drone\limpo")
    origem_sujo = Path(r"F:\meus_dados_drone\sujo")
    destino_limpo = Path(r"F:\dataset_2classes_meus\limpo")
    destino_sujo = Path(r"F:\dataset_2classes_meus\sujo")

    if not origem_limpo.exists():
        raise FileNotFoundError(f"Diretório de origem (limpo) não encontrado: {origem_limpo}")
    if not origem_sujo.exists():
        raise FileNotFoundError(f"Diretório de origem (sujo) não encontrado: {origem_sujo}")

    destino_limpo.mkdir(parents=True, exist_ok=True)
    destino_sujo.mkdir(parents=True, exist_ok=True)

    # Caminho do modelo YOLO11 treinado para detecção de módulos
    modelo_yolo = Path(r"F:\modelos_salvos\yolo\yolo11n_solar_dust_roboflow_v3\weights\best.pt")
    if not modelo_yolo.exists():
        raise FileNotFoundError(f"Modelo YOLO para detecção não encontrado: {modelo_yolo}")

    detector = DetectorModulos(caminho_modelo=str(modelo_yolo), modelo_size="n")

    logger.info("Iniciando geração de recortes para LIMPO (casaemanoel/gustavo1/setor2)...")
    total_limpo = gerar_crops_por_classe(detector, origem_limpo, destino_limpo, "limpo")
    logger.info("Total de novos módulos LIMPOS recortados: %d", total_limpo)

    logger.info("Iniciando geração de recortes para SUJO (casaemanoel/gustavo1/setor2)...")
    total_sujo = gerar_crops_por_classe(detector, origem_sujo, destino_sujo, "sujo")
    logger.info("Total de novos módulos SUJOS recortados: %d", total_sujo)

    logger.info("Processo concluído. Recortes salvos em %s", destino_limpo.parent)


if __name__ == "__main__":
    main()
