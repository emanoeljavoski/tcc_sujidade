"""Gera recortes de módulos LIMPOS a partir das imagens de drone em
F:\\meus_dados_drone\\limpo, usando o detector YOLO11, e adiciona esses
recortes ao conjunto de treino binário em F:\\dataset_2classes_final\\train\\limpo.

Objetivo: aumentar a quantidade de exemplos limpos no dataset binário.
"""

from pathlib import Path
import logging
from typing import List

from aplicacao.modelos.detector_modulos import DetectorModulos


def listar_imagens(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    arquivos: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            arquivos.append(p)
    return sorted(arquivos)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("gerar_crops_limpos")

    origem_root = Path(r"F:\meus_dados_drone\limpo")
    destino_limpo = Path(r"F:\dataset_2classes_final\train\limpo")

    if not origem_root.exists():
        raise FileNotFoundError(f"Diretório de origem não encontrado: {origem_root}")
    destino_limpo.mkdir(parents=True, exist_ok=True)

    logger.info("Listando imagens em %s", origem_root)
    imagens = listar_imagens(origem_root)
    logger.info("Encontradas %d imagens de drone limpas", len(imagens))

    if not imagens:
        logger.warning("Nenhuma imagem encontrada em %s", origem_root)
        return

    # Caminho do modelo YOLO11 treinado para detecção de módulos
    modelo_yolo = Path(r"F:\modelos_salvos\yolo\yolo11n_solar_dust_roboflow_v3\weights\best.pt")
    if not modelo_yolo.exists():
        raise FileNotFoundError(f"Modelo YOLO para detecção não encontrado: {modelo_yolo}")

    detector = DetectorModulos(caminho_modelo=str(modelo_yolo), modelo_size="n")

    total_modulos = 0
    for idx, img_path in enumerate(imagens, start=1):
        logger.info("(%d/%d) Detectando módulos em %s", idx, len(imagens), img_path.name)
        deteccoes = detector.detectar(str(img_path), confianca_min=0.25, imgsz=1280)
        if not deteccoes:
            logger.info("  Nenhum módulo detectado em %s", img_path.name)
            continue

        logger.info("  %d módulos detectados, recortando...", len(deteccoes))
        recortes = detector.recortar_modulos(str(img_path), deteccoes, salvar_dir=str(destino_limpo))
        logger.info("  %d recortes salvos em %s", len(recortes), destino_limpo)
        total_modulos += len(recortes)

    logger.info("Processo concluído. Total de novos módulos limpos adicionados: %d", total_modulos)


if __name__ == "__main__":
    main()
