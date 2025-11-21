#!/usr/bin/env python3
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent.parent))
from aplicacao.modelos.detector_modulos import DetectorModulos
from aplicacao.config import DIRETORIOS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prep_user_modules")

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"}

def list_images(folder: Path):
    return [p for p in folder.glob("**/*") if p.suffix in SUPPORTED and p.is_file()]

def main():
    base = Path(DIRETORIOS["plantas_completas"]) / "imagens"
    sujo_dir = base / "sujo"
    lavado_dir = base / "lavado"
    out_sujo = Path(DIRETORIOS["modulos_individuais"]) / "sujo"
    out_limpo = Path(DIRETORIOS["modulos_individuais"]) / "limpo"
    out_sujo.mkdir(parents=True, exist_ok=True)
    out_limpo.mkdir(parents=True, exist_ok=True)

    # Detector com YOLO11s e conf baixa
    detector = DetectorModulos(caminho_modelo=None, modelo_size='s')
    conf = 0.30

    total_imgs = 0
    total_crops = 0

    for cond, src_dir, out_dir in [("sujo", sujo_dir, out_sujo), ("lavado", lavado_dir, out_limpo)]:
        if not src_dir.exists():
            continue
        imgs = list_images(src_dir)
        logger.info(f"Processando {len(imgs)} imagens de '{cond}' em {src_dir}")
        for img in imgs:
            total_imgs += 1
            dets = detector.detectar(str(img), confianca_min=conf)
            if not dets:
                continue
            recs = detector.recortar_modulos(str(img), dets, salvar_dir=str(out_dir))
            total_crops += len(recs)
    logger.info(f"Concluído. Imagens processadas: {total_imgs}. Módulos recortados: {total_crops}.")

if __name__ == "__main__":
    main()
