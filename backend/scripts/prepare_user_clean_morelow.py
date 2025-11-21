#!/usr/bin/env python3
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent.parent))
from aplicacao.modelos.detector_modulos import DetectorModulos
from aplicacao.config import DIRETORIOS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prep_user_clean_morelow")

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"}

def list_images(folder: Path):
    return [p for p in folder.glob("**/*") if p.suffix in SUPPORTED and p.is_file() and not p.name.startswith("._")]

def main():
    base = Path(DIRETORIOS["plantas_completas"]) / "imagens"
    lavado_dir = base / "lavado"
    out_limpo = Path(DIRETORIOS["modulos_individuais"]) / "limpo"
    out_limpo.mkdir(parents=True, exist_ok=True)

    detector = DetectorModulos(caminho_modelo=None, modelo_size='s')
    conf_primary = 0.20
    conf_fallback = 0.15

    imgs = list_images(lavado_dir) if lavado_dir.exists() else []
    logger.info(f"Processando {len(imgs)} imagens 'lavado' com conf {conf_primary} e fallback {conf_fallback}")

    total = 0
    crops = 0
    for img in imgs:
        total += 1
        dets = detector.detectar(str(img), confianca_min=conf_primary)
        if not dets:
            dets = detector.detectar(str(img), confianca_min=conf_fallback)
        if not dets:
            continue
        recs = detector.recortar_modulos(str(img), dets, salvar_dir=str(out_limpo))
        crops += len(recs)

    logger.info(f"Concluído LIMPO (morelow). Imagens processadas: {total}. Módulos recortados: {crops}.")

if __name__ == "__main__":
    main()
