"""
Baixa datasets públicos de sujidade em painéis solares.
- DeepSolarEye (Google Drive)
- Roboflow Solar Panel Dust Detection (requer ROBOFLOW_API_KEY)
- Duke UAV (Figshare API)

Uso:
    python scripts/download_datasets.py \
        --out-dir dados/datasets_publicos \
        --roboflow-version 1

Obs.:
- Por padrão salva em dados/datasets_publicos dentro do repositório. 
  Para usar o caminho pedido pelo usuário, passe --out-dir /home/claude/datasets_publicos
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import zipfile
import tarfile
import time
from pathlib import Path
from typing import Optional

import requests

# Dependências opcionais (tratadas com fallback amigável)
try:
    import gdown  # para Google Drive
except Exception:  # pragma: no cover
    gdown = None

logger = logging.getLogger("download_datasets")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _extract_any(archive: Path, dest: Path) -> bool:
    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive, 'r') as zf:
                zf.extractall(dest)
            return True
        if archive.suffix.lower() in {".tar", ".gz", ".tgz", ".bz2"}:
            try:
                with tarfile.open(archive, 'r:*') as tf:
                    tf.extractall(dest)
                return True
            except Exception:
                return False
        return False
    except Exception as e:
        logger.error(f"Erro extraindo {archive}: {e}")
        return False


# ------------------ DeepSolarEye (Google Drive) ------------------
DEEPEYE_FILE_ID = "1qB5dPWZMi2-12sLHDykHb9i6GibbJ46l"


def download_deepsolareye(out_dir: Path) -> Optional[Path]:
    """Baixa DeepSolarEye de um link público do Google Drive.
    Retorna o diretório base onde os dados foram extraídos/colocados.
    """
    dst_base = out_dir / "DeepSolarEye"
    ensure_dir(dst_base)

    # Tenta baixar como arquivo zip, senão tenta baixar pasta
    dst_zip = dst_base / "deepsolareye.zip"
    try:
        if gdown is None:
            logger.warning("gdown não instalado. Instale para baixar do Google Drive: pip install gdown")
            return None
        downloaded_ok = False
        if not dst_zip.exists():
            logger.info("📥 Baixando DeepSolarEye (tentando arquivo zip)...")
            out = gdown.download(id=DEEPEYE_FILE_ID, output=str(dst_zip), quiet=False, fuzzy=True)
            downloaded_ok = bool(out)
        else:
            downloaded_ok = True
            logger.info("✅ Zip já existente. Pulando download de arquivo.")

        if downloaded_ok and dst_zip.exists():
            # Extrair se for zip
            if _extract_any(dst_zip, dst_base):
                logger.info("✅ DeepSolarEye extraído.")
            else:
                logger.info("ℹ️ Arquivo baixado não é zip/tar reconhecido. Manter como está.")
        else:
            # Fallback: tentar baixar pasta inteira
            logger.info("📥 Tentando baixar como pasta do Google Drive...")
            try:
                gdown.download_folder(id=DEEPEYE_FILE_ID, output=str(dst_base), quiet=False, use_cookies=False)
                logger.info("✅ DeepSolarEye (pasta) baixado.")
            except Exception as ee:
                logger.error(f"Falha no download de pasta: {ee}")
                return None
        return dst_base
    except Exception as e:
        logger.error(f"❌ Erro ao baixar DeepSolarEye: {e}")
        return None


# ------------------ Roboflow (Universe) ------------------

def download_roboflow(out_dir: Path, api_key: Optional[str], workspace: str,
                      project: str, version: int = 1, format: str = "pytorch") -> Optional[Path]:
    """Baixa dataset do Roboflow Universe via SDK (requer ROBOFLOW_API_KEY)."""
    try:
        if not api_key:
            logger.info("🔑 ROBOFLOW_API_KEY não definido. Pulando Roboflow.")
            return None
        from roboflow import Roboflow  # lazy import
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        ds = proj.version(version).download(model_format=format, location=str(out_dir / "Roboflow"))
        logger.info(f"✅ Roboflow baixado em: {ds.location}")
        return Path(ds.location)
    except Exception as e:
        logger.error(f"❌ Erro ao baixar Roboflow: {e}")
        return None


# ------------------ Duke UAV (Figshare API) ------------------
FIGSHARE_ARTICLE_ID = 18093890


def download_duke_uav(out_dir: Path) -> Optional[Path]:
    """Baixa o Duke UAV dataset via Figshare API (melhor esforço)."""
    try:
        api_url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
        r = requests.get(api_url, timeout=30)
        r.raise_for_status()
        data = r.json()
        files = data.get("files", [])
        if not files:
            logger.warning("⚠️ Nenhum arquivo listado na API Figshare. Pulando.")
            return None
        dst = out_dir / "Duke_UAV"
        ensure_dir(dst)
        for f in files:
            name = f.get("name")
            url = f.get("download_url")
            if not url or not name:
                continue
            logger.info(f"📥 Baixando {name}...")
            local_path = dst / name
            if local_path.exists():
                logger.info("   já existe, pulando.")
                continue
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(local_path, 'wb') as fp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fp.write(chunk)
            # Extrair se for arquivo compactado
            try:
                _extract_any(local_path, dst)
            except Exception:
                pass
        logger.info(f"✅ Duke UAV salvo em: {dst}")
        return dst
    except Exception as e:
        logger.error(f"❌ Erro no download Duke UAV: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Baixar datasets públicos de sujidade")
    parser.add_argument("--out-dir", type=str, default=str(Path("dados") / "datasets_publicos"),
                        help="Diretório de saída para os datasets")
    parser.add_argument("--roboflow-workspace", type=str, default="solarpaneldustdetection")
    parser.add_argument("--roboflow-project", type=str, default="solar-panel-dust-detection-5qnt0")
    parser.add_argument("--roboflow-version", type=int, default=1)
    parser.add_argument("--roboflow-format", type=str, default="pytorch", choices=["pytorch", "yolov8"]) 
    parser.add_argument("--skip-duke", action="store_true")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    logger.info("=" * 80)
    logger.info("🚀 DOWNLOAD DE DATASETS PÚBLICOS")
    logger.info("=" * 80)

    # DeepSolarEye
    dse_dir = download_deepsolareye(out_dir)

    # Roboflow
    roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    rf_dir = download_roboflow(out_dir, roboflow_api_key, args.roboflow_workspace,
                               args.roboflow_project, args.roboflow_version, args.roboflow_format)

    # Duke UAV (opcional)
    duke_dir = None
    if not args.skip_duke:
        duke_dir = download_duke_uav(out_dir)

    logger.info("=" * 80)
    logger.info("📊 RESUMO DE DOWNLOADS")
    logger.info(f"DeepSolarEye: {'OK' if dse_dir else 'N/A'}")
    logger.info(f"Roboflow: {'OK' if rf_dir else 'N/A'}")
    logger.info(f"Duke UAV: {'OK' if duke_dir else 'N/A'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
