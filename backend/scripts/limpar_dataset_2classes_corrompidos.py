import os
from pathlib import Path

from PIL import Image


DATASET_DIR = Path(r"D:\dataset_2classes_final")
BACKUP_DIR = DATASET_DIR / "_corrompidas"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS


def limpar_dataset() -> None:
    total_checadas = 0
    total_movidas = 0

    print("Verificando imagens em:", DATASET_DIR)

    for split in ("train", "val", "test"):
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for img_path in class_dir.iterdir():
                if not is_image_file(img_path):
                    continue

                total_checadas += 1

                try:
                    # Abrir o arquivo em modo binário e verificar com PIL
                    with open(img_path, "rb") as f:
                        img = Image.open(f)
                        img.verify()
                except Exception as e:
                    # Qualquer erro de IO/PIL -> mover para pasta _corrompidas
                    rel = img_path.relative_to(DATASET_DIR)
                    dest = BACKUP_DIR / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)

                    print(f"Movendo arquivo problemático: {img_path} -> {dest} ({e})")
                    try:
                        img_path.replace(dest)
                        total_movidas += 1
                    except Exception as e2:
                        print(f"Falha ao mover {img_path}: {e2}")

    print(f"Total de imagens checadas: {total_checadas}")
    print(f"Total de imagens problemáticas movidas: {total_movidas}")
    if total_movidas > 0:
        print("Arquivos problemáticos foram movidos para:", BACKUP_DIR)


if __name__ == "__main__":
    limpar_dataset()
