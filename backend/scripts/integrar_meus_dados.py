#!/usr/bin/env python3
"""
Integra imagens de drone do usuário ao dataset final.
Regra: TODAS as imagens do usuário vão para o TEST SET (avaliação real).
Estrutura esperada:
/Volumes/Z Slim/meus_dados_drone/
  ├─ limpo/
  └─ sujo/
"""
from pathlib import Path
import shutil

SOURCE = Path("/Volumes/Z Slim/meus_dados_drone")
DATASET = Path("/Volumes/Z Slim/dataset_final")

CLASSES = ["limpo", "sujo"]

def integrar():
    moved = 0
    for classe in CLASSES:
        src = SOURCE / classe
        dst = DATASET / "test" / classe
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for img in src.glob(ext):
                new_name = f"MEUS_{img.name}"
                shutil.copy2(img, dst / new_name)
                moved += 1
                print(f"✓ Copiado: {classe}/{new_name}")
    print(f"\n✅ Integração concluída! Total copiados: {moved}")
    print("Test set agora contém imagens próprias + públicas.")

if __name__ == "__main__":
    integrar()
