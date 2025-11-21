"""Detector com tiling inteligente usando SAHI para processar imagens grandes.
Evita perda de detecções nas bordas através de overlap e NMS."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

logger = logging.getLogger(__name__)


class DetectorSAHI:
    """Detector YOLO11 com tiling automático via SAHI.

    Resolve problemas de:
    - Detecção em imagens muito grandes (8K+)
    - Perda de detecções nas bordas
    - Eficiência de memória GPU/RAM
    """

    def __init__(
        self,
        caminho_modelo: str,
        confidence_threshold: float = 0.4,
        device: str = "cuda:0",
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
    ) -> None:
        """Inicializa detector com SAHI."""
        self.caminho_modelo = caminho_modelo
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

        # Inicializar modelo SAHI
        self.modelo = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=caminho_modelo,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        logger.info(
            "DetectorSAHI inicializado: modelo=%s, tiles=%dx%d, overlap=%.2f",
            Path(caminho_modelo).name,
            slice_width,
            slice_height,
            overlap_width_ratio,
        )

    def detectar(
        self,
        caminho_imagem: str,
        visualizar: bool = False,
        caminho_saida_visual: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detecta objetos em imagem grande usando tiling."""
        import time

        logger.info("Detectando em imagem: %s", Path(caminho_imagem).name)
        inicio = time.time()

        resultado = get_sliced_prediction(
            caminho_imagem,
            self.modelo,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=False,
        )

        deteccoes: List[Dict[str, Any]] = []
        for pred in resultado.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            deteccoes.append(
                {
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "confianca": float(pred.score.value),
                    "classe": pred.category.name,
                    "classe_id": pred.category.id,
                }
            )

        tempo = time.time() - inicio

        img = read_image(caminho_imagem)
        altura, largura = img.shape[:2]

        logger.info(
            "Detectados %d objetos em %.2fs (imagem %dx%d)",
            len(deteccoes),
            tempo,
            largura,
            altura,
        )

        if visualizar:
            if caminho_saida_visual is None:
                caminho_saida_visual = str(
                    Path(caminho_imagem).parent
                    / f"{Path(caminho_imagem).stem}_deteccoes.jpg"
                )

            resultado.export_visuals(export_dir=str(Path(caminho_saida_visual).parent))
            logger.info("Visualização de detecções salva em: %s", caminho_saida_visual)

        return {
            "num_deteccoes": len(deteccoes),
            "deteccoes": deteccoes,
            "tempo_processamento": tempo,
            "tamanho_imagem": (altura, largura),
        }

    def detectar_lote(self, caminhos_imagens: List[str], visualizar: bool = False) -> List[Dict[str, Any]]:
        """Detecta em múltiplas imagens."""
        resultados: List[Dict[str, Any]] = []

        for i, caminho in enumerate(caminhos_imagens):
            logger.info("Imagem %d/%d", i + 1, len(caminhos_imagens))
            resultado = self.detectar(caminho, visualizar=visualizar)
            resultado["caminho_imagem"] = caminho
            resultados.append(resultado)

        total_deteccoes = sum(r["num_deteccoes"] for r in resultados)
        logger.info(
            "Total de %d detecções em %d imagens", total_deteccoes, len(caminhos_imagens)
        )

        return resultados
