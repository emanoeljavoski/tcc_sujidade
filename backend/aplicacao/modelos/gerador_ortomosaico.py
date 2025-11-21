"""
Gerador de ortomosaico usando OpenStitching para imagens de drone.
Otimizado para painéis solares com processamento em lotes e eliminação de áreas pretas.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from stitching import AffineStitcher
import gc

logger = logging.getLogger(__name__)


class GeradorOrtomosaico:
    """Gera ortomosaicos a partir de imagens de drone usando OpenStitching.

    Resolve problemas de:
    - Áreas pretas no resultado
    - Falhas com muitas imagens (>10)
    - Baixa qualidade de blending
    """

    def __init__(
        self,
        detector: str = "sift",
        confidence_threshold: float = 0.2,
        medium_megapix: float = 0.4,
        low_megapix: float = 0.1,
        batch_size: int = 20,
    ) -> None:
        """Inicializa o gerador de ortomosaico.

        Args:
            detector: Tipo de detector ("sift", "orb", "akaze")
            confidence_threshold: Threshold de confiança para matching (0.1-1.0)
            medium_megapix: Resolução para detecção de features (0.3-0.6 MP)
            low_megapix: Resolução para encontrar costuras
            batch_size: Número de imagens por lote (ajustar conforme RAM)
        """
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.medium_megapix = medium_megapix
        self.low_megapix = low_megapix
        self.batch_size = batch_size

        logger.info(
            "GeradorOrtomosaico inicializado: detector=%s, batch_size=%d",
            detector,
            batch_size,
        )

    def _criar_stitcher(self) -> AffineStitcher:
        """Cria instância de stitcher com configuração otimizada."""
        return AffineStitcher(
            detector=self.detector,
            confidence_threshold=self.confidence_threshold,
            medium_megapix=self.medium_megapix,
            low_megapix=self.low_megapix,
            matcher_type="affine",
            estimator="affine",
            crop=True,  # Remove bordas pretas automaticamente
        )

    def _preprocessar_imagem(self, imagem: np.ndarray) -> np.ndarray:
        """Pré-processa imagem para melhorar qualidade do stitching."""
        # Reduzir ruído (ajuda matching)
        img = cv2.fastNlMeansDenoisingColored(imagem, None, 5, 5, 7, 21)

        # Melhorar contraste (CLAHE) para destacar bordas dos painéis
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

        return img

    def _remover_bordas_pretas(self, imagem: np.ndarray) -> np.ndarray:
        """Remove bordas pretas remanescentes após stitching."""
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Encontrar maior contorno (região não-preta)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            return imagem[y : y + h, x : x + w]

        return imagem

    def gerar_ortomosaico(
        self,
        caminhos_imagens: List[str],
        caminho_saida: str,
        preprocessar: bool = True,
        remover_bordas: bool = True,
    ) -> Dict[str, Any]:
        """Gera ortomosaico a partir de lista de imagens."""
        num_imagens = len(caminhos_imagens)
        logger.info("Iniciando geração de ortomosaico com %d imagens", num_imagens)

        try:
            # Processar em lotes se necessário
            if num_imagens <= self.batch_size:
                ortomosaico = self._processar_lote(caminhos_imagens, preprocessar)
            else:
                ortomosaico = self._processar_hierarquico(caminhos_imagens, preprocessar)

            # Remover bordas pretas se solicitado
            if remover_bordas:
                logger.info("Removendo bordas pretas do ortomosaico")
                ortomosaico = self._remover_bordas_pretas(ortomosaico)

            # Salvar resultado
            destino = Path(caminho_saida)
            destino.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destino), ortomosaico)

            altura, largura = ortomosaico.shape[:2]
            logger.info(
                "Ortomosaico gerado com sucesso (%dx%d) salvo em %s",
                largura,
                altura,
                destino,
            )

            return {
                "sucesso": True,
                "caminho_ortomosaico": str(destino),
                "num_imagens": num_imagens,
                "tamanho_final": (altura, largura),
            }

        except Exception as e:
            logger.error("Erro ao gerar ortomosaico: %s", e)
            return {
                "sucesso": False,
                "erro": str(e),
                "num_imagens": num_imagens,
            }

    def _processar_lote(self, caminhos: List[str], preprocessar: bool) -> np.ndarray:
        """Processa um único lote de imagens."""
        logger.info("Processando lote de %d imagens", len(caminhos))

        imagens = []
        for caminho in caminhos:
            img = cv2.imread(caminho)
            if img is None:
                logger.warning("Não foi possível carregar imagem: %s", caminho)
                continue

            if preprocessar:
                img = self._preprocessar_imagem(img)

            imagens.append(img)

        if len(imagens) < 2:
            raise ValueError(f"Imagens insuficientes para stitching: {len(imagens)}")

        stitcher = self._criar_stitcher()
        panorama = stitcher.stitch(imagens)

        del imagens
        gc.collect()

        return panorama

    def _processar_hierarquico(self, caminhos: List[str], preprocessar: bool) -> np.ndarray:
        """Processa grande quantidade de imagens hierarquicamente."""
        logger.info(
            "Processamento hierárquico: %d imagens em lotes de %d",
            len(caminhos),
            self.batch_size,
        )

        resultados_lotes: List[np.ndarray] = []

        for i in range(0, len(caminhos), self.batch_size):
            lote = caminhos[i : i + self.batch_size]
            logger.info(
                "  Lote %d/%d",
                i // self.batch_size + 1,
                (len(caminhos) - 1) // self.batch_size + 1,
            )

            try:
                resultado = self._processar_lote(lote, preprocessar)
                resultados_lotes.append(resultado)
            except Exception as e:
                logger.warning("  Lote falhou: %s", e)
                continue

            gc.collect()

        if len(resultados_lotes) == 0:
            raise ValueError("Nenhum lote foi processado com sucesso")

        if len(resultados_lotes) == 1:
            return resultados_lotes[0]

        logger.info("Combinando %d lotes em ortomosaico final", len(resultados_lotes))
        stitcher = self._criar_stitcher()
        panorama_final = stitcher.stitch(resultados_lotes)

        del resultados_lotes
        gc.collect()

        return panorama_final


def gerar_ortomosaico_rapido(
    caminhos_imagens: List[str], caminho_saida: str, batch_size: int = 20
) -> Dict[str, Any]:
    """Função simplificada para gerar ortomosaico rapidamente."""
    gerador = GeradorOrtomosaico(batch_size=batch_size)
    return gerador.gerar_ortomosaico(caminhos_imagens, caminho_saida)
