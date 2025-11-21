import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging

try:
    # OpenStitching: wrapper Python para o pipeline de stitching do OpenCV
    from stitching import AffineStitcher
    OPENSTITCHING_AVAILABLE = True
except Exception:
    AffineStitcher = None  # type: ignore
    OPENSTITCHING_AVAILABLE = False

logger = logging.getLogger(__name__)


class DroneMosaicing:
    """Geração de ortomosaico simplificado para imagens de drone.

    Pipeline baseado em técnicas clássicas de visão computacional:
    - Detecção de features: SIFT
    - Matching: FLANN (KD-Tree)
    - Estimativa de homografia: RANSAC
    - Warp e blend incremental das imagens

    Requisitos assumidos (compatíveis com o repositório Drone-Images-Mosaicing):
    - Imagens com pelo menos ~70% de overlap entre vizinhas
    - Ordem das imagens representa aproximadamente a sequência do voo
    """

    def __init__(
        self,
        max_resolution: int = 2000,
        match_ratio: float = 0.75,
        ransac_threshold: float = 5.0,
    ) -> None:
        # Desativar OpenCL para evitar erros de memória em GPUs
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

        self.max_resolution = max_resolution
        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold

        # Criar detector SIFT
        self.sift = cv2.SIFT_create(nfeatures=5000)

        # Configurar matcher FLANN para SIFT (descriptors float32)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # OpenStitching (opcional) para cenas planares (telhados, plantas solares)
        self.use_openstitching: bool = OPENSTITCHING_AVAILABLE
        self.affine_stitcher: Optional["AffineStitcher"] = None
        if self.use_openstitching:
            try:
                self.affine_stitcher = AffineStitcher(
                    detector="sift",
                    confidence_threshold=0.2,
                    crop=True,
                )
                logger.info(
                    "DroneMosaicing: OpenStitching (AffineStitcher) inicializado com detector=sift, "
                    "confidence_threshold=0.2, crop=True",
                )
            except Exception as e:
                logger.warning(
                    "DroneMosaicing: falha ao inicializar AffineStitcher (%s). "
                    "OpenStitching será desativado.",
                    e,
                )
                self.use_openstitching = False
                self.affine_stitcher = None

        logger.info(
            "DroneMosaicing inicializado (max_resolution=%d, match_ratio=%.2f, ransac_threshold=%.2f)",
            self.max_resolution,
            self.match_ratio,
            self.ransac_threshold,
        )

    # ===================== API pública =====================

    def gerar_ortomosaico(self, caminhos_imagens: List[str], caminho_saida: str) -> str:
        """Gera um ortomosaico a partir de uma sequência de imagens de drone.

        Args:
            caminhos_imagens: Lista de caminhos de imagens (ordem do voo).
            caminho_saida: Caminho para salvar o ortomosaico final.

        Returns:
            Caminho do arquivo salvo (sempre um caminho válido).
        """
        if not caminhos_imagens:
            raise ValueError("Nenhuma imagem fornecida para gerar ortomosaico (DroneMosaicing)")

        logger.info("DroneMosaicing: iniciando ortomosaico com %d imagens", len(caminhos_imagens))

        # Tentativa 1: usar OpenStitching (AffineStitcher) se disponível
        if self.use_openstitching and self.affine_stitcher is not None:
            try:
                logger.info(
                    "DroneMosaicing: tentando gerar ortomosaico com OpenStitching "
                    "(AffineStitcher, detector=sift)..."
                )
                pano = self.affine_stitcher.stitch(caminhos_imagens)
                if pano is not None:
                    try:
                        h_p, w_p = pano.shape[:2]
                        logger.info(
                            "DroneMosaicing: ortomosaico OpenStitching gerado com tamanho %dx%d",
                            w_p,
                            h_p,
                        )
                    except Exception:
                        logger.info("DroneMosaicing: ortomosaico OpenStitching gerado")
                    return self._salvar_resultado(pano, caminho_saida)
                else:
                    logger.warning(
                        "DroneMosaicing: OpenStitching retornou None; caindo para pipeline clássico SIFT/FLANN."
                    )
            except Exception as e:
                logger.error(
                    "DroneMosaicing: erro ao usar OpenStitching (%s). "
                    "Usando pipeline clássico SIFT/FLANN.",
                    e,
                )

        # Tentativa 2: pipeline clássico com SIFT + FLANN + homografia
        imagens = []
        for p in caminhos_imagens:
            img = cv2.imread(p)
            if img is None:
                logger.warning("DroneMosaicing: imagem inválida ou não encontrada: %s", p)
                continue
            img_resized = self._resize_keep_aspect(img, self.max_resolution)
            imagens.append(img_resized)

        if not imagens:
            raise ValueError("Nenhuma imagem válida carregada para ortomosaico (DroneMosaicing)")

        if len(imagens) == 1:
            # Apenas uma imagem: salvar diretamente
            logger.info("DroneMosaicing: apenas 1 imagem, salvando sem stitching")
            return self._salvar_resultado(imagens[0], caminho_saida)

        # Inicializar mosaico com a primeira imagem
        mosaico = imagens[0].copy()

        # Coordenadas do mosaico em sistema próprio (usaremos warps relativos)
        for i in range(1, len(imagens)):
            img = imagens[i]
            logger.info("DroneMosaicing: processando par %d/%d", i, len(imagens) - 1)

            try:
                mosaico = self._stitch_pair(mosaico, img)
            except Exception as e:
                logger.error(
                    "DroneMosaicing: falha ao fazer stitch do par %d (%s)",
                    i,
                    str(e),
                )
                # Fallback: continuar com o mosaico atual sem incorporar esta imagem
                continue

        logger.info("DroneMosaicing: ortomosaico gerado com sucesso (tamanho: %dx%d)", mosaico.shape[1], mosaico.shape[0])
        return self._salvar_resultado(mosaico, caminho_saida)

    # ===================== Métodos auxiliares =====================

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, max_dim: int) -> np.ndarray:
        h, w = img.shape[:2]
        maior = max(h, w)
        if maior <= max_dim:
            return img
        escala = max_dim / float(maior)
        novo_tam = (int(w * escala), int(h * escala))
        return cv2.resize(img, novo_tam, interpolation=cv2.INTER_AREA)

    def _detect_and_compute(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        if descriptors is None or len(keypoints) == 0:
            logger.warning("DroneMosaicing: nenhuma feature SIFT detectada")
            return None, None
        return np.array(keypoints), descriptors

    def _match_features(self, des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
        # KNN matching com k=2
        matches_knn = self.flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches_knn:
            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)
        logger.info("DroneMosaicing: %d matches bons após ratio test", len(good_matches))
        return good_matches

    def _estimate_homography(
        self,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
    ) -> Optional[np.ndarray]:
        if len(matches) < 4:
            logger.warning("DroneMosaicing: matches insuficientes (%d) para homografia", len(matches))
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        if H is None:
            logger.warning("DroneMosaicing: homografia não encontrada")
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        logger.info("DroneMosaicing: homografia estimada com %d inliers", inliers)
        return H

    def _stitch_pair(self, base: np.ndarray, nova: np.ndarray) -> np.ndarray:
        """Faz o stitch de uma nova imagem ao mosaico base.

        A homografia é estimada mapeando a nova imagem para o sistema de coordenadas do mosaico.
        """
        # Detectar e descrever features
        kp1, des1 = self.sift.detectAndCompute(base, None)
        kp2, des2 = self.sift.detectAndCompute(nova, None)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            logger.warning("DroneMosaicing: features insuficientes em um dos frames; retornando base sem alterações")
            return base

        # Matching + Lowe's ratio
        matches_knn = self.flann.knnMatch(des2, des1, k=2)  # nova -> base
        good_matches = []
        for m, n in matches_knn:
            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            logger.warning("DroneMosaicing: matches bons insuficientes (%d); retornando base", len(good_matches))
            return base

        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        if H is None:
            logger.warning("DroneMosaicing: homografia não encontrada ao unir par; retornando base")
            return base

        # Calcular tamanho do novo mosaico após o warp da nova imagem
        h_base, w_base = base.shape[:2]
        h_nova, w_nova = nova.shape[:2]

        corners_nova = np.float32([[0, 0], [0, h_nova], [w_nova, h_nova], [w_nova, 0]]).reshape(-1, 1, 2)
        corners_nova_transf = cv2.perspectiveTransform(corners_nova, H)

        corners_base = np.float32([[0, 0], [0, h_base], [w_base, h_base], [w_base, 0]]).reshape(-1, 1, 2)

        all_corners = np.concatenate((corners_base, corners_nova_transf), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translacao = [-xmin, -ymin]
        logger.info("DroneMosaicing: nova área do mosaico (%d,%d) - (%d,%d)", xmin, ymin, xmax, ymax)

        # Warp da nova imagem para o sistema do mosaico expandido
        H_translacao = np.array(
            [[1, 0, translacao[0]], [0, 1, translacao[1]], [0, 0, 1]], dtype=np.float64
        )
        tamanho_saida = (xmax - xmin, ymax - ymin)

        mosaico_warp = cv2.warpPerspective(nova, H_translacao @ H, tamanho_saida)

        # Colar o mosaico base na imagem de saída
        mosaico_saida = mosaico_warp.copy()
        mosaico_saida[translacao[1] : translacao[1] + h_base, translacao[0] : translacao[0] + w_base] = base

        return mosaico_saida

    @staticmethod
    def _salvar_resultado(img: np.ndarray, caminho_saida: str) -> str:
        destino = Path(caminho_saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destino), img)
        logger.info("DroneMosaicing: resultado salvo em %s", destino)
        return str(destino)
