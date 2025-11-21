"""
Detector de Módulos Fotovoltaicos usando YOLO11
Desenvolvido para TCC - Engenharia Mecatrônica

UPGRADE: YOLOv8 → YOLO11 (42% mais eficiente, 2-5% mais acurácia)
"""
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)

class DetectorModulos:
    """
    YOLO11 para detecção de módulos fotovoltaicos em plantas completas.
    
    Funcionalidades:
    - Detecção em tempo real
    - Suporte a MPS (Apple Silicon)
    - Carregamento de modelos pré-treinados
    - Exportação de resultados em múltiplos formatos
    - 42% mais eficiente que YOLOv8
    - 2-5% mais acurácia que YOLOv8
    """
    
    def __init__(self, caminho_modelo=None, modelo_size='n'):
        """
        Inicializa o detector de módulos com YOLO11.
        
        Args:
            caminho_modelo (str): Caminho para modelo .pt treinado
            modelo_size (str): Tamanho do modelo ('n'=nano, 's'=small)
        """
        # Priorizar CUDA no Dell (GPU NVIDIA), depois MPS (Mac), senão CPU
        if torch.cuda.is_available():
            self.dispositivo = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.dispositivo = "mps"
        else:
            self.dispositivo = "cpu"
        self.modelo_size = modelo_size
        logger.info(f"🔧 Inicializando detector YOLO11{modelo_size} no dispositivo: {self.dispositivo}")
        
        # Carregar modelo
        if caminho_modelo and Path(caminho_modelo).exists():
            self.modelo = YOLO(caminho_modelo)
            logger.info(f"✅ Modelo YOLO11 carregado: {caminho_modelo}")
        else:
            # Inicia com YOLO11n (nano - rápido) pré-treinado em COCO
            if modelo_size == 'n':
                self.modelo = YOLO('yolo11n.pt')
                logger.info("📥 Usando YOLO11n pré-treinado (COCO) - Ultra rápido")
            elif modelo_size == 's':
                self.modelo = YOLO('yolo11s.pt')
                logger.info("📥 Usando YOLO11s pré-treinado (COCO) - Melhor acurácia")
            else:
                self.modelo = YOLO('yolo11n.pt')
                logger.info("📥 Usando YOLO11n pré-treinado (COCO) - Padrão")
        
        # Métricas do modelo
        self.classe_alvo = None  # Será definido durante o treinamento
        self.num_classes = 1     # Apenas 1 classe: módulo fotovoltaico
        
    def detectar(
        self,
        imagem_path: str,
        confianca_min: float = 0.5,
        imgsz: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detecta módulos fotovoltaicos na imagem.

        Args:
            imagem_path: Caminho da imagem para detecção
            confianca_min: Confiança mínima para considerar detecção
            imgsz: Tamanho da imagem para inferência (None = cálculo automático)

        Returns:
            Lista de detecções com bounding boxes e confianças
        """
        try:
            # Definir parâmetros de inferência
            conf_usar = confianca_min

            if imgsz is not None:
                imgsz_usar = imgsz
            else:
                imgsz_usar = 640
                try:
                    imagem = cv2.imread(imagem_path)
                    if imagem is not None:
                        h, w = imagem.shape[:2]
                        # Para imagens muito grandes, usar resolução maior e confiança um pouco menor
                        if h > 4000 or w > 4000:
                            imgsz_usar = 1280
                            conf_usar = min(confianca_min, 0.35)
                except Exception:
                    # Em caso de falha ao ler a imagem, manter parâmetros padrão
                    pass

            # Executar inferência
            resultados = self.modelo.predict(
                source=imagem_path,
                conf=conf_usar,
                device=self.dispositivo,
                verbose=False,
                save=False,
                imgsz=imgsz_usar,
            )
            
            # Processar resultados
            deteccoes = []
            for r in resultados:
                for box in r.boxes:
                    # Extrair bounding box [x1, y1, x2, y2]
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    confianca = float(box.conf[0])
                    
                    # Apenas se for uma classe relevante
                    if self.classe_alvo is None or int(box.cls[0]) == self.classe_alvo:
                        deteccoes.append({
                            'bbox': bbox,
                            'confianca': confianca,
                            'classe': int(box.cls[0]) if self.classe_alvo is None else self.classe_alvo
                        })
            
            logger.info(
                f"Detectados {len(deteccoes)} módulos em {Path(imagem_path).name} "
                f"(imgsz={imgsz_usar}, conf_min={conf_usar})"
            )
            return deteccoes
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção: {e}")
            return []
    
    def detectar_batch(self, imagens_paths: list, confianca_min: float = 0.5, imgsz: int = 640):
        """
        Detecta módulos em múltiplas imagens (batch processing).
        
        Args:
            imagens_paths (list): Lista de caminhos das imagens
            confianca_min (float): Confiança mínima
            
        Returns:
            dict: Resultados por imagem
        """
        resultados = {}
        
        try:
            # Batch inference
            batch_results = self.modelo.predict(
                source=imagens_paths,
                conf=confianca_min,
                device=self.dispositivo,
                verbose=False,
                save=False,
                imgsz=imgsz,
            )
            
            # Processar cada resultado
            for i, r in enumerate(batch_results):
                imagem_path = imagens_paths[i]
                deteccoes = []
                
                for box in r.boxes:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    confianca = float(box.conf[0])
                    
                    if self.classe_alvo is None or int(box.cls[0]) == self.classe_alvo:
                        deteccoes.append({
                            'bbox': bbox,
                            'confianca': confianca,
                            'classe': int(box.cls[0]) if self.classe_alvo is None else self.classe_alvo
                        })
                
                resultados[imagem_path] = deteccoes
                
            logger.info(f"Batch processado: {len(imagens_paths)} imagens (imgsz={imgsz}, conf_min={confianca_min})")
            return resultados
            
        except Exception as e:
            logger.error(f"❌ Erro no batch processing: {e}")
            return {path: [] for path in imagens_paths}
    
    def recortar_modulos(self, imagem_path: str, deteccoes: list, salvar_dir: str = None):
        """
        Recorta os módulos detectados da imagem original.
        
        Args:
            imagem_path (str): Caminho da imagem original
            deteccoes (list): Lista de detecções do método detectar()
            salvar_dir (str): Diretório para salvar recortes
            
        Returns:
            list: Lista de imagens recortadas (PIL Image)
        """
        import cv2
        from PIL import Image
        
        try:
            # Carregar imagem
            imagem = cv2.imread(imagem_path)
            if imagem is None:
                raise ValueError(f"Imagem não encontrada: {imagem_path}")
            
            recortes = []
            
            # Criar diretório de salvamento
            if salvar_dir:
                Path(salvar_dir).mkdir(parents=True, exist_ok=True)
            
            src_stem = Path(imagem_path).stem
            for i, det in enumerate(deteccoes):
                # Extrair bounding box
                x1, y1, x2, y2 = map(int, det['bbox'])
                
                # Recortar módulo
                modulo_recortado = imagem[y1:y2, x1:x2]
                
                # Converter para PIL
                modulo_pil = Image.fromarray(cv2.cvtColor(modulo_recortado, cv2.COLOR_BGR2RGB))
                recortes.append(modulo_pil)
                
                # Salvar se diretório especificado
                if salvar_dir:
                    base = Path(salvar_dir) / f"{src_stem}_modulo_{i:03d}.jpg"
                    caminho_salvar = base
                    # Evitar sobrescrever
                    if caminho_salvar.exists():
                        contador = 1
                        while True:
                            cand = Path(salvar_dir) / f"{src_stem}_modulo_{i:03d}_{contador}.jpg"
                            if not cand.exists():
                                caminho_salvar = cand
                                break
                            contador += 1
                    modulo_pil.save(caminho_salvar)
            
            logger.info(f"✂️ Recortados {len(recortes)} módulos")
            return recortes
            
        except Exception as e:
            logger.error(f"❌ Erro ao recortar módulos: {e}")
            return []
    
    def desenhar_deteccoes(self, imagem_path: str, deteccoes: list, salvar_path: str = None):
        """
        Desenha bounding boxes na imagem.
        
        Args:
            imagem_path (str): Caminho da imagem
            deteccoes (list): Lista de detecções
            salvar_path (str): Caminho para salvar imagem anotada
            
        Returns:
            str: Imagem anotada em base64 ou caminho do arquivo
        """
        import cv2
        import base64
        
        try:
            # Carregar imagem
            imagem = cv2.imread(imagem_path)
            if imagem is None:
                raise ValueError(f"Imagem não encontrada: {imagem_path}")
            
            # Desenhar cada bounding box
            for det in deteccoes:
                x1, y1, x2, y2 = map(int, det['bbox'])
                confianca = det['confianca']
                
                # Cor verde para módulos fotovoltaicos
                cor = (0, 255, 0)
                
                # Desenhar retângulo
                cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, 3)
                
                # Label com confiança
                label = f"MODULO {confianca*100:.1f}%"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Fundo do label
                cv2.rectangle(imagem, (x1, y1-label_h-10), (x1+label_w, y1), cor, -1)
                
                # Texto do label
                cv2.putText(imagem, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Salvar ou converter para base64
            if salvar_path:
                cv2.imwrite(salvar_path, imagem)
                logger.info(f"🖼️ Imagem anotada salva: {salvar_path}")
                return salvar_path
            else:
                # Converter para base64
                _, buffer = cv2.imencode('.jpg', imagem)
                imagem_base64 = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/jpeg;base64,{imagem_base64}"
                
        except Exception as e:
            logger.error(f"❌ Erro ao desenhar detecções: {e}")
            return None
    
    def obter_informacoes_modelo(self):
        """
        Retorna informações sobre o modelo carregado.
        
        Returns:
            dict: Informações do modelo
        """
        try:
            info = {
                'modelo': str(type(self.modelo.model).__name__),
                'dispositivo': self.dispositivo,
                'num_classes': self.num_classes,
                'input_size': self.modelo.model.args.get('imgsz', 640),
                'pretrained': self.classe_alvo is not None,
                'parameters': sum(p.numel() for p in self.modelo.model.parameters()),
            }
            
            if self.classe_alvo is not None:
                info['classe_alvo'] = self.classe_alvo
                info['status'] = 'Treinado para painéis solares'
            else:
                info['status'] = 'YOLOv8n pré-treinado (COCO)'
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter informações: {e}")
            return {}
    
    def validar_imagem(self, imagem_path: str):
        """
        Valida se a imagem é adequada para detecção.
        
        Args:
            imagem_path (str): Caminho da imagem
            
        Returns:
            dict: Resultado da validação
        """
        try:
            import cv2
            
            # Verificar se arquivo existe
            if not Path(imagem_path).exists():
                return {'valida': False, 'erro': 'Arquivo não encontrado'}
            
            # Tentar ler imagem
            imagem = cv2.imread(imagem_path)
            if imagem is None:
                return {'valida': False, 'erro': 'Formato de imagem inválido'}
            
            h, w = imagem.shape[:2]
            
            # Verificar tamanho mínimo
            if w < 320 or h < 320:
                return {
                    'valida': False, 
                    'erro': f'Imagem muito pequena: {w}x{h} (mínimo: 320x320)'
                }
            
            # Verificar proporção (não muito alongada)
            aspect_ratio = w / h
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                return {
                    'valida': False, 
                    'erro': f'Proporção muito extrema: {aspect_ratio:.2f}'
                }
            
            return {
                'valida': True,
                'dimensoes': (w, h),
                'aspect_ratio': aspect_ratio,
                'tamanho_mb': Path(imagem_path).stat().st_size / (1024*1024)
            }
            
        except Exception as e:
            return {'valida': False, 'erro': str(e)}

# Função utilitária para criar detector
def criar_detector(caminho_modelo=None):
    """
    Função fábrica para criar instância do detector.
    
    Args:
        caminho_modelo (str): Caminho para modelo treinado
        
    Returns:
        DetectorModulos: Instância do detector
    """
    return DetectorModulos(caminho_modelo)

# Teste rápido
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Criar detector
    detector = criar_detector()
    
    # Mostrar informações
    info = detector.obter_informacoes_modelo()
    print("📊 Informações do Detector:")
    for k, v in info.items():
        print(f"   {k}: {v}")
    
    print("✅ Detector inicializado com sucesso!")
