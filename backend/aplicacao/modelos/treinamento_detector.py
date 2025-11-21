"""
Treinamento do Detector YOLO11 para Painéis Solares
Desenvolvido para TCC - Engenharia Mecatrônica
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import json
import time
import logging
from typing import Dict, Callable, Optional

from aplicacao.config import HARDWARE as CONFIG_HARDWARE

logger = logging.getLogger(__name__)

# Estado global do treinamento (para API polling)
estado_treinamento = {
    'treinando': False,
    'epoca_atual': 0,
    'total_epocas': 0,
    'progresso': 0,
    'metricas': {
        'mAP50': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'loss': 0.0
    },
    'tempo_restante_seg': 0,
    'inicio_treinamento': None,
    'erro': None
}

class TreinadorDetector:
    """
    Classe para treinamento do YOLO11 especializado em detecção de painéis solares.
    
    Features:
    - Callback customizado para progress tracking
    - Suporte a MPS (Apple Silicon)
    - Early stopping automático
    - Salvamento automático do melhor modelo
    - Geração de relatórios detalhados
    """
    
    def __init__(self, caminho_dataset_yaml: str, modelo_base: str = 'yolo11n.pt'):
        """
        Inicializa o treinador do detector.
        
        Args:
            caminho_dataset_yaml (str): Caminho para arquivo dataset.yaml
            modelo_base (str): Modelo base ('yolov8n.pt', 'yolov8s.pt', etc.)
        """
        self.caminho_dataset = Path(caminho_dataset_yaml)
        self.modelo_base = modelo_base

        # Seleciona dispositivo com base na prioridade configurada (mps > cuda > cpu por padrão)
        try:
            prioridades = CONFIG_HARDWARE.get("device_prioridade", ["mps", "cuda", "cpu"])
        except Exception:
            prioridades = ["mps", "cuda", "cpu"]

        dispositivo_escolhido = torch.device("cpu")
        for nome in prioridades:
            if nome == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    dispositivo_escolhido = torch.device("mps")
                    break
            elif nome == "cuda":
                if torch.cuda.is_available():
                    dispositivo_escolhido = torch.device("cuda")
                    break
            elif nome == "cpu":
                dispositivo_escolhido = torch.device("cpu")
                break

        self.dispositivo = dispositivo_escolhido.type
        
        # Validar dataset
        if not self.caminho_dataset.exists():
            raise FileNotFoundError(f"Dataset YAML não encontrado: {caminho_dataset_yaml}")
        
        logger.info(f"🎯 Treinador YOLO11 inicializado")
        logger.info(f"   Dataset: {caminho_dataset_yaml}")
        logger.info(f"   Modelo base: {modelo_base}")
        logger.info(f"   Dispositivo: {self.dispositivo}")
    
    def treinar(
        self,
        epocas: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        lr: float = 0.01,
        patience: int = 10,
        save_period: int = 5,
        callback_progresso: Optional[Callable] = None,
        diretorio_saida: str = "modelos_salvos/detector_yolo"
    ) -> Dict:
        """
        Executa treinamento completo do detector.
        
        Args:
            epocas (int): Número de épocas de treinamento
            batch_size (int): Tamanho do batch
            imgsz (int): Tamanho da imagem (quadrada)
            lr (float): Learning rate
            patience (int): Paciência para early stopping
            save_period (int): Período de salvamento (épocas)
            callback_progresso (callable): Callback para atualizar progresso
            diretorio_saida (str): Diretório para salvar modelos
            
        Returns:
            dict: Resultados do treinamento
        """
        
        # Atualizar estado global
        global estado_treinamento
        estado_treinamento.update({
            'treinando': True,
            'epoca_atual': 0,
            'total_epocas': epocas,
            'progresso': 0,
            'inicio_treinamento': time.time(),
            'erro': None
        })
        
        try:
            logger.info("🚀 Iniciando treinamento do detector YOLOv8...")
            
            # Criar diretório de saída
            Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
            
            # Carregar modelo base
            modelo = YOLO(self.modelo_base)
            logger.info(f"✅ Modelo base carregado: {self.modelo_base}")
            
            # Configurar callback customizado
            def on_epoch_end(trainer):
                """Callback ao final de cada época"""
                epoca = trainer.epoch
                total_epocas = trainer.epochs
                
                # Atualizar estado global
                estado_treinamento['epoca_atual'] = epoca
                estado_treinamento['progresso'] = int((epoca / total_epocas) * 100)
                
                # Extrair métricas do treinador
                try:
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        metricas = trainer.metrics
                        
                        # Métricas principais do YOLO
                        estado_treinamento['metricas'].update({
                            'mAP50': float(metricas.get('metrics/mAP50(B)', 0)),
                            'precision': float(metricas.get('metrics/precision(B)', 0)),
                            'recall': float(metricas.get('metrics/recall(B)', 0)),
                            'loss': float(metricas.get('val/box_loss', 0))  # Usar validation loss
                        })
                    
                    # Estimar tempo restante
                    if estado_treinamento['inicio_treinamento']:
                        tempo_decorrido = time.time() - estado_treinamento['inicio_treinamento']
                        if epoca > 0:
                            tempo_por_epoca = tempo_decorrido / epoca
                            epocas_restantes = total_epocas - epoca
                            estado_treinamento['tempo_restante_seg'] = int(tempo_por_epoca * epocas_restantes)
                    
                    # Log de progresso
                    logger.info(f"📊 Época {epoca}/{total_epocas} ({estado_treinamento['progresso']}%)")
                    logger.info(f"   mAP50: {estado_treinamento['metricas']['mAP50']:.3f}")
                    logger.info(f"   Precision: {estado_treinamento['metricas']['precision']:.3f}")
                    logger.info(f"   Recall: {estado_treinamento['metricas']['recall']:.3f}")
                    logger.info(f"   Loss: {estado_treinamento['metricas']['loss']:.3f}")
                    
                    # Chamar callback externo se fornecido
                    if callback_progresso:
                        callback_progresso(epoca, total_epocas, estado_treinamento['metricas'])
                        
                except Exception as e:
                    logger.error(f"❌ Erro no callback: {e}")
            
            # Registrar callback
            modelo.add_callback("on_train_epoch_end", on_epoch_end)
            
            # Iniciar treinamento
            logger.info(f"🏋️ Iniciando treinamento com {epocas} épocas...")
            
            resultados = modelo.train(
                data=str(self.caminho_dataset),
                epochs=epocas,
                imgsz=imgsz,
                batch=batch_size,
                lr0=lr,
                device=self.dispositivo,
                patience=patience,
                save_period=save_period,
                project=diretorio_saida,
                name='treinamento',
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',
                augment=True,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0
            )
            
            # Finalizar estado
            estado_treinamento['treinando'] = False
            estado_treinamento['progresso'] = 100
            
            # Processar resultados
            melhor_modelo_path = None
            for file_path in Path(diretorio_saida).rglob("best.pt"):
                melhor_modelo_path = str(file_path)
                break
            
            if not melhor_modelo_path:
                raise FileNotFoundError("Modelo treinado não encontrado")
            
            # Gerar relatório
            relatorio = self._gerar_relatorio_treinamento(
                resultados, melhor_modelo_path, diretorio_saida
            )
            
            logger.info("✅ Treinamento concluído com sucesso!")
            logger.info(f"🏆 Melhor modelo salvo em: {melhor_modelo_path}")
            
            return {
                'status': 'sucesso',
                'modelo_path': melhor_modelo_path,
                'relatorio': relatorio,
                'metricas_finais': estado_treinamento['metricas'].copy()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            
            # Atualizar estado com erro
            estado_treinamento.update({
                'treinando': False,
                'erro': str(e),
                'progresso': 0
            })
            
            return {
                'status': 'erro',
                'erro': str(e),
                'metricas_finais': estado_treinamento['metricas'].copy()
            }
    
    def _gerar_relatorio_treinamento(self, resultados, modelo_path: str, diretorio_saida: str) -> Dict:
        """
        Gera relatório detalhado do treinamento.
        
        Args:
            resultados: Resultados do treinamento YOLO
            modelo_path (str): Caminho do melhor modelo
            diretorio_saida (str): Diretório de saída
            
        Returns:
            dict: Relatório do treinamento
        """
        try:
            # Estatísticas do treinamento
            relatorio = {
                'resumo': {
                    'status': 'concluido',
                    'modelo_base': self.modelo_base,
                    'modelo_treinado': modelo_path,
                    'dataset': str(self.caminho_dataset),
                    'dispositivo': self.dispositivo,
                    'epocas_treinadas': estado_treinamento['epoca_atual'],
                    'tempo_total_seg': round(time.time() - estado_treinamento['inicio_treinamento'], 2)
                },
                'metricas_finais': estado_treinamento['metricas'].copy(),
                'hiperparametros': {
                    'batch_size': 16,
                    'imgsz': 640,
                    'lr0': 0.01,
                    'optimizer': 'AdamW',
                    'augment': True
                },
                'arquivos_gerados': self._listar_arquivos_saida(diretorio_saida)
            }
            
            # Salvar relatório em JSON
            relatorio_path = Path(diretorio_saida) / 'relatorio_treinamento.json'
            with open(relatorio_path, 'w', encoding='utf-8') as f:
                json.dump(relatorio, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 Relatório salvo: {relatorio_path}")
            return relatorio
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar relatório: {e}")
            return {'erro': str(e)}
    
    def _listar_arquivos_saida(self, diretorio_saida: str) -> list:
        """
        Lista arquivos gerados no treinamento.
        
        Args:
            diretorio_saida (str): Diretório de saída
            
        Returns:
            list: Lista de arquivos
        """
        arquivos = []
        saida_dir = Path(diretorio_saida)
        
        try:
            for file_path in saida_dir.rglob("*"):
                if file_path.is_file():
                    arquivos.append({
                        'nome': file_path.name,
                        'caminho': str(file_path),
                        'tamanho_mb': round(file_path.stat().st_size / (1024*1024), 2),
                        'tipo': 'modelo' if file_path.suffix == '.pt' else 'outro'
                    })
        except Exception as e:
            logger.error(f"❌ Erro ao listar arquivos: {e}")
        
        return arquivos

def obter_status_treinamento() -> Dict:
    """
    Retorna status atual do treinamento (para API polling).
    
    Returns:
        dict: Status do treinamento
    """
    return estado_treinamento.copy()

def resetar_status_treinamento():
    """Reseta o status de treinamento."""
    global estado_treinamento
    estado_treinamento = {
        'treinando': False,
        'epoca_atual': 0,
        'total_epocas': 0,
        'progresso': 0,
        'metricas': {
            'mAP50': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'loss': 0.0
        },
        'tempo_restante_seg': 0,
        'inicio_treinamento': None,
        'erro': None
    }

def iniciar_treinamento_async(
    caminho_dataset_yaml: str,
    epocas: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    lr: float = 0.01
) -> Dict:
    """
    Inicia treinamento em background (async).
    
    Args:
        caminho_dataset_yaml (str): Caminho para dataset.yaml
        epocas (int): Número de épocas
        batch_size (int): Tamanho do batch
        imgsz (int): Tamanho da imagem
        lr (float): Learning rate
        
    Returns:
        dict: Status inicial
    """
    import threading
    
    def treinar_background():
        """Função que roda em background"""
        try:
            treinador = TreinadorDetector(caminho_dataset_yaml)
            resultado = treinador.treinar(
                epocas=epocas,
                batch_size=batch_size,
                imgsz=imgsz,
                lr=lr
            )
            logger.info("✅ Treinamento background concluído")
        except Exception as e:
            logger.error(f"❌ Erro no treinamento background: {e}")
            estado_treinamento['erro'] = str(e)
            estado_treinamento['treinando'] = False
    
    # Resetar estado
    resetar_status_treinamento()
    
    # Iniciar thread
    thread = threading.Thread(target=treinar_background, daemon=True)
    thread.start()
    
    return {
        'status': 'iniciado',
        'mensagem': 'Treinamento iniciado em background',
        'detalhes': {
            'dataset': caminho_dataset_yaml,
            'epocas': epocas,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'lr': lr
        }
    }

# Função fábrica
def criar_treinador(caminho_dataset_yaml: str, modelo_base: str = 'yolov8n.pt'):
    """
    Função fábrica para criar treinador.
    
    Args:
        caminho_dataset_yaml (str): Caminho para dataset.yaml
        modelo_base (str): Modelo base
        
    Returns:
        TreinadorDetector: Instância do treinador
    """
    return TreinadorDetector(caminho_dataset_yaml, modelo_base)

# Teste rápido
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Teste do Treinador YOLOv8")
    print("Para usar em produção:")
    print("1. treinador = TreinadorDetector('dados/plantas_completas/dataset.yaml')")
    print("2. resultado = treinador.treinar(epocas=50)")
    
    print("✅ Treinador importado com sucesso!")
