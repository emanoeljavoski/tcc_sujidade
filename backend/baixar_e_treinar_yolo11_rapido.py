"""
Script para baixar dataset Roboflow pequeno e treinar YOLO11 rapidamente
"""
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def baixar_dataset_roboflow():
    """Baixa dataset pequeno do Roboflow."""
    try:
        from roboflow import Roboflow
        
        logger.info("="*80)
        logger.info("BAIXANDO DATASET ROBOFLOW (AERIAL SOLAR PANELS - ~400 imagens)")
        logger.info("="*80)
        
        # Inicializar Roboflow com API key pública
        rf = Roboflow(api_key="q1tjW7hVYDHUYgwzJbSt")
        
        # Baixar dataset pequeno (Aerial Solar Panels - Brad Dwyer)
        project = rf.workspace("brad-dwyer").project("aerial-solar-panels")
        dataset = project.version(3).download(
            "yolov8",
            location=r"F:\dataset_yolo_treino_rapido"
        )
        
        logger.info("Dataset baixado em: F:\\dataset_yolo_treino_rapido")
        return r"F:\dataset_yolo_treino_rapido\data.yaml"
        
    except ImportError:
        logger.error("Biblioteca 'roboflow' não instalada. Instale com: pip install roboflow")
        return None
    except Exception as e:
        logger.error(f"Erro ao baixar dataset: {e}")
        return None

def treinar_yolo11(dataset_yaml):
    """Treina YOLO11 com configuração rápida."""
    sys.path.insert(0, str(Path(__file__).parent))
    from aplicacao.modelos.treinamento_detector import TreinadorDetector
    
    logger.info("="*80)
    logger.info("TREINAMENTO RÁPIDO YOLO11")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Configuração: 15 épocas, batch=16, imgsz=640")
    logger.info("="*80)
    
    # Diretório de saída
    diretorio_saida = r'F:\modelos_salvos\detector_yolo11_roboflow'
    
    # Criar treinador
    treinador = TreinadorDetector(
        caminho_dataset_yaml=dataset_yaml,
        modelo_base='yolo11n.pt'
    )
    
    # Treinar
    resultado = treinador.treinar(
        epocas=15,
        batch_size=16,
        imgsz=640,
        lr=0.01,
        patience=10,
        save_period=5,
        diretorio_saida=diretorio_saida
    )
    
    # Exibir resultados
    logger.info("="*80)
    logger.info("TREINAMENTO CONCLUÍDO")
    logger.info("="*80)
    
    if resultado['status'] == 'sucesso':
        metricas = resultado.get('metricas_finais', {})
        logger.info(f"Status: {resultado['status']}")
        logger.info(f"mAP50: {metricas.get('mAP50', 0):.4f}")
        logger.info(f"Precision: {metricas.get('precision', 0):.4f}")
        logger.info(f"Recall: {metricas.get('recall', 0):.4f}")
        logger.info(f"Loss: {metricas.get('loss', 0):.4f}")
        logger.info(f"Modelo: {resultado.get('modelo_path', 'N/A')}")
        logger.info(f"Relatório: {diretorio_saida}\\relatorio_treinamento.json")
        logger.info("="*80)
    else:
        logger.error(f"Erro: {resultado.get('erro', 'Desconhecido')}")
    
    return resultado

def main():
    """Pipeline completo: baixar + treinar."""
    # 1. Baixar dataset
    dataset_yaml = baixar_dataset_roboflow()
    if not dataset_yaml:
        logger.error("Falha ao baixar dataset. Abortando.")
        return {'status': 'erro', 'erro': 'Falha no download'}
    
    # 2. Treinar
    resultado = treinar_yolo11(dataset_yaml)
    return resultado

if __name__ == "__main__":
    try:
        resultado = main()
        sys.exit(0 if resultado.get('status') == 'sucesso' else 1)
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)
