"""Script rápido para treinar YOLO11 com o dataset Roboflow Solar PV Maintenance Combined.

Desenvolvido para obter métricas reais do detector em tempo reduzido.
"""
import sys
import logging
from pathlib import Path

# Adicionar backend ao path
sys.path.insert(0, str(Path(__file__).parent))

from aplicacao.modelos.treinamento_detector import TreinadorDetector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Treina YOLO11 com configuração rápida para obter métricas."""
    
    # Caminho do dataset Roboflow (já baixado e pronto)
    dataset_yaml = r'F:\datasets_publicos_rgb\aereos_drone\solar_pv_maintenance_combined\data.yaml'
    
    # Diretório de saída
    diretorio_saida = r'F:\modelos_salvos\detector_yolo11_roboflow'
    
    logger.info("="*80)
    logger.info("TREINAMENTO RÁPIDO YOLO11 - DATASET ROBOFLOW")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Saída: {diretorio_saida}")
    logger.info(f"Configuração: 15 épocas, batch=16, imgsz=640")
    logger.info("="*80)
    
    # Criar treinador
    treinador = TreinadorDetector(
        caminho_dataset_yaml=dataset_yaml,
        modelo_base='yolo11n.pt'
    )
    
    # Treinar com configuração rápida
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
        logger.info(f"Modelo salvo em: {resultado.get('modelo_path', 'N/A')}")
        logger.info(f"Relatório: {diretorio_saida}/relatorio_treinamento.json")
        logger.info("="*80)
    else:
        logger.error(f"Erro: {resultado.get('erro', 'Erro desconhecido')}")
    
    return resultado

if __name__ == "__main__":
    try:
        resultado = main()
        sys.exit(0 if resultado['status'] == 'sucesso' else 1)
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)
