"""Script para treinar YOLO11 com o dataset Aerial-Solar-Panels-13 (já baixado).

Execução rápida (15 épocas) para obter métricas reais do detector.

Inclui um monkeypatch simples em cv2.imshow para contornar o fato de que o
ambiente utiliza opencv-python-headless (que não expõe imshow), evitando o
erro de importação dentro da biblioteca Ultralytics.
"""
import sys
import logging
from pathlib import Path

# Monkeypatch de cv2.imshow antes de importar Ultralytics/YOLO
try:
    import cv2  # type: ignore
    if not hasattr(cv2, "imshow"):
        def _noop_imshow(*args, **kwargs):  # pragma: no cover
            return None
        cv2.imshow = _noop_imshow  # type: ignore[attr-defined]
except Exception:
    # Se não conseguir importar cv2, deixamos seguir; o TreinadorDetector lidará com o erro.
    pass

# Adicionar backend ao path
sys.path.insert(0, str(Path(__file__).parent))

from aplicacao.modelos.treinamento_detector import TreinadorDetector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Treina YOLO11 com dataset Aerial-Solar-Panels-13."""
    
    # Caminho do dataset (já tem imagens baixadas!)
    dataset_yaml = r'F:\datasets_publicos\Aerial-Solar-Panels-13\data.yaml'
    
    # Diretório de saída
    diretorio_saida = r'F:\modelos_salvos\detector_yolo11_aerial_rapido'
    
    logger.info("="*80)
    logger.info("TREINAMENTO RÁPIDO YOLO11 - AERIAL SOLAR PANELS")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Saída: {diretorio_saida}")
    logger.info(f"Configuração: 15 épocas, batch=16, imgsz=640, CUDA")
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
    logger.info("RESULTADO DO TREINAMENTO")
    logger.info("="*80)
    
    if resultado['status'] == 'sucesso':
        metricas = resultado.get('metricas_finais', {})
        logger.info(f"✅ Status: SUCESSO")
        logger.info(f"📊 mAP50: {metricas.get('mAP50', 0):.4f}")
        logger.info(f"📊 Precision: {metricas.get('precision', 0):.4f}")
        logger.info(f"📊 Recall: {metricas.get('recall', 0):.4f}")
        logger.info(f"📊 Box Loss: {metricas.get('loss', 0):.4f}")
        logger.info(f"💾 Modelo: {resultado.get('modelo_path', 'N/A')}")
        logger.info(f"📄 Relatório: {diretorio_saida}\\relatorio_treinamento.json")
        logger.info("="*80)
        logger.info("✅ MÉTRICAS SALVAS - PRONTO PARA PREENCHER TCC")
    else:
        logger.error(f"❌ Erro: {resultado.get('erro', 'Desconhecido')}")
    
    return resultado

if __name__ == "__main__":
    try:
        resultado = main()
        sys.exit(0 if resultado.get('status') == 'sucesso' else 1)
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}", exc_info=True)
        sys.exit(1)
