"""
Inicialização do pacote de modelos
"""
from .detector_modulos import DetectorModulos, criar_detector
from .classificador_sujidade import ClassificadorSujidade, criar_classificador
from .pipeline_completo import PipelineInspecao, criar_pipeline

__all__ = [
    'DetectorModulos',
    'ClassificadorSujidade', 
    'PipelineInspecao',
    'criar_detector',
    'criar_classificador',
    'criar_pipeline'
]
