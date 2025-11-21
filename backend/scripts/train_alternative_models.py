#!/usr/bin/env python3
"""
Script para treinar modelos alternativos (ResNet50, EfficientNet-B5)
caso o modelo principal não atinja acurácia desejada
"""
import sys
import argparse
from pathlib import Path
import json
sys.path.append(str(Path(__file__).parent.parent))

from aplicacao.modelos.treinamento_classificador import TreinadorClassificador, obter_status_treinamento_classificador
from aplicacao.modelos.classificador_resnet import ClassificadorResNet
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_resnet50(dataset_path: str, epochs: int = 30, lr: float = 0.001, batch_size: int = 16,
                   use_focal: bool = False, unfreeze_epoch: int = 5, unfreeze_lr_factor: float = 0.1,
                   resume: str | None = None):
    """Treina modelo ResNet50."""
    logger.info("🚀 Iniciando treinamento com ResNet50...")
    
    # Criar treinador (vai usar ResNet50 se especificado)
    treinador = TreinadorClassificador(
        diretorio_dataset=dataset_path,
        modelo_base='resnet50'
    )
    
    # Treinar
    resultado = treinador.treinar(
        epocas=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        diretorio_saida="/Volumes/Z Slim/modelos_salvos/classificador_resnet50",
        use_focal=use_focal,
        unfreeze_epoch=unfreeze_epoch,
        unfreeze_lr_factor=unfreeze_lr_factor,
        resume_weights_path=resume
    )
    
    logger.info("✅ Treinamento ResNet50 concluído!")
    return resultado

def train_efficientnet_b5(dataset_path: str, epochs: int = 30, lr: float = 0.001, batch_size: int = 8,
                          use_focal: bool = False, unfreeze_epoch: int = 5, unfreeze_lr_factor: float = 0.1,
                          resume: str | None = None):
    """Treina modelo EfficientNet-B5 (maior que B4)."""
    logger.info("🚀 Iniciando treinamento com EfficientNet-B5...")
    
    # Criar treinador com B5
    treinador = TreinadorClassificador(
        diretorio_dataset=dataset_path,
        modelo_base='efficientnet_b5'
    )
    
    # Treinar (batch_size menor devido ao tamanho do modelo)
    resultado = treinador.treinar(
        epocas=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        diretorio_saida="/Volumes/Z Slim/modelos_salvos/classificador_b5",
        use_focal=use_focal,
        unfreeze_epoch=unfreeze_epoch,
        unfreeze_lr_factor=unfreeze_lr_factor,
        resume_weights_path=resume
    )
    
    logger.info("✅ Treinamento EfficientNet-B5 concluído!")
    return resultado

def train_with_more_augmentation(dataset_path: str, epochs: int = 50, lr: float = 0.0005, batch_size: int = 16,
                                 use_focal: bool = False, unfreeze_epoch: int = 5, unfreeze_lr_factor: float = 0.1,
                                 resume: str | None = None):
    """Treina com mais augmentation e épocas."""
    logger.info("🚀 Iniciando treinamento com augmentation aumentada...")
    
    treinador = TreinadorClassificador(
        diretorio_dataset=dataset_path,
        modelo_base='efficientnet_b4'
    )
    
    # Callback para persistir status a cada época (para dashboard em tempo real)
    out_dir = Path("/Volumes/Z Slim/modelos_salvos/classificador_augmented")
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status_treinamento.json"

    def callback_progresso(epoca_atual: int, total_epocas: int, metricas: dict):
        try:
            estado = obter_status_treinamento_classificador()
            # Garantir campos mínimos
            estado.update({
                'treinando': True,
                'epoca_atual': epoca_atual,
                'total_epocas': total_epocas,
                'progresso': int((epoca_atual/ max(1,total_epocas)) * 100),
            })
            if 'tempo_restante_sec' not in estado and 'tempo_restante_seg' in estado:
                estado['tempo_restante_sec'] = estado.get('tempo_restante_seg')
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(estado, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # Treinar por mais épocas com LR menor
    resultado = treinador.treinar(
        epocas=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        diretorio_saida=str(out_dir),
        use_focal=use_focal,
        unfreeze_epoch=unfreeze_epoch,
        unfreeze_lr_factor=unfreeze_lr_factor,
        resume_weights_path=resume,
        callback_progresso=callback_progresso
    )
    
    logger.info("✅ Treinamento com augmentation concluído!")
    return resultado

def main():
    parser = argparse.ArgumentParser(description="Treinar modelos alternativos")
    parser.add_argument('--dataset', type=str, required=True, help='Caminho para o dataset')
    parser.add_argument('--model', type=str, choices=['resnet50', 'efficientnet_b5', 'augmented'], 
                       required=True, help='Modelo a treinar')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--focal', action='store_true', help='Usar FocalLoss')
    parser.add_argument('--unfreeze-epoch', type=int, default=5, help='Época para descongelar todas as camadas')
    parser.add_argument('--unfreeze-lr-factor', type=float, default=0.1, help='Fator para reduzir LR ao descongelar')
    parser.add_argument('--resume', type=str, default=None, help='Caminho para checkpoint/weights para retomar')
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        logger.error(f"Dataset não encontrado: {args.dataset}")
        sys.exit(1)
    
    if args.model == 'resnet50':
        train_resnet50(args.dataset, args.epochs, args.lr, args.batch_size,
                       use_focal=args.focal, unfreeze_epoch=args.unfreeze_epoch,
                       unfreeze_lr_factor=args.unfreeze_lr_factor, resume=args.resume)
    elif args.model == 'efficientnet_b5':
        train_efficientnet_b5(args.dataset, args.epochs, args.lr, args.batch_size,
                              use_focal=args.focal, unfreeze_epoch=args.unfreeze_epoch,
                              unfreeze_lr_factor=args.unfreeze_lr_factor, resume=args.resume)
    elif args.model == 'augmented':
        train_with_more_augmentation(args.dataset, args.epochs, args.lr, args.batch_size,
                                     use_focal=args.focal, unfreeze_epoch=args.unfreeze_epoch,
                                     unfreeze_lr_factor=args.unfreeze_lr_factor, resume=args.resume)

if __name__ == "__main__":
    main()
