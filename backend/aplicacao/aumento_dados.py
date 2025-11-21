"""
Módulo de Data Augmentation Agressivo para Painéis Solares
Sistema de Inspeção de Painéis Solares - TCC Engenharia Mecatrônica

Objetivo: Expandir 100 imagens → 1500+ variações efetivas
Estratégias específicas para imagens de drones de painéis solares
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
from PIL import Image
import os
from typing import List, Tuple, Dict, Any
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AumentoDadosPainelSolar:
    """
    Pipeline de aumento especializado para painéis solares
    capturados por drones em diferentes condições climáticas
    """
    
    def __init__(self, modo='treino', n_aumentos=15):
        self.modo = modo
        self.n_aumentos = n_aumentos
        self.transformacao = self._obter_pipeline_aumento()
        
    def _obter_pipeline_aumento(self):
        """
        Pipeline COMPLETO com transformações específicas para painéis solares:
        - Geométricas: flips, rotação (±25°), shift-scale-rotate
        - Iluminação: CLAHE, brightness/contrast (±30%), RandomGamma
        - Condições drone: RandomSunFlare, RandomShadow, RandomRain, RandomFog
        - Qualidade: GaussianBlur, MotionBlur (drone em movimento), GaussNoise
        - Simulação sujidade: CoarseDropout (simula dejetos)
        """
        
        if self.modo == 'treino':
            # Pipeline AGRESSIVO para treinamento
            return A.Compose([
                # Transformações Geométricas
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=25, p=0.7, border_mode=cv2.BORDER_REFLECT),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.8,
                    border_mode=cv2.BORDER_REFLECT
                ),
                
                # Transformações de Iluminação
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.8
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.RandomSunFlare(
                    src_radius=100,
                    num_flare_circles_lower=0,
                    num_flare_circles_upper=6,
                    p=0.3
                ),
                
                # Condições Climáticas (Drone)
                A.RandomShadow(
                    shadow_dimension=5,
                    p=0.4
                ),
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=1,
                    brightness_coefficient=0.7,
                    rain_type="drizzle",
                    p=0.3
                ),
                A.RandomFog(
                    fog_coef_lower=0.1,
                    fog_coef_upper=0.3,
                    alpha_coef=0.08,
                    p=0.2
                ),
                
                # Qualidade de Imagem (Drone em movimento)
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),
                A.MotionBlur(
                    blur_limit=7,
                    allow_shifted=True,
                    p=0.3
                ),
                A.GaussNoise(
                    var_limit=(10.0, 50.0),
                    mean=0,
                    p=0.4
                ),
                
                # Simulação de Sujidade e Defeitos
                A.CoarseDropout(
                    max_holes=8,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=4,
                    min_width=4,
                    fill_value=0,
                    p=0.5
                ),
                
                # Transformações de Cor
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.6
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=0.4
                ),
                
                # Perspectiva (ângulos de drone)
                A.Perspective(
                    scale=(0.05, 0.1),
                    p=0.3
                ),
                
                # Normalização final
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Pipeline leve para validação/teste
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __call__(self, image):
        """Aplica transformação na imagem"""
        return self.transformacao(image=image)


class SolarPanelAugmentation(AumentoDadosPainelSolar):
    """Alias de compatibilidade para AumentoDadosPainelSolar.

    Aceita a assinatura em inglês (mode, n_augmentations) usada em alguns
    pontos do código e a converte para (modo, n_aumentos) esperada pela
    classe base AumentoDadosPainelSolar.
    """

    def __init__(self, mode: str = 'train', n_augmentations: int = 15, **kwargs):
        # Converter parâmetros para a convenção da classe base
        if mode in ['train', 'treino']:
            modo = 'treino'
        else:
            modo = 'val'
        n_aumentos = n_augmentations

        super().__init__(modo=modo, n_aumentos=n_aumentos)

class AugmentedSolarDataset(Dataset):
    """
    Dataset que gera N augmentações por imagem ONLINE durante treinamento
    Cada época deve ver versões DIFERENTES da mesma imagem
    """
    
    def __init__(
        self, 
        images: List[np.ndarray], 
        labels: List[int], 
        image_paths: List[str] = None,
        n_augmentations: int = 15,
        mode: str = 'train',
        seed: int = 42
    ):
        """
        Args:
            images: Lista de arrays numpy (H, W, C)
            labels: Lista de labels inteiros (0-3)
            image_paths: Lista opcional de caminhos das imagens
            n_augmentations: Número de augmentações por imagem original
            mode: 'train' ou 'val'
            seed: Semente para reprodutibilidade
        """
        self.images = images
        self.labels = labels
        self.image_paths = image_paths or [f"image_{i}" for i in range(len(images))]
        self.n_augmentations = n_augmentations
        self.mode = mode
        self.seed = seed
        
        # Validação
        assert len(images) == len(labels), "Images e labels devem ter mesmo tamanho"
        assert len(images) > 0, "Dataset não pode estar vazio"
        
        # Se for treinamento, expande dataset com augmentações
        if mode == 'train':
            self.expanded_images = []
            self.expanded_labels = []
            self.expanded_paths = []
            
            logger.info(f"Expandindo dataset: {len(images)} → {len(images) * n_augmentations} imagens")
            
            for idx, (img, label, path) in enumerate(zip(images, labels, image_paths)):
                for aug_idx in range(n_augmentations):
                    self.expanded_images.append(img)
                    self.expanded_labels.append(label)
                    self.expanded_paths.append(f"{path}_aug_{aug_idx}")
            
            logger.info(f"Dataset expandido para {len(self.expanded_images)} amostras")
        else:
            # Para validação, usa imagens originais
            self.expanded_images = images
            self.expanded_labels = labels
            self.expanded_paths = image_paths
        
        # Configura augmentation
        self.augmentation = SolarPanelAugmentation(mode=mode, n_augmentations=n_augmentations)
        
        # Configura seed para reprodutibilidade
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def __len__(self):
        return len(self.expanded_images)
    
    def __getitem__(self, idx):
        """
        Retorna imagem augmentada e label
        Cada chamada gera uma augmentação DIFERENTE (online)
        """
        image = self.expanded_images[idx].copy()
        label = self.expanded_labels[idx]
        path = self.expanded_paths[idx]
        
        # Aplica augmentation online
        try:
            if self.mode == 'train':
                # Para treinamento, aplica augmentação aleatória
                augmented = self.augmentation(image=image)
                image_tensor = augmented['image']
            else:
                # Para validação, apenas normaliza
                augmented = self.augmentation(image=image)
                image_tensor = augmented['image']
                
        except Exception as e:
            logger.error(f"Erro na augmentação da imagem {path}: {e}")
            # Fallback: apenas normalização
            augmented = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])(image=image)
            image_tensor = augmented['image']
        
        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'path': path
        }
    
    def get_class_distribution(self):
        """Retorna distribuição das classes no dataset"""
        unique, counts = np.unique(self.expanded_labels, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        total = sum(counts)
        
        logger.info("Distribuição das classes:")
        class_names = ['Limpo', 'Pouco Sujo', 'Sujo', 'Muito Sujo']
        for class_id, count in distribution.items():
            percentage = (count / total) * 100
            logger.info(f"  Classe {class_id} ({class_names[class_id]}): {count} ({percentage:.1f}%)")
        
        return distribution
    
    def get_statistics(self):
        """Retorna estatísticas do dataset"""
        stats = {
            'total_samples': len(self),
            'original_images': len(self.images),
            'augmentation_factor': self.n_augmentations if self.mode == 'train' else 1,
            'class_distribution': self.get_class_distribution(),
            'image_shape': self.images[0].shape if len(self.images) > 0 else None
        }
        return stats


class DatasetSolarAumentado(AugmentedSolarDataset):
    """Alias de compatibilidade para AugmentedSolarDataset.

    Mantém suporte a imports existentes de DatasetSolarAumentado usados em
    funções como criar_datasets_aumentados e treinar_modelo_final.py,
    aceitando parâmetros em português e repassando para o construtor
    original (AugmentedSolarDataset).
    """

    def __init__(
        self,
        imagens: List[np.ndarray],
        labels: List[int],
        caminhos_imagens: List[str] = None,
        n_aumentos: int = 15,
        modo: str = 'treino',
        seed: int = 42
    ):
        # Mapear nomes de parâmetros para o construtor base
        mode = 'train' if modo == 'treino' else 'val'
        super().__init__(
            images=imagens,
            labels=labels,
            image_paths=caminhos_imagens,
            n_augmentations=n_aumentos,
            mode=mode,
            seed=seed
        )

def criar_datasets_aumentados(
    imagens_treino: List[np.ndarray],
    labels_treino: List[int],
    caminhos_imagens_treino: List[str],
    imagens_val: List[np.ndarray] = None,
    labels_val: List[int] = None,
    caminhos_imagens_val: List[str] = None,
    n_aumentos: int = 15,
    batch_size: int = 16,
    num_workers: int = 4
):
    """
    Cria DataLoaders com aumento agressivo para treinamento
    
    Args:
        imagens_treino: Lista de imagens de treinamento
        labels_treino: Labels correspondentes
        caminhos_imagens_treino: Caminhos das imagens (para logging)
        imagens_val: Imagens de validação (opcional)
        labels_val: Labels de validação (opcional)
        caminhos_imagens_val: Caminhos das imagens de validação
        n_aumentos: Número de aumentos por imagem original
        batch_size: Tamanho do batch
        num_workers: Número de workers para DataLoader
        
    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders de treino e validação
    """
    
    logger.info(f" Criando datasets com aumento {n_aumentos}x")
    logger.info(f" Dataset original: {len(imagens_treino)} → {len(imagens_treino) * n_aumentos} amostras")
    
    # Dataset de treinamento com aumento
    dataset_treino = DatasetSolarAumentado(
        imagens=imagens_treino,
        labels=labels_treino,
        caminhos_imagens=caminhos_imagens_treino,
        n_aumentos=n_aumentos,
        modo='treino'
    )
    
    loader_treino = torch.utils.data.DataLoader(
        dataset_treino,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Dataset de validação (sem aumento ou aumento mínimo)
    loader_val = None
    if imagens_val is not None:
        dataset_val = DatasetSolarAumentado(
            imagens=imagens_val,
            labels=labels_val,
            caminhos_imagens=caminhos_imagens_val or [f"val_{i}" for i in range(len(imagens_val))],
            n_aumentos=1,  # Sem aumento para validação
            modo='val'
        )
        
        loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False
        )
    
    logger.info(f"DataLoaders criados:")
    logger.info(f"  Treinamento: {len(dataset_treino)} amostras em {len(loader_treino)} batches")
    if loader_val:
        logger.info(f"  Validação: {len(dataset_val)} amostras em {len(loader_val)} batches")
    
    return loader_treino, loader_val

def visualizar_aumentos(
    imagem: np.ndarray,
    pipeline_aumento: AumentoDadosPainelSolar,
    n_amostras: int = 5,
    caminho_salvar: str = None
):
    """
    Visualiza exemplos de aumentos para debug
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, n_amostras + 1, figsize=(20, 4))
    
    # Imagem original
    axes[0].imshow(imagem)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Aumentos
    for i in range(n_amostras):
        aumentada = pipeline_aumento(image=imagem)
        imagem_aumentada = aumentada['image'].permute(1, 2, 0).numpy()
        # Desnormalizar para visualização
        imagem_aumentada = imagem_aumentada * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        imagem_aumentada = np.clip(imagem_aumentada, 0, 1)
        
        axes[i + 1].imshow(imagem_aumentada)
        axes[i + 1].set_title(f'Aumento {i+1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=150, bbox_inches='tight')
        logger.info(f"Aumentos salvas em: {caminho_salvar}")
        logger.info(f"Augmentações salvas em: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Teste do módulo
    logger.info("Testando módulo de augmentation...")
    
    # Criar imagem de teste
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_label = 1
    
    # Testar augmentation
    augmentation = SolarPanelAugmentation(mode='train', n_augmentations=15)
    
    # Criar dataset de teste
    dataset = AugmentedSolarDataset(
        images=[test_image],
        labels=[test_label],
        image_paths=['test_image'],
        n_augmentations=5,
        mode='train'
    )
    
    logger.info(f"Dataset de teste criado: {len(dataset)} amostras")
    
    # Testar uma amostra
    sample = dataset[0]
    logger.info(f"Shape da imagem augmentada: {sample['image'].shape}")
    logger.info(f"Label: {sample['label']}")
    
    # Visualizar augmentações
    try:
        visualize_augmentations(test_image, augmentation, n_samples=5)
    except Exception as e:
        logger.warning(f"Não foi possível visualizar: {e}")
    
    logger.info("✅ Módulo de augmentation testado com sucesso!")
