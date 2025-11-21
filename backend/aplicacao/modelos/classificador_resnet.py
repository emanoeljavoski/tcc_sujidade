"""
Classificador ResNet50 para Sujidade em Painéis Solares
Alternativa ao EfficientNet para comparação de performance
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from pathlib import Path
import logging
from typing import Optional, List
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class ClassificadorResNet:
    """
    ResNet50 para classificar módulos fotovoltaicos em níveis de sujidade.
    """
    
    def __init__(self, caminho_modelo=None, num_classes=2):
        """
        Inicializa o classificador de sujidade com ResNet50.
        
        Args:
            caminho_modelo (str): Caminho para modelo .pth treinado
            num_classes (int): Número de classes (default: 2 para classificação binária limpo/sujo)
        """
        self.dispositivo = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.num_classes = num_classes
        
        # Definir nomes das classes baseado no número
        if num_classes == 2:
            self.classes = ['limpo', 'sujo']
        elif num_classes == 4:
            self.classes = ['limpo', 'pouco sujo', 'sujo', 'muito sujo']
        else:
            self.classes = [f'classe_{i}' for i in range(num_classes)]
        
        logger.info(f"🔧 Inicializando classificador ResNet50 no dispositivo: {self.dispositivo}")
        
        # Criar modelo
        self.modelo = self._criar_resnet()
        
        # Carregar pesos se disponível
        if caminho_modelo and Path(caminho_modelo).exists():
            self.carregar_modelo(caminho_modelo)
        else:
            logger.info("📥 Usando ResNet50 pré-treinado (ImageNet)")
        
        # Transforms para inference
        self.transform_inference = self._criar_transforms_inference()
        
    def _criar_resnet(self):
        """
        Cria arquitetura ResNet50 com transfer learning.
        
        Returns:
            torch.nn.Module: Modelo ResNet50 customizado
        """
        logger.info("🏗️ Criando ResNet50 com transfer learning...")
        
        # Carregar ResNet50 pré-treinado no ImageNet
        try:
            modelo = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except Exception as e:
            logger.warning(f"Falha ao baixar/carregar pesos ResNet50: {e}. Usando weights=None.")
            modelo = models.resnet50(weights=None)
        
        # ResNet50 tem 2048 features na última camada
        n_features = 2048
        
        # Congelar camadas iniciais para evitar overfitting
        logger.info("🔒 Congelando camadas iniciais (transfer learning)")
        total_params = len(list(modelo.parameters()))
        trainable_start = max(0, total_params - 15)  # Descongelar apenas últimas 15 camadas
        
        for i, param in enumerate(modelo.parameters()):
            if i < trainable_start:
                param.requires_grad = False
        
        # Substituir classificador final
        if self.num_classes == 2:
            modelo.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(n_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, self.num_classes)
            )
        else:
            modelo.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(n_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, self.num_classes)
            )
        
        modelo = modelo.to(self.dispositivo)
        
        # Contar parâmetros treináveis
        total = sum(p.numel() for p in modelo.parameters())
        trainable = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        
        logger.info(f"✅ ResNet50 criado com {self.num_classes} classes")
        logger.info(f"📊 Parâmetros treináveis: {trainable:,} / {total:,}")
        
        return modelo
    
    def _criar_transforms_inference(self):
        """Cria transformações para inferência."""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet usa 224x224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def salvar_modelo(self, caminho: str):
        """
        Salva o modelo treinado em arquivo .pth.
        """
        try:
            payload = {
                'state_dict': self.modelo.state_dict(),
                'num_classes': self.num_classes,
                'classes': self.classes,
                'dispositivo': str(self.dispositivo),
                'arquitetura': 'resnet50',
                'tipo': 'multiclass' if self.num_classes > 2 else 'binary'
            }
            torch.save(payload, caminho)
            logger.info(f"💾 Modelo ResNet50 salvo em: {caminho}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelo: {e}")
    
    def carregar_modelo(self, caminho: str):
        """
        Carrega modelo treinado de arquivo .pth.
        """
        try:
            checkpoint = torch.load(caminho, map_location=self.dispositivo)
            self.modelo.load_state_dict(checkpoint['state_dict'])
            self.modelo.eval()
            logger.info(f"✅ Modelo ResNet50 carregado de: {caminho}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def prever(self, imagem_path: str) -> dict:
        """
        Realiza predição em uma imagem.
        
        Args:
            imagem_path (str): Caminho para a imagem
            
        Returns:
            dict: Resultado da predição com classe e confiança
        """
        try:
            # Carregar e preprocessar imagem
            imagem = Image.open(imagem_path).convert('RGB')
            tensor = self.transform_inference(imagem).unsqueeze(0).to(self.dispositivo)
            
            # Inferência
            self.modelo.eval()
            with torch.no_grad():
                outputs = self.modelo(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            classe_pred = self.classes[predicted.item()]
            confianca = confidence.item()
            
            # Distribuição de probabilidades
            probs_dict = {
                self.classes[i]: float(probabilities[0][i])
                for i in range(len(self.classes))
            }
            
            return {
                'classe': classe_pred,
                'confianca': confianca,
                'probabilidades': probs_dict
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            raise
