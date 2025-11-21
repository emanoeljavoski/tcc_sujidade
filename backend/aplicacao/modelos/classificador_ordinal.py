"""
Classificador Ordinal para Níveis de Sujidade em Painéis Solares
Desenvolvido para TCC - Engenharia Mecatrônica

IMPLEMENTAÇÃO DE ORDINAL CLASSIFICATION:
- Trata classes como ORDENADAS (limpo < pouco sujo < sujo < muito sujo)
- 3 classificadores binários em cascata
- Loss que penaliza menos erros entre classes adjacentes
- Melhor para classes com ordem natural
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

logger = logging.getLogger(__name__)

class OrdinalEfficientNet(nn.Module):
    """
    Classificação ordinal para níveis de sujidade.
    Trata como ORDEM ao invés de classes independentes.
    
    Arquitetura:
    - EfficientNet-B4 como backbone
    - 3 classificadores binários em cascata
    - threshold_1: Limpo vs [Pouco, Sujo, Muito]
    - threshold_2: [Limpo, Pouco] vs [Sujo, Muito]
    - threshold_3: [Limpo, Pouco, Sujo] vs Muito
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        
        # Backbone EfficientNet-B4
        self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None)
        n_features = 1792  # B4 features
        
        # Remove classificador original
        self.backbone.classifier = nn.Identity()
        
        # 3 classificadores binários em cascata para 4 classes
        self.thresholds = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(n_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 1)  # Saída binária
            )
            for _ in range(num_classes - 1)  # 3 thresholds para 4 classes
        ])
        
        self.num_classes = num_classes
        
        # Congelar camadas iniciais
        if pretrained:
            total_params = len(list(self.backbone.parameters()))
            trainable_start = max(0, total_params - 20)
            
            for i, param in enumerate(self.backbone.parameters()):
                if i < trainable_start:
                    param.requires_grad = False
                    
            logger.info(f"🔒 Congeladas {trainable_start} camadas do backbone")
    
    def forward(self, x):
        """
        Forward pass com classificação ordinal.
        
        Args:
            x: Tensor de entrada (batch, 3, 224, 224)
            
        Returns:
            predictions: Classes preditas (0, 1, 2, 3)
            thresholds: Probabilidades dos 3 thresholds
        """
        # Extrair features
        features = self.backbone(x)
        
        # Calcular probabilidades dos thresholds
        threshold_probs = []
        for threshold_classifier in self.thresholds:
            prob = torch.sigmoid(threshold_classifier(features))
            threshold_probs.append(prob)
        
        thresholds_tensor = torch.cat(threshold_probs, dim=1)
        
        # Converter thresholds para classe ordinal
        # Se threshold > 0.5, soma 1 à classe
        predictions = (thresholds_tensor > 0.5).sum(dim=1)
        
        return predictions, thresholds_tensor

def ordinal_loss(thresholds, targets, device='cpu'):
    """
    Loss function para classificação ordinal.
    Penaliza menos erros entre classes adjacentes.
    
    Ex: confundir "pouco sujo" com "sujo" é menos grave que
        confundir "limpo" com "muito sujo"
    
    Args:
        thresholds: Probabilidades dos thresholds (batch, n_thresholds)
        targets: Classes verdadeiras (batch,)
        device: Dispositivo de computação
        
    Returns:
        loss: BCE loss ponderado
    """
    batch_size = targets.size(0)
    n_thresholds = thresholds.size(1)
    
    # Converter labels para targets binários dos thresholds
    # Para classe 0: [0, 0, 0] (nenhum threshold ativado)
    # Para classe 1: [1, 0, 0] (primeiro threshold ativado)
    # Para classe 2: [1, 1, 0] (primeiro e segundo ativados)
    # Para classe 3: [1, 1, 1] (todos ativados)
    threshold_targets = torch.zeros(batch_size, n_thresholds).to(device)
    
    for i in range(n_thresholds):
        threshold_targets[:, i] = (targets > i).float()
    
    # BCE loss para cada threshold
    bce_loss = F.binary_cross_entropy(thresholds, threshold_targets, reduction='none')
    
    # Ponderação: erros em thresholds mais altos são menos graves
    # Isso reflete que errar classes adjacentes é menos problemático
    weights = torch.tensor([1.0, 0.8, 0.6][:n_thresholds]).to(device)
    weighted_loss = (bce_loss * weights.unsqueeze(0)).mean()
    
    return weighted_loss

def ordinal_accuracy(predictions, targets):
    """
    Calcula acurácia para classificação ordinal.
    """
    return (predictions == targets).float().mean().item()

def adjacent_accuracy(predictions, targets):
    """
    Calcula acurácia considerando classes adjacentes como corretas.
    Útil para avaliar se o modelo está "próximo" da resposta correta.
    """
    adjacent_correct = (torch.abs(predictions - targets) <= 1).float().mean().item()
    return adjacent_correct

class ClassificadorOrdinal:
    """
    Wrapper para classificador ordinal com EfficientNet-B4.
    """
    
    def __init__(self, caminho_modelo=None, num_classes=4):
        """
        Inicializa classificador ordinal.
        
        Args:
            caminho_modelo: Caminho para modelo salvo
            num_classes: Número de classes ordinais
        """
        self.dispositivo = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.num_classes = num_classes
        self.classes = ['limpo', 'pouco sujo', 'sujo', 'muito sujo'][:num_classes]
        
        logger.info(f"🔧 Inicializando classificador ordinal no dispositivo: {self.dispositivo}")
        
        # Criar modelo
        self.modelo = OrdinalEfficientNet(num_classes=num_classes, pretrained=True)
        self.modelo.to(self.dispositivo)
        
        # Carregar pesos se disponível
        if caminho_modelo:
            self.carregar_modelo(caminho_modelo)
        else:
            logger.info("📥 Usando EfficientNet-B4 pré-treinado (ImageNet)")
        
        # Transforms para inference
        self.transform_inference = self._criar_transforms_inference()
    
    def _criar_transforms_inference(self):
        """Transforms para inference."""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocessar_imagem(self, imagem):
        """Preprocessa imagem para o modelo."""
        from PIL import Image
        
        if isinstance(imagem, str):
            imagem_pil = Image.open(imagem).convert('RGB')
        elif isinstance(imagem, np.ndarray):
            if imagem.dtype == np.uint8:
                imagem_pil = Image.fromarray(imagem)
            else:
                imagem_pil = Image.fromarray((imagem * 255).astype(np.uint8))
        else:
            imagem_pil = imagem.convert('RGB')
        
        tensor = self.transform_inference(imagem_pil)
        return tensor.unsqueeze(0).to(self.dispositivo)
    
    def classificar(self, imagem):
        """
        Classifica imagem usando abordagem ordinal.
        
        Args:
            imagem: PIL Image, caminho ou numpy array
            
        Returns:
            dict: Resultado com classe ordinal e confianças
        """
        try:
            # Preprocessar
            tensor = self.preprocessar_imagem(imagem)
            
            # Inferência
            self.modelo.eval()
            with torch.no_grad():
                predictions, thresholds = self.modelo(tensor)
                
                # Converter para classe
                classe_predita = predictions.item()
                
                # Calcular confiança baseada nos thresholds
                # Confiança média dos thresholds relevantes
                if classe_predita == 0:
                    confianca = (1 - thresholds[0][0]).item()
                elif classe_predita == self.num_classes - 1:
                    confianca = thresholds[0][-1].item()
                else:
                    confianca = abs(thresholds[0][classe_predita-1] - thresholds[0][classe_predita]).item()
                
                confianca = max(0, min(1, confianca)) * 100
                
                # Probabilidades estimadas para cada classe
                probs = self._calcular_probabilidades_ordenadas(thresholds[0])
            
            return {
                'classe': self.classes[classe_predita],
                'classe_idx': classe_predita,
                'confianca': confianca,
                'probabilidades': dict(zip(self.classes, probs)),
                'thresholds': thresholds[0].cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na classificação ordinal: {e}")
            return {
                'classe': 'erro',
                'classe_idx': -1,
                'confianca': 0.0,
                'probabilidades': {},
                'erro': str(e)
            }
    
    def _calcular_probabilidades_ordenadas(self, thresholds):
        """
        Calcula probabilidades para cada classe baseado nos thresholds.
        """
        probs = []
        
        # Probabilidade da classe 0: P(t1=0)
        probs.append((1 - thresholds[0].item()) * 100)
        
        # Probabilidade da classe 1: P(t1=1, t2=0)
        probs.append(thresholds[0].item() * (1 - thresholds[1].item()) * 100)
        
        # Probabilidade da classe 2: P(t1=1, t2=1, t3=0)
        if len(thresholds) > 2:
            probs.append(thresholds[0].item() * thresholds[1].item() * (1 - thresholds[2].item()) * 100)
        else:
            probs.append(0.0)
        
        # Probabilidade da classe 3: P(t1=1, t2=1, t3=1)
        if len(thresholds) > 2:
            probs.append(thresholds[0].item() * thresholds[1].item() * thresholds[2].item() * 100)
        else:
            probs.append(0.0)
        
        # Normalizar para somar 100%
        total = sum(probs)
        if total > 0:
            probs = [p / total * 100 for p in probs]
        
        return probs
    
    def treinar_ordinal(
        self,
        imagens_treino: List[np.ndarray],
        labels_treino: List[int],
        imagens_val: List[np.ndarray] = None,
        labels_val: List[int] = None,
        epocas: int = 30,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        salvar_modelo: str = None
    ):
        """
        Treina modelo com loss ordinal.
        
        Args:
            imagens_treino: Lista de imagens de treinamento
            labels_treino: Labels (0, 1, 2, 3)
            imagens_val: Imagens de validação (opcional)
            labels_val: Labels de validação (opcional)
            epocas: Número de épocas
            batch_size: Tamanho do batch
            learning_rate: Taxa de aprendizado
            salvar_modelo: Caminho para salvar modelo
            
        Returns:
            dict: Histórico de treinamento
        """
        logger.info("🚀 Iniciando treinamento com classificação ordinal")
        
        # Preparar dados
        from torch.utils.data import DataLoader, TensorDataset
        
        # Converter para tensores
        train_images = torch.stack([self.preprocessar_imagem(img).squeeze(0) for img in imagens_treino])
        train_labels = torch.tensor(labels_treino, dtype=torch.long)
        
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Validação
        val_loader = None
        if imagens_val and labels_val:
            val_images = torch.stack([self.preprocessar_imagem(img).squeeze(0) for img in imagens_val])
            val_labels = torch.tensor(labels_val, dtype=torch.long)
            val_dataset = TensorDataset(val_images, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Otimizador
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.modelo.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        # Histórico
        historico = {
            'train_loss': [],
            'train_acc': [],
            'train_adj_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_adj_acc': []
        }
        
        melhor_acc_val = 0.0
        
        for epoca in range(epocas):
            # Treinamento
            self.modelo.train()
            train_loss = 0.0
            train_correct = 0
            train_adj_correct = 0
            train_total = 0
            
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(self.dispositivo)
                batch_labels = batch_labels.to(self.dispositivo)
                
                optimizer.zero_grad()
                
                # Forward
                predictions, thresholds = self.modelo(batch_images)
                
                # Loss ordinal
                loss = ordinal_loss(thresholds, batch_labels, self.dispositivo)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Métricas
                train_loss += loss.item()
                train_correct += (predictions == batch_labels).sum().item()
                train_adj_correct += (torch.abs(predictions - batch_labels) <= 1).sum().item()
                train_total += batch_labels.size(0)
            
            # Validação
            val_loss, val_acc, val_adj_acc = 0.0, 0.0, 0.0
            if val_loader:
                self.modelo.eval()
                val_correct = 0
                val_adj_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_images, batch_labels in val_loader:
                        batch_images = batch_images.to(self.dispositivo)
                        batch_labels = batch_labels.to(self.dispositivo)
                        
                        predictions, thresholds = self.modelo(batch_images)
                        loss = ordinal_loss(thresholds, batch_labels, self.dispositivo)
                        
                        val_loss += loss.item()
                        val_correct += (predictions == batch_labels).sum().item()
                        val_adj_correct += (torch.abs(predictions - batch_labels) <= 1).sum().item()
                        val_total += batch_labels.size(0)
                
                val_acc = 100 * val_correct / val_total
                val_adj_acc = 100 * val_adj_correct / val_total
                val_loss /= len(val_loader)
            
            # Métricas de treinamento
            train_acc = 100 * train_correct / train_total
            train_adj_acc = 100 * train_adj_correct / train_total
            train_loss /= len(train_loader)
            
            # Scheduler
            scheduler.step()
            
            # Salvar histórico
            historico['train_loss'].append(train_loss)
            historico['train_acc'].append(train_acc)
            historico['train_adj_acc'].append(train_adj_acc)
            historico['val_loss'].append(val_loss)
            historico['val_acc'].append(val_acc)
            historico['val_adj_acc'].append(val_adj_acc)
            
            # Salvar melhor modelo
            if val_acc > melhor_acc_val:
                melhor_acc_val = val_acc
                if salvar_modelo:
                    self.salvar_modelo(salvar_modelo)
            
            # Log
            logger.info(
                f"Época {epoca+1}/{epocas} | "
                f"Loss: {train_loss:.4f} | "
                f"Acc: {train_acc:.2f}% | "
                f"Adj Acc: {train_adj_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Val Adj Acc: {val_adj_acc:.2f}%"
            )
        
        logger.info(f"✅ Treinamento ordinal concluído! Melhor acurácia: {melhor_acc_val:.2f}%")
        
        return {
            'historico': historico,
            'melhor_acuracia': melhor_acc_val,
            'epocas_treinadas': epocas
        }
    
    def avaliar_ordinal(self, imagens_test: List[np.ndarray], labels_test: List[int]):
        """
        Avaliação completa com métricas ordinais.
        
        Args:
            imagens_test: Imagens de teste
            labels_test: Labels verdadeiros
            
        Returns:
            dict: Relatório completo de avaliação
        """
        logger.info("📊 Avaliando classificador ordinal...")
        
        self.modelo.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for img, label in zip(imagens_test, labels_test):
                resultado = self.classificar(img)
                predictions.append(resultado['classe_idx'])
                true_labels.append(label)
        
        predictions = torch.tensor(predictions)
        true_labels = torch.tensor(labels_test)
        
        # Métricas
        accuracy = ordinal_accuracy(predictions, true_labels)
        adjacent_acc = adjacent_accuracy(predictions, true_labels)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels.numpy(), predictions.numpy())
        
        # Classification report
        report = classification_report(
            true_labels.numpy(), 
            predictions.numpy(),
            target_names=self.classes,
            output_dict=True
        )
        
        # Análise de erros por severidade
        erros = torch.abs(predictions - true_labels)
        erro_medio = erros.float().mean().item()
        erro_max = erros.max().item()
        
        relatorio = {
            'accuracy': accuracy * 100,
            'adjacent_accuracy': adjacent_acc * 100,
            'erro_medio': erro_medio,
            'erro_maximo': erro_max,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'total_amostras': len(true_labels)
        }
        
        logger.info(f"✅ Avaliação concluída!")
        logger.info(f"Acurácia exata: {relatorio['accuracy']:.2f}%")
        logger.info(f"Acurácia adjacente (±1 classe): {relatorio['adjacent_accuracy']:.2f}%")
        logger.info(f"Erro médio: {relatorio['erro_medio']:.2f} classes")
        
        return relatorio
    
    def salvar_modelo(self, caminho):
        """Salva modelo ordinal."""
        torch.save({
            'modelo_state_dict': self.modelo.state_dict(),
            'num_classes': self.num_classes,
            'classes': self.classes,
            'dispositivo': str(self.dispositivo),
            'tipo': 'ordinal'
        }, caminho)
        logger.info(f"💾 Modelo ordinal salvo em: {caminho}")
    
    def carregar_modelo(self, caminho):
        """Carrega modelo ordinal."""
        checkpoint = torch.load(caminho, map_location=self.dispositivo)
        self.modelo.load_state_dict(checkpoint['modelo_state_dict'])
        self.num_classes = checkpoint['num_classes']
        self.classes = checkpoint['classes']
        logger.info(f"📥 Modelo ordinal carregado de: {caminho}")
    
    def get_informacoes(self):
        """Retorna informações do modelo ordinal."""
        total_params = sum(p.numel() for p in self.modelo.parameters())
        trainable_params = sum(p.numel() for p in self.modelo.parameters() if p.requires_grad)
        
        return {
            'arquitetura': 'OrdinalEfficientNet-B4',
            'tipo': 'ordinal_classification',
            'num_classes': self.num_classes,
            'classes': self.classes,
            'num_thresholds': self.num_classes - 1,
            'dispositivo': str(self.dispositivo),
            'total_parametros': total_params,
            'parametros_treinaveis': trainable_params,
            'percentagem_treinavel': (trainable_params / total_params) * 100
        }

if __name__ == "__main__":
    # Teste do classificador ordinal
    logger.info("🧪 Testando classificador ordinal...")
    
    classificador = ClassificadorOrdinal(num_classes=4)
    info = classificador.get_informacoes()
    
    logger.info("✅ Classificador ordinal criado com sucesso!")
    logger.info(f"📊 Informações: {info}")
