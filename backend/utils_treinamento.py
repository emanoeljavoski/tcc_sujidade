"""
Utilitários para Treinamento - TCC
Funções auxiliares para validação cruzada e avaliação de modelos
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import List, Tuple, Dict, Any
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class CarregadorDados:
    """Carrega imagens do diretório de dados"""
    
    def __init__(self, caminho_dados: str = 'backend/dados/modulos_limpos_sujos'):
        self.caminho_dados = Path(caminho_dados)
        self.classes = ['Limpo', 'Sujo']
    
    def carregar_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega todas as imagens do diretório.
        
        Returns:
            imagens: Array numpy com imagens (N, H, W, 3)
            labels: Array numpy com labels (N,)
        """
        imagens = []
        labels = []
        
        print("\nCarregando dataset...")
        
        for idx, classe in enumerate(self.classes):
            # Normalizar nome da pasta
            nome_pasta = classe.lower()

            # Caminhos possíveis para esta classe:
            # 1) Estrutura simples: {caminho_dados}/limpo, {caminho_dados}/sujo
            # 2) Estrutura com splits: {caminho_dados}/{train,val,test}/{limpo,sujo}
            caminhos_classe = []

            caminho_simples = self.caminho_dados / nome_pasta
            if caminho_simples.exists():
                caminhos_classe.append(caminho_simples)
            else:
                # Verificar estrutura com train/val/test
                for split in ["train", "val", "test"]:
                    candidato = self.caminho_dados / split / nome_pasta
                    if candidato.exists():
                        caminhos_classe.append(candidato)
            
            if not caminhos_classe:
                raise FileNotFoundError(
                    f"\nERRO: Pasta não encontrada para a classe '{nome_pasta}' em {self.caminho_dados}\n\n"
                    "Certifique-se de ter suas imagens organizadas em uma das estruturas a seguir:\n"
                    f"  {self.caminho_dados}/limpo/ e {self.caminho_dados}/sujo/  (estrutura simples)\n"
                    f"  ou {self.caminho_dados}/{{train,val,test}}/limpo/ e .../sujo/  (estrutura com splits)\n\n"
                    f"Faça upload das suas imagens antes de executar o treinamento!"
                )
            
            # Buscar imagens em todos os caminhos válidos para esta classe
            arquivos_img = []
            for caminho_classe in caminhos_classe:
                arquivos_img += list(caminho_classe.glob('*.jpg'))
                arquivos_img += list(caminho_classe.glob('*.png'))
                arquivos_img += list(caminho_classe.glob('*.jpeg'))
                arquivos_img += list(caminho_classe.glob('*.JPG'))
                arquivos_img += list(caminho_classe.glob('*.PNG'))
            
            if len(arquivos_img) == 0:
                raise ValueError(
                    f"\nERRO: Nenhuma imagem encontrada em {caminho_classe}\n"
                    "Faça upload das suas imagens primeiro!"
                )
            
            print(f"   Carregando {classe}: {len(arquivos_img)} imagens...")
            
            for img_path in arquivos_img:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224))  # Resize padrão
                    imagens.append(np.array(img))
                    labels.append(idx)
                except Exception as e:
                    print(f"   Erro ao carregar {img_path}: {e}")
        
        imagens = np.array(imagens)
        labels = np.array(labels)
        
        print("\nDataset carregado:")
        print(f"   Total: {len(imagens)} imagens")
        print(f"   Limpo: {np.sum(labels == 0)} imagens")
        print(f"   Sujo: {np.sum(labels == 1)} imagens")
        print(f"   Shape: {imagens.shape}")
        
        return imagens, labels


class AvaliadorModelo:
    """Avalia modelo e calcula métricas"""
    
    @staticmethod
    def avaliar(modelo, loader_val, y_val_true, dispositivo='mps'):
        """
        Avalia modelo no conjunto de validação.
        
        Args:
            modelo: Modelo PyTorch
            loader_val: DataLoader de validação
            y_val_true: Labels verdadeiros
            dispositivo: Dispositivo (mps/cpu)
            
        Returns:
            dict: Métricas de avaliação
        """
        modelo.eval()
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for batch in loader_val:
                imagens = batch[0] if not isinstance(batch, dict) else batch['image']
                imagens = imagens.to(dispositivo)
                outputs = modelo(imagens)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
        
        # Calcular métricas
        acc = accuracy_score(y_val_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_true, y_pred, average='weighted', zero_division=0
        )
        cm = confusion_matrix(y_val_true, y_pred)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'y_true': y_val_true.tolist(),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    @staticmethod
    def agregar_metricas(resultados_folds: List[Dict]) -> Dict:
        """
        Agrega métricas de todos os folds.
        
        Args:
            resultados_folds: Lista com resultados de cada fold
            
        Returns:
            dict: Métricas agregadas
        """
        acuracias = [fold['accuracy'] for fold in resultados_folds]
        precisions = [fold['precision'] for fold in resultados_folds]
        recalls = [fold['recall'] for fold in resultados_folds]
        f1s = [fold['f1_score'] for fold in resultados_folds]
        
        return {
            'acuracia_media': np.mean(acuracias),
            'acuracia_std': np.std(acuracias),
            'acuracia_min': np.min(acuracias),
            'acuracia_max': np.max(acuracias),
            'precision_media': np.mean(precisions),
            'recall_media': np.mean(recalls),
            'f1_media': np.mean(f1s),
            'tempo_total_horas': sum(f.get('tempo_treino', 0) for f in resultados_folds) / 3600
        }


class GeradorVisualizacoes:
    """Gera visualizações para TCC"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classes = ['Limpo', 'Sujo']
    
    def gerar_matriz_confusao(self, resultados_folds: List[Dict]):
        """Gera matriz de confusão agregada"""
        
        # Agregar matrizes de todos os folds
        cm_total = np.zeros((2, 2))
        for fold in resultados_folds:
            cm_total += np.array(fold['confusion_matrix'])
        
        # Plotar
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_total, 
            annot=True, 
            fmt='g', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            cbar_kws={'label': 'Número de Predições'}
        )
        plt.title('Matriz de Confusão - Validação Cruzada 5-Fold', 
                 fontsize=14, pad=20, fontweight='bold')
        plt.ylabel('Real', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        plt.tight_layout()
        
        caminho = self.output_dir / 'matriz_confusao.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Matriz de confusão salva: {caminho}")
    
    def gerar_curvas_treinamento(self, historico: Dict):
        """Gera curvas de loss e accuracy"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epocas = range(1, len(historico['train_loss']) + 1)
        
        # Loss
        ax1.plot(epocas, historico['train_loss'], 'b-', label='Treino', linewidth=2)
        ax1.plot(epocas, historico['val_loss'], 'r-', label='Validação', linewidth=2)
        ax1.set_title('Perda durante Treinamento', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epocas, historico['train_acc'], 'b-', label='Treino', linewidth=2)
        ax2.plot(epocas, historico['val_acc'], 'r-', label='Validação', linewidth=2)
        ax2.set_title('Acurácia durante Treinamento', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('Acurácia', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        caminho = self.output_dir / 'curvas_treinamento.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Curvas de treinamento salvas: {caminho}")
    
    def gerar_comparacao_folds(self, resultados_folds: List[Dict]):
        """Gera comparação de acurácia entre folds"""
        
        folds_nums = [f['fold'] for f in resultados_folds]
        folds_acc = [f['accuracy'] for f in resultados_folds]
        media_acc = np.mean(folds_acc)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(folds_nums, folds_acc, color='steelblue', alpha=0.8, edgecolor='black')
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.axhline(
            y=media_acc, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f"Média: {media_acc:.2%}"
        )
        plt.title('Acurácia por Fold - Validação Cruzada', 
                 fontsize=14, pad=20, fontweight='bold')
        plt.xlabel('Fold', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        caminho = self.output_dir / 'comparacao_folds.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Comparação entre folds salva: {caminho}")


def estimar_tempo_treinamento(num_imagens: int, num_folds: int = 5, num_epocas: int = 40) -> float:
    """
    Estima tempo total de treinamento.
    
    Args:
        num_imagens: Número total de imagens
        num_folds: Número de folds
        num_epocas: Número de épocas por fold
        
    Returns:
        float: Tempo estimado em horas
    """
    # Estimativa conservadora baseada em EfficientNet-B4 no M1 Pro
    # ~2-3 minutos por época com batch_size=8
    minutos_por_epoca = 2.5
    
    # Com data augmentation 15x
    imagens_aumentadas = num_imagens * 15
    
    # Ajustar baseado no tamanho do dataset
    if imagens_aumentadas < 1000:
        minutos_por_epoca = 1.5
    elif imagens_aumentadas > 3000:
        minutos_por_epoca = 3.5
    
    tempo_total_min = minutos_por_epoca * num_epocas * num_folds
    return tempo_total_min / 60  # Converter para horas
