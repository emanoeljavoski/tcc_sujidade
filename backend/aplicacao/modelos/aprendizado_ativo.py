"""
Sistema de Active Learning para Expansibilidade do Dataset
Desenvolvido para TCC - Engenharia Mecatrônica

FUNCIONALIDADE:
- Sugere quais imagens novas são mais importantes para anotar
- Usa MC Dropout para medir incerteza do modelo
- Prioriza imagens com maior incerteza (maximiza ganho de performance)
- Sistema expansível que se adapta quando conseguir mais imagens
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
import os

# Import dos modelos
from .classificador import ClassificadorSujidade
from .ordinal_classifier import ClassificadorOrdinal

logger = logging.getLogger(__name__)

class ActiveLearningModule:
    """
    Sistema expansível com active learning.
    Sugere quais imagens novas são mais importantes anotar.
    
    Estratégias implementadas:
    1. MC Dropout - Mede incerteza via predições estocásticas
    2. Entropia - Prioriza predições mais incertas
    3. Margin Sampling - Prioriza predições com menor margem
    4. Committee - Usa múltiplos modelos (ensemble)
    """
    
    def __init__(self, modelo_treinado, strategy='mc_dropout'):
        """
        Inicializa módulo de active learning.
        
        Args:
            modelo_treinado: Modelo treinado (ClassificadorSujidade ou ClassificadorOrdinal)
            strategy: Estratégia de active learning ('mc_dropout', 'entropy', 'margin', 'committee')
        """
        self.modelo = modelo_treinado
        self.modelo_original = modelo_treinado  # Backup
        self.dispositivo = modelo_treinado.dispositivo
        self.strategy = strategy
        
        logger.info(f"🤖 Inicializando Active Learning com estratégia: {strategy}")
        
        # Configurar modelo para MC Dropout se necessário
        if strategy == 'mc_dropout':
            self._configurar_mc_dropout()
        
        # Para committee, criar ensemble de modelos
        if strategy == 'committee':
            self._criar_committee()
    
    def _configurar_mc_dropout(self):
        """
        Configura modelo para MC Dropout.
        Ativa dropout durante inferência para medição de incerteza.
        """
        logger.info("🎲 Configurando MC Dropout...")
        
        # Força dropout ativo durante inferência
        def set_dropout_train(m):
            if isinstance(m, nn.Dropout):
                m.train()
        
        self.modelo.modelo.apply(set_dropout_train)
        logger.info("✅ Dropout configurado para modo MC")
    
    def _criar_committee(self):
        """
        Cria committee de modelos para ensemble.
        Usa diferentes checkpoints do treinamento.
        """
        logger.info("👥 Criando committee de modelos...")
        
        # Procurar por diferentes folds salvos
        modelos_committee = []
        folds_dir = Path('modelos_salvos')
        
        if folds_dir.exists():
            fold_files = list(folds_dir.glob('fold_*_efficientnet_b4.pth'))
            fold_files.sort()
            
            # Usar até 5 modelos diferentes
            for i, fold_file in enumerate(fold_files[:5]):
                try:
                    modelo_committee = ClassificadorSujidade(num_classes=self.modelo.num_classes)
                    modelo_committee.carregar_modelo(str(fold_file))
                    modelos_committee.append(modelo_committee)
                    logger.info(f"✅ Carregado modelo do fold {i+1}")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao carregar {fold_file}: {e}")
        
        if len(modelos_committee) == 0:
            # Fallback: usar o mesmo modelo com diferentes configurações
            logger.warning("⚠️ Nenhum modelo de fold encontrado, usando modelo original")
            modelos_committee = [self.modelo]
        
        self.committee = modelos_committee
        logger.info(f"👥 Committee criado com {len(self.committee)} modelos")
    
    def predict_with_uncertainty(self, imagem, n_samples=10):
        """
        Predição + medida de incerteza usando estratégia selecionada.
        
        Args:
            imagem: Imagem para analisar
            n_samples: Número de amostras para MC Dropout
            
        Returns:
            dict: Predição e métricas de incerteza
        """
        try:
            if self.strategy == 'mc_dropout':
                return self._predict_mc_dropout(imagem, n_samples)
            elif self.strategy == 'entropy':
                return self._predict_entropy(imagem)
            elif self.strategy == 'margin':
                return self._predict_margin(imagem)
            elif self.strategy == 'committee':
                return self._predict_committee(imagem)
            else:
                raise ValueError(f"Estratégia desconhecida: {self.strategy}")
                
        except Exception as e:
            logger.error(f"❌ Erro na predição com incerteza: {e}")
            return {
                'prediction': 0,
                'uncertainty': 1.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _predict_mc_dropout(self, imagem, n_samples=10):
        """
        Predição com MC Dropout.
        Ativa dropout em inferência e faz múltiplas predições.
        """
        self.modelo.modelo.eval()
        
        # Força dropout ativo
        def set_dropout_train(m):
            if isinstance(m, nn.Dropout):
                m.train()
        
        self.modelo.modelo.apply(set_dropout_train)
        
        # Múltiplas predições estocásticas
        predictions = []
        probabilidades = []
        
        with torch.no_grad():
            img_tensor = self.modelo.preprocessar_imagem(imagem)
            
            for _ in range(n_samples):
                outputs = self.modelo.modelo(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                predictions.append(pred)
                probabilidades.append(probs.cpu().numpy()[0])
        
        # Estatísticas
        predictions = np.array(predictions)
        probabilidades = np.array(probabilidades)
        
        # Predição final (majority vote)
        unique, counts = np.unique(predictions, return_counts=True)
        final_prediction = unique[np.argmax(counts)]
        
        # Medidas de incerteza
        mean_probs = np.mean(probabilidades, axis=0)
        std_probs = np.std(probabilidades, axis=0)
        
        # Entropia como incerteza
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        # Confiança baseada na consistência das predições
        confidence = np.max(counts) / n_samples
        
        # Incerteza baseada na variância
        uncertainty = np.mean(std_probs)
        
        return {
            'prediction': int(final_prediction),
            'uncertainty': float(uncertainty),
            'confidence': float(confidence),
            'entropy': float(entropy),
            'mean_probabilities': mean_probs.tolist(),
            'std_probabilities': std_probs.tolist(),
            'prediction_distribution': dict(zip(unique, counts.tolist()))
        }
    
    def _predict_entropy(self, imagem):
        """
        Predição com incerteza baseada em entropia.
        """
        resultado = self.modelo.classificar(imagem)
        
        if 'erro' in resultado:
            return {
                'prediction': 0,
                'uncertainty': 1.0,
                'confidence': 0.0,
                'error': resultado['erro']
            }
        
        probs = np.array(list(resultado['probabilidades'].values())) / 100
        
        # Entropia
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalizar entropia para [0, 1]
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        return {
            'prediction': resultado['classe_idx'],
            'uncertainty': float(normalized_entropy),
            'confidence': resultado['confianca'] / 100,
            'entropy': float(entropy),
            'probabilities': probs.tolist()
        }
    
    def _predict_margin(self, imagem):
        """
        Predição com incerteza baseada em margin sampling.
        Margem = diferença entre as 2 maiores probabilidades.
        """
        resultado = self.modelo.classificar(imagem)
        
        if 'erro' in resultado:
            return {
                'prediction': 0,
                'uncertainty': 1.0,
                'confidence': 0.0,
                'error': resultado['erro']
            }
        
        probs = np.array(list(resultado['probabilidades'].values())) / 100
        
        # Ordenar probabilidades
        sorted_probs = np.sort(probs)[::-1]
        
        # Margin = diferença entre top 2
        if len(sorted_probs) >= 2:
            margin = sorted_probs[0] - sorted_probs[1]
        else:
            margin = 0.0
        
        # Incerteza = 1 - margin (quanto menor a margem, mais incerto)
        uncertainty = 1 - margin
        
        return {
            'prediction': resultado['classe_idx'],
            'uncertainty': float(uncertainty),
            'confidence': resultado['confianca'] / 100,
            'margin': float(margin),
            'probabilities': probs.tolist()
        }
    
    def _predict_committee(self, imagem):
        """
        Predição usando committee de modelos.
        Mede discordância entre diferentes modelos.
        """
        if not hasattr(self, 'committee'):
            logger.warning("⚠️ Committee não disponível, usando modelo único")
            return self._predict_entropy(imagem)
        
        predictions = []
        confiancas = []
        probabilidades = []
        
        # Predições de cada modelo do committee
        for modelo in self.committee:
            resultado = modelo.classificar(imagem)
            if 'erro' not in resultado:
                predictions.append(resultado['classe_idx'])
                confiancas.append(resultado['confianca'] / 100)
                probs = np.array(list(resultado['probabilidades'].values())) / 100
                probabilidades.append(probs)
        
        if len(predictions) == 0:
            return {
                'prediction': 0,
                'uncertainty': 1.0,
                'confidence': 0.0,
                'error': 'Nenhuma predição bem-sucedida'
            }
        
        # Predição final (majority vote)
        unique, counts = np.unique(predictions, return_counts=True)
        final_prediction = unique[np.argmax(counts)]
        
        # Medidas de incerteza
        vote_entropy = -np.sum((counts / len(predictions)) * np.log(counts / len(predictions) + 1e-10))
        
        # Discordância nas probabilidades
        if len(probabilidades) > 1:
            mean_probs = np.mean(probabilidades, axis=0)
            std_probs = np.std(probabilidades, axis=0)
            prob_disagreement = np.mean(std_probs)
        else:
            prob_disagreement = 0.0
        
        # Confiança baseada na concordância do voto
        vote_confidence = np.max(counts) / len(predictions)
        
        # Incerteza combinada
        uncertainty = (vote_entropy / np.log(len(unique)) + prob_disagreement) / 2
        
        return {
            'prediction': int(final_prediction),
            'uncertainty': float(uncertainty),
            'confidence': float(vote_confidence),
            'vote_entropy': float(vote_entropy),
            'prob_disagreement': float(prob_disagreement),
            'committee_agreement': dict(zip(unique.tolist(), counts.tolist())),
            'mean_probabilities': mean_probs.tolist() if len(probabilidades) > 1 else probabilidades[0].tolist()
        }
    
    def suggest_samples(self, novas_imagens, n=20, diversity_factor=0.3):
        """
        Analisa imagens novas e sugere as N mais importantes para anotar.
        Prioriza imagens onde modelo tem maior incerteza.
        
        Args:
            novas_imagens: Lista de imagens para analisar
            n: Número de imagens a sugerir
            diversity_factor: Fator para diversificar seleção (0-1)
            
        Returns:
            dict: Sugestões com análises detalhadas
        """
        logger.info(f"🎯 Analisando {len(novas_imagens)} imagens para active learning...")
        
        if len(novas_imagens) == 0:
            return {
                'total_imagens': 0,
                'sugestoes': [],
                'message': 'Nenhuma imagem para analisar'
            }
        
        # Analisar cada imagem
        analises = []
        
        for idx, img in enumerate(novas_imagens):
            try:
                resultado = self.predict_with_uncertainty(img)
                
                analise = {
                    'indice': idx,
                    'incerteza': resultado['uncertainty'],
                    'confianca': resultado['confidence'],
                    'predicao': resultado['prediction'],
                    'estrategia': self.strategy
                }
                
                # Adicionar métricas específicas da estratégia
                if 'entropy' in resultado:
                    analise['entropia'] = resultado['entropy']
                if 'margin' in resultado:
                    analise['margem'] = resultado['margin']
                if 'vote_entropy' in resultado:
                    analise['entropia_voto'] = resultado['vote_entropy']
                
                analises.append(analise)
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao analisar imagem {idx}: {e}")
                analises.append({
                    'indice': idx,
                    'incerteza': 1.0,  # Máxima incerteza em caso de erro
                    'confianca': 0.0,
                    'predicao': 0,
                    'erro': str(e)
                })
        
        # Ordenar por incerteza (maiores primeiro)
        analises.sort(key=lambda x: x['incerteza'], reverse=True)
        
        # Aplicar diversificação se solicitado
        if diversity_factor > 0 and len(analises) > n:
            analises = self._diversify_selection(analises, n, diversity_factor)
        else:
            analises = analises[:n]
        
        # Preparar resultado
        sugestoes = []
        for i, analise in enumerate(analises):
            sugestao = {
                'rank': i + 1,
                'indice_imagem': analise['indice'],
                'prioridade': 'ALTA' if analise['incerteza'] > 0.7 else 'MEDIA' if analise['incerteza'] > 0.4 else 'BAIXA',
                'incerteza': round(analise['incerteza'], 3),
                'confianca': round(analise['confianca'], 3),
                'predicao': analise['predicao'],
                'razao': self._get_prioridade_reason(analise)
            }
            
            # Adicionar métricas específicas
            if 'entropia' in analise:
                sugestao['entropia'] = round(analise['entropia'], 3)
            if 'margem' in analise:
                sugestao['margem'] = round(analise['margem'], 3)
            
            sugestoes.append(sugestao)
        
        # Estatísticas gerais
        incertezas = [a['incerteza'] for a in analises]
        confiancas = [a['confianca'] for a in analises]
        
        resultado = {
            'total_imagens': len(novas_imagens),
            'imagens_sugeridas': len(sugestoes),
            'estrategia': self.strategy,
            'sugestoes': sugestoes,
            'estatisticas': {
                'incerteza_media': np.mean(incertezas),
                'incerteza_max': np.max(incertezas),
                'incerteza_min': np.min(incertezas),
                'confianca_media': np.mean(confiancas),
                'distribuicao_prioridades': {
                    'ALTA': sum(1 for s in sugestoes if s['prioridade'] == 'ALTA'),
                    'MEDIA': sum(1 for s in sugestoes if s['prioridade'] == 'MEDIA'),
                    'BAIXA': sum(1 for s in sugestoes if s['prioridade'] == 'BAIXA')
                }
            },
            'message': f"Anote estas {len(sugestoes)} imagens primeiro para maximizar ganho de performance"
        }
        
        # Log das top sugestões
        logger.info(f"\n🎯 TOP {min(10, len(sugestoes))} IMAGENS PRIORITÁRIAS:")
        for i, sugestao in enumerate(sugestoes[:10]):
            logger.info(
                f"   {i+1}. Imagem #{sugestao['indice_imagem']}: "
                f"Incerteza={sugestao['incerteza']:.3f}, "
                f"Prioridade={sugestao['prioridade']}, "
                f"Predição={sugestao['predicao']}"
            )
        
        return resultado
    
    def _diversify_selection(self, analises, n, diversity_factor):
        """
        Diversifica seleção para evitar clustering de amostras similares.
        """
        if len(analises) <= n:
            return analises
        
        # Seleção baseada em incerteza e diversidade
        selecionadas = []
        restantes = analises.copy()
        
        # Primeiro, selecionar a mais incerta
        selecionadas.append(restantes.pop(0))
        
        # Depois, balancear entre incerteza e diversidade
        while len(selecionadas) < n and restantes:
            melhor_score = -1
            melhor_candidato = None
            melhor_idx = -1
            
            for i, candidato in enumerate(restantes):
                # Score baseado na incerteza
                score_base = candidato['incerteza']
                
                # Penalizar se for muito similar aos já selecionados
                penalidade = 0
                for sel in selecionadas:
                    # Similaridade baseada na predição e incerteza
                    if (candidato['predicao'] == sel['predicao'] and 
                        abs(candidato['incerteza'] - sel['incerteza']) < 0.1):
                        penalidade += 0.2
                
                # Score final
                score_final = score_base * (1 - diversity_factor) - penalidade * diversity_factor
                
                if score_final > melhor_score:
                    melhor_score = score_final
                    melhor_candidato = candidato
                    melhor_idx = i
            
            if melhor_candidato:
                selecionadas.append(melhor_candidato)
                restantes.pop(melhor_idx)
            else:
                break
        
        return selecionadas
    
    def _get_prioridade_reason(self, analise):
        """
        Gera razão da prioridade baseada na estratégia.
        """
        if analise['incerteza'] > 0.7:
            if self.strategy == 'mc_dropout':
                return "Alta variabilidade nas predições estocásticas"
            elif self.strategy == 'entropy':
                return "Alta entropia - distribuição muito uniforme"
            elif self.strategy == 'margin':
                return "Margem muito pequena - classes muito próximas"
            elif self.strategy == 'committee':
                return "Alta discordância entre modelos do committee"
        elif analise['incerteza'] > 0.4:
            return "Incerteza moderada - anotação valiosa"
        else:
            return "Baixa incerteza - menor prioridade"
    
    def visualize_uncertainty_distribution(self, novas_imagens, save_path=None):
        """
        Visualiza distribuição de incertezas para análise.
        
        Args:
            novas_imagens: Lista de imagens para analisar
            save_path: Caminho para salvar visualização
        """
        logger.info("📊 Gerando visualização da distribuição de incertezas...")
        
        # Coletar incertezas
        incertezas = []
        confiancas = []
        
        for img in novas_imagens:
            resultado = self.predict_with_uncertainty(img)
            incertezas.append(resultado['uncertainty'])
            confiancas.append(resultado['confidence'])
        
        # Criar visualização
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histograma de incertezas
        axes[0, 0].hist(incertezas, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 0].set_title('Distribuição de Incertezas')
        axes[0, 0].set_xlabel('Incerteza')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].axvline(np.mean(incertezas), color='red', linestyle='--', label=f'Média: {np.mean(incertezas):.3f}')
        axes[0, 0].legend()
        
        # Histograma de confianças
        axes[0, 1].hist(confiancas, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('Distribuição de Confianças')
        axes[0, 1].set_xlabel('Confiança')
        axes[0, 1].set_ylabel('Frequência')
        axes[0, 1].axvline(np.mean(confiancas), color='red', linestyle='--', label=f'Média: {np.mean(confiancas):.3f}')
        axes[0, 1].legend()
        
        # Scatter plot: Incerteza vs Confiança
        axes[1, 0].scatter(confiancas, incertezas, alpha=0.6, s=30)
        axes[1, 0].set_title('Incerteza vs Confiança')
        axes[1, 0].set_xlabel('Confiança')
        axes[1, 0].set_ylabel('Incerteza')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot comparativo
        data = [incertezas, confiancas]
        labels = ['Incerteza', 'Confiança']
        axes[1, 1].boxplot(data, labels=labels)
        axes[1, 1].set_title('Comparativo: Incerteza vs Confiança')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Visualização salva em: {save_path}")
        
        plt.show()
        
        return {
            'incerteza_media': np.mean(incertezas),
            'incerteza_std': np.std(incertezas),
            'confianca_media': np.mean(confiancas),
            'confianca_std': np.std(confiancas),
            'total_amostras': len(incertezas)
        }
    
    def export_suggestions(self, suggestions, filename='active_learning_suggestions.json'):
        """
        Exporta sugestões para arquivo JSON.
        
        Args:
            suggestions: Dicionário de sugestões
            filename: Nome do arquivo
        """
        os.makedirs('outputs', exist_ok=True)
        
        filepath = os.path.join('outputs', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(suggestions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Sugestões exportadas para: {filepath}")
        
        return filepath

if __name__ == "__main__":
    # Teste do módulo
    logger.info("🧪 Testando módulo de Active Learning...")
    
    # Criar modelo de teste
    modelo_teste = ClassificadorSujidade(num_classes=4)
    
    # Criar módulo active learning
    al_module = ActiveLearningModule(modelo_teste, strategy='mc_dropout')
    
    logger.info("✅ Módulo Active Learning testado com sucesso!")
