"""Script de treinamento completo do classificador binário.

Treina EfficientNet-B4 com validação cruzada 5-fold sobre o dataset
binário consolidado, gerando métricas e artefatos usados no projeto.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import logging
import sys

# Adicionar path do backend
sys.path.append(str(Path(__file__).parent))

from aplicacao.modelos.classificador_sujidade import ClassificadorSujidade
from aplicacao.aumento_dados import AumentoDadosPainelSolar, criar_datasets_aumentados
from aplicacao.modelos.otimizacoes_m1 import OtimizadorM1Pro
from utils_treinamento import (CarregadorDados, AvaliadorModelo, 
                               GeradorVisualizacoes, estimar_tempo_treinamento)
from geracao_documentacao import GeradorDocumentacaoTCC

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('treinamento.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TreinadorTCC:
    """Treinador completo com validação cruzada e geração de relatórios."""
    
    def __init__(self, caminho_dados: str = r'F:\\dataset_2classes_meus_public_50_50'):
        self.caminho_dados = caminho_dados
        self.output_dir = Path('outputs/treinamento_final')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.classes = ['Limpo', 'Sujo']
        self.resultados = {
            'inicio': datetime.now().isoformat(),
            'folds': [],
            'metricas_agregadas': {},
            'configuracao': {}
        }
        
        # Inicializar otimizador para o ambiente M1
        self.otimizador_m1 = OtimizadorM1Pro()
        self.config_m1 = self.otimizador_m1._obter_config_otima()
        
        # Dispositivo
        if torch.backends.mps.is_available():
            self.dispositivo = 'mps'
        elif torch.cuda.is_available():
            self.dispositivo = 'cuda'
        else:
            self.dispositivo = 'cpu'
        logger.info("Dispositivo de treino: %s", self.dispositivo)
    
    def executar_treinamento_completo(self):
        """
        Pipeline completo:
        1. Carrega dados
        2. Validação cruzada 5-fold
        3. Gera todas as métricas e visualizações
        4. Salva documentação
        """
        
        print("\n" + "=" * 80)
        print("TREINAMENTO FINAL DO CLASSIFICADOR BINÁRIO")
        print("   Ambiente de referência: MacBook M1 Pro 8GB RAM")
        print("=" * 80)
        
        # ETAPA 1: Carregar dados
        print("\nETAPA 1/4: Carregando dataset...")
        carregador = CarregadorDados(self.caminho_dados)
        imagens, labels = carregador.carregar_dataset()
        
        # Estimativa de tempo
        tempo_estimado = estimar_tempo_treinamento(len(imagens))
        print(f"\nTempo estimado: ~{tempo_estimado:.1f} horas")
        print("   Recomenda-se manter o equipamento conectado à energia e sem outras cargas pesadas durante o treino.\n")
        
        # ETAPA 2: Validação cruzada 5-fold
        print("\nETAPA 2/4: Validação cruzada 5-fold...")
        self._validacao_cruzada(imagens, labels)
        
        # ETAPA 3: Gerar visualizações
        print("\nETAPA 3/4: Gerando visualizações...")
        self._gerar_visualizacoes()
        
        # ETAPA 4: Gerar documentação
        print("\nETAPA 4/4: Gerando documentação LaTeX...")
        self._gerar_documentacao()
        
        print("\n" + "=" * 80)
        print("TREINAMENTO COMPLETO")
        print("=" * 80)
        print(f"\nResultados salvos em: {self.output_dir}")
        print(f"Acurácia média: {self.resultados['metricas_agregadas']['acuracia_media']:.2%}")
        print(f"Desvio padrão: {self.resultados['metricas_agregadas']['acuracia_std']:.2%}")
        print("\nArquivos auxiliares gerados (tabelas, figuras e resumo) podem ser usados na documentação técnica.")
    
    def _validacao_cruzada(self, imagens, labels):
        """Executa validação cruzada 5-fold"""
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Salvar configuração
        self.resultados['configuracao'] = {
            'arquitetura': 'EfficientNet-B4',
            'num_classes': 2,
            'batch_size': self.config_m1['tamanho_lote'],
            'num_workers': self.config_m1['num_workers'],
            'learning_rate': 0.001,
            'dropout': 0.3,
            'num_epocas': 10,
            'data_augmentation': '5x',
            'n_folds': 5,
            'gradient_accumulation': self.config_m1['gradient_accumulation_steps']
        }
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(imagens, labels)):
            print(f"\n   {'='*60}")
            print(f"   FOLD {fold_idx + 1}/5")
            print(f"   {'='*60}")
            
            # Dividir dados
            X_train, X_val = imagens[train_idx], imagens[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            print(f"   Treino: {len(X_train)} | Validação: {len(X_val)}")
            
            # Criar dataloaders com data augmentation
            from torch.utils.data import DataLoader
            from aplicacao.aumento_dados import DatasetSolarAumentado
            
            aumentador_treino = AumentoDadosPainelSolar(modo='treino', n_aumentos=5)
            aumentador_val = AumentoDadosPainelSolar(modo='val', n_aumentos=1)
            
            dataset_treino = DatasetSolarAumentado(
                imagens=X_train,
                labels=y_train,
                caminhos_imagens=[f"train_{i}" for i in range(len(X_train))],
                n_aumentos=5,
                modo='treino'
            )
            
            dataset_val = DatasetSolarAumentado(
                imagens=X_val,
                labels=y_val,
                caminhos_imagens=[f"val_{i}" for i in range(len(X_val))],
                n_aumentos=1,
                modo='val'
            )
            
            loader_treino = DataLoader(
                dataset_treino,
                batch_size=self.config_m1['tamanho_lote'],
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
            
            loader_val = DataLoader(
                dataset_val,
                batch_size=self.config_m1['tamanho_lote_grande'],
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
            
            # Criar modelo novo para este fold
            modelo = ClassificadorSujidade(num_classes=2)
            modelo.modelo = modelo.modelo.to(self.dispositivo)
            
            # Warm-up MPS
            if self.dispositivo == 'mps':
                self.otimizador_m1.warmup_mps(modelo.modelo)
            
            # Treinar
            print("   Iniciando treinamento...")
            inicio = time.time()
            
            historico = self._treinar_fold(modelo, loader_treino, loader_val)
            
            tempo_treino = time.time() - inicio
            
            # Avaliar
            print("   Avaliando modelo...")
            avaliador = AvaliadorModelo()
            metricas = avaliador.avaliar(modelo.modelo, loader_val, y_val, self.dispositivo)
            metricas['tempo_treino'] = tempo_treino
            metricas['fold'] = fold_idx + 1
            metricas['historico'] = historico
            
            self.resultados['folds'].append(metricas)
            
            # Salvar modelo deste fold
            torch.save(
                modelo.modelo.state_dict(), 
                self.output_dir / f'modelo_fold{fold_idx+1}.pth'
            )
            
            print(f"   Fold {fold_idx+1} concluído")
            print(f"      Acurácia: {metricas['accuracy']:.2%}")
            print(f"      Tempo: {tempo_treino/60:.1f} min")
            
            # Limpar memória
            del modelo
            if self.dispositivo == 'mps':
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
        
        # Agregar resultados
        avaliador = AvaliadorModelo()
        self.resultados['metricas_agregadas'] = avaliador.agregar_metricas(self.resultados['folds'])
    
    def _treinar_fold(self, modelo, loader_treino, loader_val):
        """Treina um fold com gradient accumulation"""
        
        criterio = nn.CrossEntropyLoss()
        otimizador = optim.Adam(
            filter(lambda p: p.requires_grad, modelo.modelo.parameters()),
            lr=0.001,
            weight_decay=1e-4
        )
        
        accum_steps = self.config_m1['gradient_accumulation_steps']
        num_epocas = 10
        
        historico = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        melhor_val_acc = 0.0
        
        for epoca in range(num_epocas):
            # Treino
            modelo.modelo.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            otimizador.zero_grad()
            
            for batch_idx, batch in enumerate(loader_treino):
                if isinstance(batch, dict):
                    imagens = batch['image']
                    labels_batch = batch['label']
                else:
                    imagens, labels_batch = batch
                imagens = imagens.to(self.dispositivo)
                labels_batch = labels_batch.to(self.dispositivo)
                
                # Forward
                outputs = modelo.modelo(imagens)
                loss = criterio(outputs, labels_batch)
                
                # Backward com acumulação
                loss = loss / accum_steps
                loss.backward()
                
                # Atualiza pesos apenas a cada N batches
                if (batch_idx + 1) % accum_steps == 0:
                    otimizador.step()
                    otimizador.zero_grad()
                
                # Métricas
                train_loss += loss.item() * accum_steps
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
                
                # Limpar cache periodicamente
                if (batch_idx + 1) % self.config_m1['empty_cache_frequency'] == 0:
                    if self.dispositivo == 'mps':
                        torch.mps.empty_cache()
            
            train_acc = train_correct / train_total
            train_loss = train_loss / len(loader_treino)
            
            # Validação
            modelo.modelo.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in loader_val:
                    if isinstance(batch, dict):
                        imagens = batch['image']
                        labels_batch = batch['label']
                    else:
                        imagens, labels_batch = batch
                    imagens = imagens.to(self.dispositivo)
                    labels_batch = labels_batch.to(self.dispositivo)
                    
                    outputs = modelo.modelo(imagens)
                    loss = criterio(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
            
            val_acc = val_correct / val_total
            val_loss = val_loss / len(loader_val)
            
            # Salvar histórico
            historico['train_loss'].append(train_loss)
            historico['train_acc'].append(train_acc)
            historico['val_loss'].append(val_loss)
            historico['val_acc'].append(val_acc)
            
            # Log em toda época
            print(f"      Época {epoca+1}/{num_epocas} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
            
            # Salvar melhor modelo
            if val_acc > melhor_val_acc:
                melhor_val_acc = val_acc
        
        return historico
    
    def _gerar_visualizacoes(self):
        """Gera todas as visualizações"""
        
        gerador = GeradorVisualizacoes(self.output_dir)
        
        gerador.gerar_matriz_confusao(self.resultados['folds'])
        gerador.gerar_curvas_treinamento(self.resultados['folds'][0]['historico'])
        gerador.gerar_comparacao_folds(self.resultados['folds'])
    
    def _gerar_documentacao(self):
        """Gera documentação LaTeX"""
        
        gerador = GeradorDocumentacaoTCC(self.output_dir)
        
        gerador.gerar_todas_tabelas(self.resultados)
        gerador.salvar_json_completo(self.resultados)
        gerador.gerar_resumo_markdown(self.resultados)


if __name__ == '__main__':
    print("\nIniciando treinamento do modelo final...")
    print("Certifique-se de ter organizado as imagens em:")
    print("   backend/dados/modulos_limpos_sujos/limpo/")
    print("   backend/dados/modulos_limpos_sujos/sujo/\n")
    
    try:
        # Só aguarda ENTER se houver terminal interativo disponível
        if sys.stdin is not None and sys.stdin.isatty():
            input("Pressione ENTER para continuar ou Ctrl+C para cancelar...")
    except EOFError:
        # Em execuções não interativas (por exemplo, IDE/automatizado), segue sem pausa
        pass
    
    treinador = TreinadorTCC()
    treinador.executar_treinamento_completo()
