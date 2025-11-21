"""
Script de Treinamento 5-Fold OTIMIZADO para DELL com CUDA
Criado especificamente para resolver problemas de:
- Pickle/multiprocessing no Windows
- Falta de feedback visual durante treino
- Performance máxima em GPU NVIDIA

Configuração rápida mas realista:
- 10 épocas por fold
- Augmentation 5x
- Logs a cada 50 batches
- Num_workers=0 (evita pickle no Windows)
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import sys

# Adicionar path do backend
sys.path.append(str(Path(__file__).parent))

from aplicacao.modelos.classificador_sujidade import ClassificadorSujidade
from aplicacao.aumento_dados import DatasetSolarAumentado

print("=" * 80)
print("TREINAMENTO 5-FOLD OTIMIZADO PARA DELL + CUDA")
print("=" * 80)
print()

# Verificar CUDA
if not torch.cuda.is_available():
    print("ERRO: CUDA não está disponível.")
    print("   Verifique se há GPU NVIDIA e se o PyTorch com CUDA está instalado.")
    sys.exit(1)

print(f"CUDA disponível: {torch.cuda.get_device_name(0)}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# Configurações otimizadas para DELL com CUDA
BATCH_SIZE = 16  # Aumentado para melhor uso da GPU
NUM_EPOCAS = 10
N_AUGMENTATIONS = 5
LEARNING_RATE = 0.001
DEVICE = 'cuda'
LOG_A_CADA_N_BATCHES = 50  # Log frequente para ver progresso

# Dataset
DATASET_PATH = r'F:\dataset_2classes_meus_public_50_50'

# Output
OUTPUT_DIR = Path('outputs/treinamento_5fold_dell')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset: {DATASET_PATH}")
print("Configuração:")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - Épocas por fold: {NUM_EPOCAS}")
print(f"   - Augmentation: {N_AUGMENTATIONS}x")
print(f"   - Learning rate: {LEARNING_RATE}")
print(f"   - Device: {DEVICE}")
print()


def carregar_dataset(dataset_path):
    """Carrega imagens do dataset 50/50"""
    from PIL import Image
    
    imagens = []
    labels = []
    
    print("Carregando dataset...")
    
    for split in ['train', 'val', 'test']:
        for label_idx, classe in enumerate(['limpo', 'sujo']):
            pasta = Path(dataset_path) / split / classe
            if not pasta.exists():
                continue
            
            arquivos = list(pasta.glob('*.jpg')) + list(pasta.glob('*.png')) + list(pasta.glob('*.jpeg'))
            
            for img_path in arquivos:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((224, 224))
                    imagens.append(np.array(img))
                    labels.append(label_idx)
                except Exception as e:
                    print(f"   Erro ao carregar {img_path}: {e}")
    
    imagens = np.array(imagens)
    labels = np.array(labels)
    
    print("Dataset carregado:")
    print(f"   Total: {len(imagens)} imagens")
    print(f"   Limpo: {np.sum(labels == 0)}")
    print(f"   Sujo: {np.sum(labels == 1)}")
    print()
    
    return imagens, labels


def treinar_fold(fold_num, X_train, y_train, X_val, y_val):
    """Treina um único fold com logs detalhados"""
    
    print(f"\n{'=' * 70}")
    print(f"FOLD {fold_num}/5")
    print(f"{'=' * 70}")
    print(f"   Treino: {len(X_train)} | Validação: {len(X_val)}")
    
    # Criar datasets
    dataset_train = DatasetSolarAumentado(
        imagens=X_train,
        labels=y_train,
        caminhos_imagens=[f"train_{i}" for i in range(len(X_train))],
        n_aumentos=N_AUGMENTATIONS,
        modo='treino'
    )
    
    dataset_val = DatasetSolarAumentado(
        imagens=X_val,
        labels=y_val,
        caminhos_imagens=[f"val_{i}" for i in range(len(X_val))],
        n_aumentos=1,
        modo='val'
    )
    
    # DataLoaders (num_workers=0 para evitar pickle no Windows)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"   Dataset treino expandido: {len(dataset_train)} amostras")
    print(f"   Batches por época: {len(loader_train)}")
    print()
    
    # Criar modelo
    modelo = ClassificadorSujidade(num_classes=2)
    modelo.modelo = modelo.modelo.to(DEVICE)
    
    # Otimizador e loss
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(
        filter(lambda p: p.requires_grad, modelo.modelo.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Histórico
    historico = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    melhor_val_acc = 0.0
    inicio_fold = time.time()
    
    # Loop de treinamento
    for epoca in range(NUM_EPOCAS):
        inicio_epoca = time.time()
        
        # TREINO
        modelo.modelo.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"   Época {epoca+1}/{NUM_EPOCAS}")
        
        for batch_idx, batch in enumerate(loader_train):
            # Extrair imagens e labels (compatível com dict ou tupla)
            if isinstance(batch, dict):
                imagens = batch['image']
                labels_batch = batch['label']
            else:
                imagens, labels_batch = batch
            
            imagens = imagens.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            # Forward
            otimizador.zero_grad()
            outputs = modelo.modelo(imagens)
            loss = criterio(outputs, labels_batch)
            
            # Backward
            loss.backward()
            otimizador.step()
            
            # Métricas
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()
            
            # Log a cada N batches
            if (batch_idx + 1) % LOG_A_CADA_N_BATCHES == 0:
                acc_atual = 100.0 * train_correct / train_total
                print(
                    f"      Batch {batch_idx+1}/{len(loader_train)} - "
                    f"Loss: {loss.item():.4f}, Acc: {acc_atual:.1f}%"
                )
        
        train_acc = train_correct / train_total
        train_loss = train_loss / len(loader_train)
        
        # VALIDAÇÃO
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
                
                imagens = imagens.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)
                
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
        
        # Melhor modelo
        if val_acc > melhor_val_acc:
            melhor_val_acc = val_acc
        
        tempo_epoca = time.time() - inicio_epoca
        
        print(f"      Época {epoca+1} concluída em {tempo_epoca:.1f}s")
        print(f"         Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
        print(f"         Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
        print()
        
        # Limpar cache CUDA
        torch.cuda.empty_cache()
    
    tempo_fold = time.time() - inicio_fold
    
    # Avaliar no conjunto de validação final
    modelo.modelo.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for batch in loader_val:
            if isinstance(batch, dict):
                imagens = batch['image']
                labels_batch = batch['label']
            else:
                imagens, labels_batch = batch
            
            imagens = imagens.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            outputs = modelo.modelo(imagens)
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels_batch.cpu().numpy())
    
    # Métricas finais
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"   FOLD {fold_num} concluído")
    print(f"      Tempo total: {tempo_fold/60:.1f} min")
    print(f"      Acurácia: {acc:.2%}")
    print(f"      Precision: {precision:.2%}")
    print(f"      Recall: {recall:.2%}")
    print(f"      F1-Score: {f1:.2%}")
    print()
    
    # Salvar modelo
    torch.save(
        modelo.modelo.state_dict(),
        OUTPUT_DIR / f'modelo_fold{fold_num}.pth'
    )
    
    return {
        'fold': fold_num,
        'tempo_treino': tempo_fold,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'historico': historico
    }


def main():
    """Pipeline principal"""
    
    inicio_total = time.time()
    
    # Carregar dataset
    imagens, labels = carregar_dataset(DATASET_PATH)
    
    # 5-Fold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    resultados = {
        'inicio': datetime.now().isoformat(),
        'configuracao': {
            'arquitetura': 'EfficientNet-B4',
            'num_classes': 2,
            'batch_size': BATCH_SIZE,
            'num_epocas': NUM_EPOCAS,
            'learning_rate': LEARNING_RATE,
            'data_augmentation': f'{N_AUGMENTATIONS}x',
            'n_folds': 5,
            'device': DEVICE
        },
        'folds': []
    }
    
    print("Iniciando validação cruzada 5-fold...")
    print()
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(imagens, labels)):
        X_train, X_val = imagens[train_idx], imagens[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        resultado_fold = treinar_fold(fold_idx + 1, X_train, y_train, X_val, y_val)
        resultados['folds'].append(resultado_fold)
        
        # Salvar resultados parciais
        with open(OUTPUT_DIR / 'resultados_parciais.json', 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    tempo_total = time.time() - inicio_total
    
    # Agregar resultados
    acuracias = [f['accuracy'] for f in resultados['folds']]
    tempos = [f['tempo_treino'] for f in resultados['folds']]
    
    resultados['metricas_agregadas'] = {
        'acuracia_media': float(np.mean(acuracias)),
        'acuracia_std': float(np.std(acuracias)),
        'acuracia_min': float(np.min(acuracias)),
        'acuracia_max': float(np.max(acuracias)),
        'tempo_total_segundos': tempo_total,
        'tempo_total_horas': tempo_total / 3600,
        'tempo_medio_por_fold_min': float(np.mean(tempos) / 60)
    }
    
    # Salvar resultados finais
    with open(OUTPUT_DIR / 'resultados_completos.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("TREINAMENTO 5-FOLD CONCLUÍDO COM SUCESSO")
    print("=" * 80)
    print()
    print("Resultados finais:")
    print(f"   Tempo total: {tempo_total/3600:.2f} horas")
    print(f"   Tempo médio por fold: {np.mean(tempos)/60:.1f} min")
    print(f"   Acurácia média: {resultados['metricas_agregadas']['acuracia_media']:.2%}")
    print(f"   Desvio padrão: {resultados['metricas_agregadas']['acuracia_std']:.2%}")
    print(
        f"   Acurácia mín/máx: {resultados['metricas_agregadas']['acuracia_min']:.2%} / "
        f"{resultados['metricas_agregadas']['acuracia_max']:.2%}"
    )
    print()
    print(f"Resultados salvos em: {OUTPUT_DIR}")
    print("   - resultados_completos.json")
    print("   - modelo_fold1.pth ... modelo_fold5.pth")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido pelo usuário (Ctrl+C)")
        print("   Resultados parciais salvos em outputs/treinamento_5fold_dell/")
    except Exception as e:
        print("\n\nErro durante o treinamento:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
