#!/usr/bin/env python3
"""
Script para analisar resultados do treinamento e sugerir melhorias
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_training_report(report_path: str):
    """Analisa o relatório de treinamento e fornece insights."""
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("=" * 80)
    print("📊 ANÁLISE DE RESULTADOS DO TREINAMENTO")
    print("=" * 80)
    
    # Resumo básico
    resumo = report['resumo']
    print(f"\n🔧 Configuração:")
    print(f"  Modelo: {resumo['modelo_base']}")
    print(f"  Dataset: {resumo['dataset']}")
    print(f"  Épocas treinadas: {resumo['epocas_treinadas']}")
    print(f"  Tempo total: {resumo['tempo_total_seg']/3600:.2f}h")
    
    # Métricas finais
    metricas = report['metricas_finais']
    print(f"\n📈 Métricas Finais:")
    print(f"  Train Loss: {metricas['train_loss']:.4f}")
    print(f"  Train Acc: {metricas['train_acc']:.4f} ({metricas['train_acc']*100:.2f}%)")
    print(f"  Val Loss: {metricas['val_loss']:.4f}")
    print(f"  Val Acc: {metricas['val_acc']:.4f} ({metricas['val_acc']*100:.2f}%)")
    
    # Métricas de teste
    if 'metricas_teste' in report:
        teste = report['metricas_teste']
        print(f"\n🎯 Métricas de Teste:")
        print(f"  Accuracy: {teste['accuracy']:.4f} ({teste['accuracy']*100:.2f}%)")
        
        print(f"\n  Por classe:")
        for classe, metrics in teste.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"    {classe}:")
                print(f"      Precision: {metrics['precision']:.4f}")
                print(f"      Recall: {metrics['recall']:.4f}")
                print(f"      F1-Score: {metrics['f1-score']:.4f}")
                print(f"      Support: {int(metrics['support'])}")
    
    # Análise de convergência
    historico = report['historico_completo']
    train_losses = historico['train_loss']
    val_losses = historico['val_loss']
    
    print(f"\n📉 Análise de Convergência:")
    
    # Verificar overfitting
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    gap = final_val_loss - final_train_loss
    
    if gap > 0.5:
        print(f"  ⚠️ OVERFITTING DETECTADO (gap: {gap:.4f})")
        print(f"     Sugestões:")
        print(f"     - Aumentar dropout")
        print(f"     - Adicionar mais augmentation")
        print(f"     - Reduzir complexidade do modelo")
    elif gap > 0.2:
        print(f"  ⚡ Leve overfitting (gap: {gap:.4f})")
        print(f"     Modelo pode se beneficiar de regularização adicional")
    else:
        print(f"  ✅ Boa generalização (gap: {gap:.4f})")
    
    # Verificar se treinou o suficiente
    if len(val_losses) >= 3:
        last_3_val = val_losses[-3:]
        if all(last_3_val[i] >= last_3_val[i+1] for i in range(len(last_3_val)-1)):
            print(f"  ✅ Convergência alcançada (val_loss estabilizou)")
        else:
            print(f"  ⚡ Modelo ainda pode melhorar com mais épocas")
    
    # Sugestões baseadas na acurácia
    val_acc = metricas['val_acc']
    print(f"\n💡 Recomendações:")
    
    if val_acc < 0.70:
        print(f"  ❌ Acurácia baixa ({val_acc*100:.1f}%)")
        print(f"     Sugestões:")
        print(f"     1. Verificar qualidade dos dados (labels corretos?)")
        print(f"     2. Aumentar complexidade do modelo (EfficientNet-B5, ResNet50)")
        print(f"     3. Treinar por mais épocas (50-100)")
        print(f"     4. Ajustar learning rate (tentar 0.0001 ou 0.0005)")
        print(f"     5. Aumentar augmentation")
    elif val_acc < 0.85:
        print(f"  ⚡ Acurácia moderada ({val_acc*100:.1f}%)")
        print(f"     Sugestões:")
        print(f"     1. Testar modelo maior (EfficientNet-B5)")
        print(f"     2. Fine-tuning: descongelar mais camadas")
        print(f"     3. Aumentar épocas de treinamento")
        print(f"     4. Ensemble de modelos")
    elif val_acc < 0.95:
        print(f"  ✅ Boa acurácia ({val_acc*100:.1f}%)")
        print(f"     Sugestões para melhorar:")
        print(f"     1. Fine-tuning adicional")
        print(f"     2. Ensemble de modelos")
        print(f"     3. Test-time augmentation")
    else:
        print(f"  🎉 Excelente acurácia ({val_acc*100:.1f}%)")
        print(f"     Modelo pronto para produção!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analyze_results.py <caminho_para_relatorio.json>")
        sys.exit(1)
    
    report_path = sys.argv[1]
    if not Path(report_path).exists():
        print(f"Erro: Arquivo não encontrado: {report_path}")
        sys.exit(1)
    
    analyze_training_report(report_path)
