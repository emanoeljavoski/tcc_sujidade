#!/usr/bin/env python3
"""
Script para monitorar o treinamento do classificador via API
"""
import requests
import time
import json
from datetime import datetime

API_URL = "http://localhost:8000/status-treinamento-classificador"
CHECK_INTERVAL = 30  # segundos

def get_status():
    try:
        response = requests.get(API_URL, timeout=5)
        return response.json()
    except Exception as e:
        print(f"Erro ao obter status: {e}")
        return None

def format_time(seconds):
    if seconds == 0:
        return "calculando..."
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h{minutes}m"

def main():
    print("🔍 Monitorando treinamento do classificador...")
    print("=" * 80)
    
    last_epoch = 0
    
    while True:
        status = get_status()
        
        if not status:
            time.sleep(CHECK_INTERVAL)
            continue
        
        if not status['treinando']:
            print("\n✅ Treinamento concluído!")
            print(f"Métricas finais:")
            print(f"  Train Loss: {status['metricas']['train_loss']:.4f}")
            print(f"  Train Acc: {status['metricas']['train_acc']:.4f}")
            print(f"  Val Loss: {status['metricas']['val_loss']:.4f}")
            print(f"  Val Acc: {status['metricas']['val_acc']:.4f}")
            break
        
        current_epoch = status['epoca_atual']
        total_epochs = status['total_epocas']
        progress = status['progresso']
        
        # Avisar quando completar uma nova época
        if current_epoch > last_epoch:
            metrics = status['metricas']
            tempo_restante = format_time(status.get('tempo_restante_seg', 0))
            
            print(f"\n📊 Época {current_epoch}/{total_epochs} ({progress}%)")
            print(f"  Train Loss: {metrics['train_loss']:.4f} | Train Acc: {metrics['train_acc']:.4f}")
            print(f"  Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['val_acc']:.4f}")
            print(f"  Tempo restante estimado: {tempo_restante}")
            
            last_epoch = current_epoch
        else:
            # Atualização silenciosa
            print(f"⏳ Época {current_epoch}/{total_epochs} em andamento... ({datetime.now().strftime('%H:%M:%S')})", end='\r')
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
