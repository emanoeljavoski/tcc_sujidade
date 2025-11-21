#!/usr/bin/env python3
"""Script simples para download rápido de um dataset do Roboflow.

Baixa o dataset "Aerial Solar Panels" (Brad Dwyer), com 53 imagens
já organizadas em formato YOLO.
"""

import sys
import os

def main():
    print("=" * 80)
    print("DOWNLOAD RÁPIDO - ROBOFLOW DATASET")
    print("=" * 80)
    
    # Verificar se roboflow está instalado
    try:
        from roboflow import Roboflow
    except ImportError:
        print("\nBiblioteca 'roboflow' não encontrada.")
        print("\nSolução sugerida:")
        print("   pip install roboflow")
        return False
    
    # Sua API key
    api_key = "q1tjW7hVYDHUYgwzJbSt"
    
    # Destino
    destino = r"F:\datasets_publicos_rgb\aereos_drone\aerial_solar_panels_brad"
    
    print(f"\nDataset: Aerial Solar Panels (Brad Dwyer)")
    print(f"Imagens: 53 (drone DJI Mavic Air 2)")
    print(f"Destino: {destino}")
    print("\nBaixando...\n")
    
    try:
        # Inicializar Roboflow
        rf = Roboflow(api_key=api_key)
        
        # Baixar dataset
        project = rf.workspace("brad-dwyer").project("aerial-solar-panels")
        dataset = project.version(3).download("yolov8", location=destino)
        
        print("\n" + "=" * 80)
        print("DOWNLOAD CONCLUÍDO")
        print("=" * 80)
        print(f"\nDataset salvo em: {destino}")
        print(f"Arquivo YAML: {destino}\\data.yaml")
        
        # Verificar estrutura
        data_yaml = os.path.join(destino, "data.yaml")
        if os.path.exists(data_yaml):
            print("\nEstrutura YOLO válida encontrada.")
            print("\nCaminho do dataset gerado:")
            print(f"   {data_yaml}")
            return data_yaml
        else:
            print(f"\ndata.yaml não encontrado em {destino}")
            return False
            
    except Exception as e:
        print(f"\nErro durante download: {e}")
        print("\nPossíveis soluções:")
        print("   1. Verificar conexão com internet")
        print("   2. Verificar se a API key está correta")
        print("   3. Tentar novamente")
        return False

if __name__ == "__main__":
    resultado = main()
    if resultado:
        print("\nSUCESSO. Caminho do dataset:")
        print(f"   {resultado}")
        sys.exit(0)
    else:
        print("\nFalha no download")
        sys.exit(1)
