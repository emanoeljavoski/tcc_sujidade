#!/usr/bin/env python3
"""Treinamento YOLO11 com ajustes para contornar erros do OpenCV.

Este script força o treino do YOLO11 mesmo quando ocorrem erros em cv2,
usando monkeypatch e alternativas de execução para permitir que o
treinamento ocorra.

Uso em linha de comando:
   python treinar_yolo11_contornando_erro.py
   python treinar_yolo11_contornando_erro.py F:\\caminho\\para\\data.yaml
"""

import sys
import os
from pathlib import Path

def monkeypatch_cv2():
    """Contorna problemas do OpenCV antes de importar YOLO."""
    print("Aplicando correções para OpenCV...")
    
    try:
        import cv2
        
        # Adicionar atributos que podem estar faltando
        if not hasattr(cv2, 'IMREAD_COLOR'):
            cv2.IMREAD_COLOR = 1
            print("   cv2.IMREAD_COLOR adicionado")
        
        if not hasattr(cv2, 'IMREAD_GRAYSCALE'):
            cv2.IMREAD_GRAYSCALE = 0
            print("   cv2.IMREAD_GRAYSCALE adicionado")
        
        if not hasattr(cv2, 'IMREAD_UNCHANGED'):
            cv2.IMREAD_UNCHANGED = -1
            print("   cv2.IMREAD_UNCHANGED adicionado")
        
        # Adicionar funções que podem estar faltando
        if not hasattr(cv2, 'imshow'):
            def imshow_dummy(winname, mat):
                pass
            cv2.imshow = imshow_dummy
            print("   cv2.imshow dummy adicionado")
        
        if not hasattr(cv2, 'waitKey'):
            def waitKey_dummy(delay=0):
                return -1
            cv2.waitKey = waitKey_dummy
            print("   cv2.waitKey dummy adicionado")
        
        if not hasattr(cv2, 'destroyAllWindows'):
            def destroyAllWindows_dummy():
                pass
            cv2.destroyAllWindows = destroyAllWindows_dummy
            print("   cv2.destroyAllWindows dummy adicionado")
        
        print("OpenCV corrigido com sucesso.\n")
        return True
        
    except ImportError:
        print("OpenCV não instalado. Tentando continuar...\n")
        return False
    except Exception as e:
        print(f"Aviso: erro ao corrigir OpenCV: {e}")
        print("   Tentando continuar mesmo assim...\n")
        return False

def encontrar_dataset_automatico():
    """Procura automaticamente por datasets YOLO."""
    print("Procurando datasets YOLO automaticamente...")
    
    locais_busca = [
        Path("F:/datasets_publicos_rgb/aereos_drone/aerial_solar_panels_brad"),
        Path("F:/datasets_publicos_rgb/aereos_drone/solar_pv_maintenance_combined"),
        Path("F:/datasets_publicos/Aerial-Solar-Panels-13"),
        Path("F:/dataset_yolo_detector_modulos"),
    ]
    
    for local in locais_busca:
        data_yaml = local / "data.yaml"
        if data_yaml.exists():
            print(f"   Encontrado: {data_yaml}")
            return str(data_yaml)
    
    print("   Nenhum dataset encontrado automaticamente")
    return None

def treinar_yolo11(caminho_yaml: str, epocas: int = 50):
    """
    Treina YOLO11 com o dataset especificado.
    
    Args:
        caminho_yaml: Caminho para o data.yaml do dataset
        epocas: Número de épocas (padrão: 50)
    """
    print("=" * 80)
    print("INICIANDO TREINO YOLO11")
    print("=" * 80)
    
    print("\nConfiguração:")
    print(f"   Dataset: {caminho_yaml}")
    print(f"   Épocas: {epocas}")
    print(f"   Modelo: yolo11n.pt (nano)")
    print(f"   Batch: 16")
    print(f"   Image Size: 640")
    
    # Verificar se o dataset existe
    if not Path(caminho_yaml).exists():
        print("\nERRO: dataset não encontrado.")
        print(f"   Caminho: {caminho_yaml}")
        return False
    
    try:
        # Tentar usar o TreinadorDetector
        print("\nTentando usar TreinadorDetector...")
        
        # Adicionar o backend ao path
        backend_path = Path(__file__).parent.parent / "backend"
        if backend_path.exists():
            sys.path.insert(0, str(backend_path))
        
        from aplicacao.modelos.treinamento_detector import TreinadorDetector
        
        # Criar diretório de saída
        diretorio_saida = Path("F:/modelos_salvos/detector_yolo11_roboflow")
        diretorio_saida.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaída: {diretorio_saida}")
        print("\nIniciando treino...")
        print("   (Isso pode demorar 15-30 minutos)")
        print("=" * 80)
        
        # Treinar
        treinador = TreinadorDetector(
            caminho_dataset_yaml=caminho_yaml,
            modelo_base="yolo11n.pt"
        )
        
        resultado = treinador.treinar(
            epocas=epocas,
            batch_size=16,
            imgsz=640,
            lr=0.001,
            patience=15,
            save_period=10,
            diretorio_saida=str(diretorio_saida)
        )
        
        print("\n" + "=" * 80)
        print("TREINO CONCLUÍDO COM SUCESSO")
        print("=" * 80)
        
        # Exibir métricas
        if 'metrics' in resultado:
            metrics = resultado['metrics']
            print("\nMÉTRICAS FINAIS:")
            print(f"   mAP@0.5: {metrics.get('mAP50', 0):.4f}")
            print(f"   mAP@0.5:0.95: {metrics.get('mAP50_95', 0):.4f}")
            print(f"   Precision: {metrics.get('precision', 0):.4f}")
            print(f"   Recall: {metrics.get('recall', 0):.4f}")
            print(f"   Box Loss: {metrics.get('box_loss', 0):.4f}")
        
        # Localização dos arquivos
        print("\nARQUIVOS GERADOS:")
        print(f"   Modelo: {diretorio_saida}/treinamento/weights/best.pt")
        print(f"   Gráficos: {diretorio_saida}/treinamento/results.png")
        print(f"   Relatório: {diretorio_saida}/relatorio_treinamento.json")
        
        return True
        
    except ImportError as e:
        print(f"\nNão foi possível usar TreinadorDetector: {e}")
        print("\nTentando método alternativo com Ultralytics direta...")
        return treinar_ultralytics_direto(caminho_yaml, epocas)
    
    except Exception as e:
        print(f"\nERRO durante treino: {e}")
        print("\nTentando método alternativo...")
        return treinar_ultralytics_direto(caminho_yaml, epocas)

def treinar_ultralytics_direto(caminho_yaml: str, epocas: int):
    """Treina usando Ultralytics direto como fallback."""
    try:
        print("\nUsando Ultralytics YOLO direto...")
        
        from ultralytics import YOLO
        
        # Criar modelo
        model = YOLO("yolo11n.pt")
        
        # Treinar
        results = model.train(
            data=caminho_yaml,
            epochs=epocas,
            imgsz=640,
            batch=16,
            lr0=0.001,
            optimizer="AdamW",
            cos_lr=True,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            project="F:/modelos_salvos/detector_yolo11_roboflow",
            name="treinamento",
            exist_ok=True,
            verbose=True
        )
        
        print("\nTreino concluído.")
        print("Resultados em: F:/modelos_salvos/detector_yolo11_roboflow/treinamento/")
        
        return True
        
    except Exception as e:
        print(f"\nERRO no método alternativo: {e}")
        return False

def main():
    """Função principal."""
    print("=" * 80)
    print("TREINO YOLO11 - CONTORNANDO PROBLEMAS DO OPENCV")
    print("=" * 80)
    
    # ETAPA 1: Corrigir OpenCV
    print("\nETAPA 1: Corrigindo OpenCV...")
    monkeypatch_cv2()
    
    # ETAPA 2: Encontrar dataset
    print("ETAPA 2: Localizando dataset...")
    
    if len(sys.argv) > 1:
        caminho_yaml = sys.argv[1]
        print(f"   Usando dataset especificado: {caminho_yaml}")
    else:
        caminho_yaml = encontrar_dataset_automatico()
        
        if not caminho_yaml:
            print("\nERRO: nenhum dataset encontrado.")
            print("   Baixe um dataset com o script baixar_roboflow_rapido.py ")
            print("   ou especifique o caminho para o arquivo data.yaml na linha de comando.")
            return False
    
    # ETAPA 3: Treinar
    print("\nETAPA 3: Treinando YOLO11...")
    
    # Perguntar número de épocas
    print("\nQuantas épocas deseja treinar?")
    print("   50 = execução mais rápida (15-20 min)")
    print("   100 = recomendado (30-45 min)")
    print("   200 = completo (1-2 horas)")
    
    try:
        epocas_input = input("\n   Digite o número de épocas [50]: ").strip()
        epocas = int(epocas_input) if epocas_input else 50
    except:
        epocas = 50
        print(f"   Usando padrão: {epocas} épocas")
    
    sucesso = treinar_yolo11(caminho_yaml, epocas)
    
    if sucesso:
        print("\n" + "=" * 80)
        print("🎉 TUDO PRONTO!")
        print("=" * 80)
        print("\n✅ Use os resultados para preencher a Tabela 4.2 do TCC")
        print("✅ Copie o results.png para a FIGURA 4.1")
        return True
    else:
        print("\n" + "=" * 80)
        print("❌ TREINO FALHOU")
        print("=" * 80)
        return False

if __name__ == "__main__":
    sucesso = main()
    sys.exit(0 if sucesso else 1)
