#!/usr/bin/env python3
"""Script utilitário para automatizar o uso do YOLO11.

Este script:
- verifica/instala dependências básicas,
- baixa um dataset de exemplo do Roboflow,
- aplica correções mínimas no OpenCV para ambientes headless,
- treina o YOLO11 em um cenário rápido de demonstração,
- gera um pequeno relatório com arquivos principais para consulta.
"""

import sys
import os
import subprocess
from pathlib import Path

class SolucaoCompletaYOLO11:
    """Solução automática completa para treinar YOLO11."""
    
    def __init__(self):
        self.etapa = 0
        self.total_etapas = 5
        self.dataset_yaml = None
        
    def print_etapa(self, titulo: str):
        """Imprime cabeçalho de etapa."""
        self.etapa += 1
        print("\n" + "=" * 80)
        print(f"ETAPA {self.etapa}/{self.total_etapas}: {titulo}")
        print("=" * 80 + "\n")
    
    def executar(self):
        """Executa solução completa."""
        print("=" * 80)
        print("SOLUÇÃO COMPLETA YOLO11 - EXECUÇÃO AUTOMÁTICA")
        print("=" * 80)
        print("\nEste script executa:")
        print("  - Verificação/instalação de dependências")
        print("  - Download de um dataset de exemplo do Roboflow (~2 min)")
        print("  - Correções básicas no OpenCV (modo headless)")
        print("  - Treinamento rápido do YOLO11 (~20-30 min)")
        print("  - Geração de um pequeno relatório para consulta")
        print("\nTempo total estimado: 20-40 minutos")
        
        input("\nPressione ENTER para começar...")
        
        # ETAPA 1: Instalar dependências
        if not self.etapa1_instalar_dependencias():
            return False
        
        # ETAPA 2: Baixar dataset
        if not self.etapa2_baixar_dataset():
            return False
        
        # ETAPA 3: Corrigir OpenCV
        if not self.etapa3_corrigir_opencv():
            return False
        
        # ETAPA 4: Treinar YOLO11
        if not self.etapa4_treinar_yolo11():
            return False
        
        # ETAPA 5: Gerar relatório
        self.etapa5_gerar_relatorio()
        
        print("\n" + "=" * 80)
        print("Execução concluída com sucesso.")
        print("=" * 80)
        
        return True
    
    def etapa1_instalar_dependencias(self):
        """Instala/verifica dependências."""
        self.print_etapa("Verificando dependências")
        
        dependencias = ['roboflow', 'ultralytics', 'opencv-python']
        
        for dep in dependencias:
            try:
                __import__(dep.replace('-', '_'))
                print(f"   {dep} já instalado")
            except ImportError:
                print(f"   Instalando {dep}...")
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', dep
                    ])
                    print(f"   {dep} instalado com sucesso")
                except Exception as e:
                    print(f"   Erro ao instalar {dep}: {e}")
                    print("   Continuando mesmo assim...")
        
        print("\nEtapa 1 concluída.")
        return True
    
    def etapa2_baixar_dataset(self):
        """Baixa dataset do Roboflow."""
        self.print_etapa("Baixando dataset do Roboflow")
        
        destino = Path("F:/datasets_publicos_rgb/aereos_drone/aerial_solar_panels_brad")
        data_yaml = destino / "data.yaml"
        
        # Verificar se já existe
        if data_yaml.exists():
            print(f"   ✅ Dataset já existe em: {destino}")
            self.dataset_yaml = str(data_yaml)
            return True
        
        print("   Dataset: Aerial Solar Panels (53 imagens)")
        print("   Baixando... (pode demorar 2-3 minutos)")
        
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key="q1tjW7hVYDHUYgwzJbSt")
            project = rf.workspace("brad-dwyer").project("aerial-solar-panels")
            dataset = project.version(3).download("yolov8", location=str(destino))
            
            if data_yaml.exists():
                print("\n   Dataset baixado com sucesso.")
                print(f"   Localização: {destino}")
                self.dataset_yaml = str(data_yaml)
                return True
            else:
                print("\n   Erro: data.yaml não encontrado após download")
                return False
                
        except Exception as e:
            print(f"\n   Erro ao baixar dataset: {e}")
            return False
    
    def etapa3_corrigir_opencv(self):
        """Aplica correções para o OpenCV."""
        self.print_etapa("Corrigindo OpenCV")
        
        try:
            import cv2
            
            # Adicionar constantes faltantes
            if not hasattr(cv2, 'IMREAD_COLOR'):
                cv2.IMREAD_COLOR = 1
                print("   cv2.IMREAD_COLOR corrigido")
            
            if not hasattr(cv2, 'IMREAD_GRAYSCALE'):
                cv2.IMREAD_GRAYSCALE = 0
                print("   cv2.IMREAD_GRAYSCALE corrigido")
            
            if not hasattr(cv2, 'IMREAD_UNCHANGED'):
                cv2.IMREAD_UNCHANGED = -1
                print("   cv2.IMREAD_UNCHANGED corrigido")
            
            # Adicionar funções dummy
            if not hasattr(cv2, 'imshow'):
                cv2.imshow = lambda *args, **kwargs: None
                print("   cv2.imshow corrigido")
            
            if not hasattr(cv2, 'waitKey'):
                cv2.waitKey = lambda *args, **kwargs: -1
                print("   cv2.waitKey corrigido")
            
            if not hasattr(cv2, 'destroyAllWindows'):
                cv2.destroyAllWindows = lambda: None
                print("   cv2.destroyAllWindows corrigido")
            
            print("\nOpenCV corrigido com sucesso.")
            return True
            
        except Exception as e:
            print(f"   Erro ao corrigir OpenCV: {e}")
            print("   Tentando continuar mesmo assim...")
            return True  # Continuar mesmo com erro
    
    def etapa4_treinar_yolo11(self):
        """Treina YOLO11."""
        self.print_etapa("Treinando YOLO11")
        
        if not self.dataset_yaml:
            print("   Dataset não encontrado!")
            return False
        
        print(f"   Dataset: {self.dataset_yaml}")
        print("   Modelo: yolo11n.pt")
        print("   Épocas: 50 (rápido)")
        print("   Batch: 16")
        print("\n   Iniciando treino...")
        print("   (isso pode levar 15-30 minutos)")
        print("   " + "-" * 76)
        
        try:
            from ultralytics import YOLO
            
            # Criar modelo
            model = YOLO("yolo11n.pt")
            
            # Treinar
            diretorio_saida = "F:/modelos_salvos/detector_yolo11_roboflow"
            
            results = model.train(  # noqa: F841 - mantido para compatibilidade com Ultralytics
                data=self.dataset_yaml,
                epochs=50,
                imgsz=640,
                batch=16,
                lr0=0.001,
                optimizer="AdamW",
                cos_lr=True,
                device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                project=diretorio_saida,
                name="treinamento",
                exist_ok=True,
                verbose=True,
                patience=15,
                save_period=10
            )
            
            print("\n   " + "-" * 76)
            print("   Treino concluído com sucesso.")
            print("\n   Resultados salvos em:")
            print(f"      {diretorio_saida}/treinamento/")
            
            return True
            
        except Exception as e:
            print(f"\n   Erro durante treino: {e}")
            return False
    
    def etapa5_gerar_relatorio(self):
        """Gera relatório final."""
        self.print_etapa("Gerando relatório para TCC")
        
        diretorio = Path("F:/modelos_salvos/detector_yolo11_roboflow/treinamento")
        
        # Verificar arquivos gerados
        arquivos_importantes = {
            'best.pt': diretorio / 'weights' / 'best.pt',
            'results.png': diretorio / 'results.png',
            'results.csv': diretorio / 'results.csv',
            'confusion_matrix.png': diretorio / 'confusion_matrix.png'
        }
        
        print("   Verificando arquivos gerados:")
        for nome, caminho in arquivos_importantes.items():
            if caminho.exists():
                print(f"      {nome} encontrado")
            else:
                print(f"      {nome} não encontrado")
        
        # Ler métricas do results.csv
        results_csv = diretorio / 'results.csv'
        if results_csv.exists():
            print("\n   Métricas para a tabela de resultados:")
            print("   " + "-" * 76)
            
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)
                
                # Pegar última linha (época final)
                ultima_epoca = df.iloc[-1]
                
                print(f"      mAP@0.5:        {ultima_epoca.get('metrics/mAP50(B)', 0):.4f}")
                print(f"      mAP@0.5:0.95:   {ultima_epoca.get('metrics/mAP50-95(B)', 0):.4f}")
                print(f"      Precision:      {ultima_epoca.get('metrics/precision(B)', 0):.4f}")
                print(f"      Recall:         {ultima_epoca.get('metrics/recall(B)', 0):.4f}")
                print(f"      Box Loss:       {ultima_epoca.get('val/box_loss', 0):.4f}")
                
            except Exception as e:
                print(f"      Erro ao ler métricas: {e}")
        
        print("\n   " + "-" * 76)
        print("\n   Próximos passos sugeridos:")
        print("      1. Copiar results.png para figuras/figura4_1_curvas_treinamento_yolo11.png")
        print("      2. Atualizar a tabela de métricas do TCC com os valores acima")
        print("      3. Utilizar best.pt como modelo treinado final")
        
        print("\nEtapa 5 concluída.")

def main():
    """Função principal."""
    solucao = SolucaoCompletaYOLO11()
    
    try:
        sucesso = solucao.executar()
        return 0 if sucesso else 1
    except KeyboardInterrupt:
        print("\n\nOperação cancelada pelo usuário.")
        return 1
    except Exception as e:
        print(f"\n\nErro inesperado: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
