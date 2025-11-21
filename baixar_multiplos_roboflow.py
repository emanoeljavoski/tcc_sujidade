#!/usr/bin/env python3
"""Script para baixar múltiplos datasets de painéis solares do Roboflow.

Baixa 9 datasets em formato YOLO a partir do Roboflow, organizando-os
em subpastas de `F:/datasets_publicos_rgb`.

Uso sugerido:
    python baixar_multiplos_roboflow.py
"""

from roboflow import Roboflow
from pathlib import Path
from datetime import datetime

class BaixadorRoboflow:
    """Baixa múltiplos datasets do Roboflow."""
    
    def __init__(self, api_key: str = "q1tjW7hVYDHUYgwzJbSt"):
        """
        Inicializa baixador.
        
        Args:
            api_key: Chave da API do Roboflow
        """
        self.api_key = api_key
        self.base_path = Path("F:/datasets_publicos_rgb")
        self.datasets_baixados = []
        self.datasets_falhados = []
    
    def baixar_todos(self):
        """Baixa todos os 9 datasets."""
        print("=" * 80)
        print("BAIXAR MÚLTIPLOS DATASETS DO ROBOFLOW")
        print("=" * 80)
        print(f"\nInício: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Destino: {self.base_path}")
        print("\nTempo estimado: 30-45 minutos")
        print("Espaço necessário: ~2-3 GB")
        
        input("\nPressione ENTER para começar...")
        
        # Inicializar Roboflow
        try:
            rf = Roboflow(api_key=self.api_key)
            print("\nConexão com Roboflow estabelecida")
        except Exception as e:
            print(f"\nErro ao conectar com Roboflow: {e}")
            return False
        
        # Lista de datasets
        datasets = [
            {
                'nome': 'Solar PV Maintenance Combined',
                'workspace': 'solar-pv-maintenance-drone-monitoring',
                'project': 'solar-pv-maintenance-combined-dataset',
                'version': 1,
                'destino': 'aereos_drone/solar_pv_maintenance_combined',
                'imagens': 2474,
                'prioridade': 5
            },
            {
                'nome': 'Aerial Solar Panels (Brad Dwyer)',
                'workspace': 'brad-dwyer',
                'project': 'aerial-solar-panels',
                'version': 3,
                'destino': 'aereos_drone/aerial_solar_panels_brad',
                'imagens': 53,
                'prioridade': 4
            },
            {
                'nome': 'Soiling Detection (Ammar)',
                'workspace': 'ammar-jabar-xbqcb',
                'project': 'soiling-detection',
                'version': 2,
                'destino': 'aereos_drone/soiling_detection_ammar',
                'imagens': 490,
                'prioridade': 4
            },
            {
                'nome': 'Solar Detection (Tagus Drone)',
                'workspace': 'tagus-drone',
                'project': 'solar-detection-pz28t',
                'version': 1,
                'destino': 'aereos_drone/solar_detection_tagus',
                'imagens': 96,
                'prioridade': 4
            },
            {
                'nome': 'Detection of Soiling (TCS)',
                'workspace': 'tcs-research-and-innovations-4dmu1',
                'project': 'detection-of-soiling-on-solar-panels',
                'version': 7,
                'destino': 'solo_binario/detection_soiling_tcs',
                'imagens': 185,
                'prioridade': 4
            },
            {
                'nome': 'Solar Panels Clean/Dusty (YoloModel)',
                'workspace': 'yolomodel-yp9un',
                'project': 'solar-panels-4uetb',
                'version': 1,
                'destino': 'solo_binario/solar_panels_yolomodel',
                'imagens': 44,
                'prioridade': 3
            },
            {
                'nome': 'Bird Drop Anomalies',
                'workspace': 'solar-panel-anomalies-frghv',
                'project': 'solar-panel-anomalies-bird-drop',
                'version': 1,
                'destino': 'aereos_drone/bird_drop_anomalies',
                'imagens': 120,
                'prioridade': 4
            },
            {
                'nome': 'Solar Panel Combine (Zindi)',
                'workspace': 'zindi-jwuva',
                'project': 'solar_panel_combine-fpprp',
                'version': 1,
                'destino': 'aereos_drone/solar_panel_combine_zindi',
                'imagens': 1039,
                'prioridade': 4
            },
            {
                'nome': 'Solar Panels RGB (Earthbook)',
                'workspace': 'earthbook-zdvbx',
                'project': 'solar-panels-rgb',
                'version': 1,
                'destino': 'solo_binario/solar_panels_rgb_earthbook',
                'imagens': 374,
                'prioridade': 4
            }
        ]
        
        # Baixar cada dataset
        total_datasets = len(datasets)
        
        for i, dataset_info in enumerate(datasets, 1):
            print("\n" + "=" * 80)
            print(f"DATASET {i}/{total_datasets}: {dataset_info['nome']}")
            print("=" * 80)
            print(f"   Imagens: {dataset_info['imagens']}")
            print(f"   Destino: {self.base_path / dataset_info['destino']}")
            
            sucesso = self._baixar_dataset(rf, dataset_info)
            
            if sucesso:
                self.datasets_baixados.append(dataset_info['nome'])
                print(f"   {dataset_info['nome']} baixado com sucesso.")
            else:
                self.datasets_falhados.append(dataset_info['nome'])
                print(f"   Falha ao baixar {dataset_info['nome']}")
        
        # Resumo final
        self._exibir_resumo()
        
        return len(self.datasets_baixados) > 0
    
    def _baixar_dataset(self, rf: Roboflow, info: dict) -> bool:
        """
        Baixa um dataset específico.
        
        Args:
            rf: Instância do Roboflow
            info: Informações do dataset
        
        Returns:
            True se sucesso, False caso contrário
        """
        destino_completo = self.base_path / info['destino']
        data_yaml = destino_completo / "data.yaml"
        
        # Verificar se já existe
        if data_yaml.exists():
            print("   Já existe - pulando")
            self.datasets_baixados.append(info['nome'])
            return True
        
        # Criar pasta de destino
        destino_completo.mkdir(parents=True, exist_ok=True)
        
        try:
            print("   Baixando...")
            
            # Acessar workspace e projeto
            project = rf.workspace(info['workspace']).project(info['project'])
            
            # Baixar
            dataset = project.version(info['version']).download(
                "yolov8",
                location=str(destino_completo)
            )
            
            # Verificar se baixou
            if data_yaml.exists():
                return True
            else:
                print("   Download completou, mas data.yaml não foi encontrado")
                return False
                
        except Exception as e:
            print(f"   Erro: {e}")
            return False
    
    def _exibir_resumo(self):
        """Exibe resumo final do download."""
        print("\n" + "=" * 80)
        print("RESUMO FINAL")
        print("=" * 80)
        
        print(f"\nDatasets baixados: {len(self.datasets_baixados)}")
        for nome in self.datasets_baixados:
            print(f"   - {nome}")
        
        if self.datasets_falhados:
            print(f"\nDatasets que falharam: {len(self.datasets_falhados)}")
            for nome in self.datasets_falhados:
                print(f"   - {nome}")
        
        print(f"\nTérmino: {datetime.now().strftime('%H:%M:%S')}")
        
        if self.datasets_baixados:
            print("\nDatasets baixados podem ser utilizados em scripts de consolidação e treino conforme a necessidade do projeto.")

def main():
    """Função principal."""
    baixador = BaixadorRoboflow()
    
    try:
        sucesso = baixador.baixar_todos()
        return 0 if sucesso else 1
    except KeyboardInterrupt:
        print("\n\nDownload cancelado pelo usuário")
        return 1
    except Exception as e:
        print(f"\n\nErro inesperado: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
