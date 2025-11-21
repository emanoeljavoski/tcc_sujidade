#!/usr/bin/env python3
"""
Script para organizar datasets do Zenodo (DeepStat, BDAPPV) em formato YOLO.

Datasets do Zenodo geralmente vêm desorganizados:
- DeepStat: Imagens aéreas sem labels YOLO
- BDAPPV: Imagens de painéis sem anotações de sujidade

Este script ajuda a verificar e, se possível, organizar esses datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

class OrganizadorDatasetZenodo:
    """Organiza datasets do Zenodo para formato YOLO."""
    
    def __init__(self, caminho_zenodo: str):
        """
        Inicializa organizador.
        
        Args:
            caminho_zenodo: Caminho para pasta extraída do Zenodo
        """
        self.caminho = Path(caminho_zenodo)
        self.nome_dataset = self.caminho.name
        
    def diagnosticar(self) -> Dict[str, any]:
        """
        Diagnostica o que há no dataset do Zenodo.
        
        Returns:
            Dicionário com informações sobre o dataset
        """
        print("=" * 80)
        print(f"DIAGNÓSTICO: {self.nome_dataset}")
        print("=" * 80)
        print(f"Caminho: {self.caminho}\n")
        
        diagnostico = {
            'existe': False,
            'total_arquivos': 0,
            'imagens': [],
            'labels_yolo': [],
            'labels_coco': [],
            'metadados': [],
            'tipo_dataset': 'desconhecido',
            'conversivel_yolo': False,
            'motivo': ''
        }
        
        if not self.caminho.exists():
            print("❌ Caminho não existe!")
            diagnostico['motivo'] = "Pasta não encontrada"
            return diagnostico
        
        diagnostico['existe'] = True
        
        # Contar arquivos por tipo
        print("Análise de arquivos:")
        print("-" * 40)
        
        # Imagens
        extensoes_imagem = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']
        for ext in extensoes_imagem:
            imgs = list(self.caminho.rglob(f'*{ext}'))
            diagnostico['imagens'].extend(imgs)
        
        print(f"Imagens: {len(diagnostico['imagens'])}")
        if diagnostico['imagens']:
            print(f"   Exemplos: {[img.name for img in diagnostico['imagens'][:3]]}")
        
        # Labels YOLO (.txt)
        labels_txt = list(self.caminho.rglob('*.txt'))
        # Filtrar README e outros txts
        labels_yolo = [l for l in labels_txt if 
                      l.stem.lower() not in ['readme', 'license', 'citation']]
        diagnostico['labels_yolo'] = labels_yolo
        
        print(f"Labels YOLO (.txt): {len(labels_yolo)}")
        if labels_yolo:
            print(f"   Exemplos: {[l.name for l in labels_yolo[:3]]}")
        
        # Labels COCO (.json)
        labels_json = list(self.caminho.rglob('*.json'))
        diagnostico['labels_coco'] = labels_json
        
        print(f"Anotações COCO (.json): {len(labels_json)}")
        if labels_json:
            print(f"   Exemplos: {[j.name for j in labels_json[:3]]}")
        
        # Metadados
        metadados = list(self.caminho.rglob('*.csv')) + \
                   list(self.caminho.rglob('*.yaml')) + \
                   list(self.caminho.rglob('*.xml'))
        diagnostico['metadados'] = metadados
        
        print(f"Metadados (CSV/YAML/XML): {len(metadados)}")
        
        print("-" * 40)
        diagnostico['total_arquivos'] = len(diagnostico['imagens']) + \
                                       len(labels_yolo) + \
                                       len(labels_json) + \
                                       len(metadados)
        
        # Determinar tipo de dataset
        if 'deepstat' in self.nome_dataset.lower():
            diagnostico['tipo_dataset'] = 'DeepStat WP5'
        elif 'bdappv' in self.nome_dataset.lower():
            diagnostico['tipo_dataset'] = 'BDAPPV'
        
        # Verificar se é conversível para YOLO
        self._avaliar_convesibilidade(diagnostico)
        
        # Exibir resultado
        self._exibir_diagnostico(diagnostico)
        
        return diagnostico
    
    def _avaliar_convesibilidade(self, diagnostico: Dict) -> None:
        """Avalia se o dataset pode ser convertido para YOLO."""
        tem_imagens = len(diagnostico['imagens']) > 0
        tem_labels_yolo = len(diagnostico['labels_yolo']) > 0
        tem_labels_coco = len(diagnostico['labels_coco']) > 0
        
        if tem_imagens and tem_labels_yolo:
            # Já tem labels YOLO - só precisa organizar
            diagnostico['conversivel_yolo'] = True
            diagnostico['motivo'] = "Tem imagens e labels YOLO - só precisa organizar estrutura"
        
        elif tem_imagens and tem_labels_coco:
            # Tem labels COCO - pode converter
            diagnostico['conversivel_yolo'] = True
            diagnostico['motivo'] = "Tem imagens e labels COCO - precisa converter para YOLO"
        
        elif tem_imagens and not tem_labels_yolo and not tem_labels_coco:
            # Só imagens, sem labels
            diagnostico['conversivel_yolo'] = False
            if diagnostico['tipo_dataset'] == 'BDAPPV':
                diagnostico['motivo'] = "BDAPPV é para detecção de painéis, não classificação de sujidade. Útil para pré-treino do detector, mas não tem labels de sujidade."
            else:
                diagnostico['motivo'] = "Dataset tem imagens mas SEM labels/anotações. Não pode ser usado para treinamento supervisionado."
        
        else:
            diagnostico['conversivel_yolo'] = False
            diagnostico['motivo'] = "Dataset incompleto ou estrutura não reconhecida"
    
    def _exibir_diagnostico(self, diagnostico: Dict) -> None:
        """Exibe resumo do diagnóstico."""
        print("\n" + "=" * 80)
        print("📋 DIAGNÓSTICO FINAL")
        print("=" * 80)
        
        print(f"\n🏷️  Tipo: {diagnostico['tipo_dataset']}")
        print(f"📊 Total de arquivos: {diagnostico['total_arquivos']}")
        
        if diagnostico['conversivel_yolo']:
            print("\nCONVERSÍVEL PARA YOLO")
            print(f"   {diagnostico['motivo']}")
        else:
            print("\nNÃO CONVERSÍVEL PARA YOLO")
            print(f"   {diagnostico['motivo']}")
        
        print("\n" + "=" * 80)
    
    def organizar_para_yolo(self, pasta_saida: str = None) -> Path:
        """
        Organiza dataset em estrutura YOLO.
        
        Args:
            pasta_saida: Pasta de saída (padrão: <nome_dataset>_yolo)
        
        Returns:
            Caminho da pasta organizada
        """
        diagnostico = self.diagnosticar()
        
        if not diagnostico['conversivel_yolo']:
            print("\n❌ Este dataset NÃO pode ser convertido para YOLO")
            print(f"   Motivo: {diagnostico['motivo']}")
            return None
        
        # Criar pasta de saída
        if pasta_saida is None:
            pasta_saida = self.caminho.parent / f"{self.nome_dataset}_yolo"
        else:
            pasta_saida = Path(pasta_saida)
        
        pasta_saida.mkdir(exist_ok=True)
        
        print(f"\nOrganizando em: {pasta_saida}")
        
        # TODO: Implementar lógica de organização baseada no tipo
        # Por enquanto, apenas informa
        
        print("\nORGANIZAÇÃO MANUAL NECESSÁRIA")
        print("\nPara datasets do Zenodo:")
        print("1. Verifique se há um README explicando a estrutura.")
        print("2. Se tiver labels COCO (.json), use ferramenta de conversão:")
        print("   https://github.com/Labelbox/coco-to-yolo")
        print("3. Se NÃO tiver labels, considere:")
        print("   - Usar o dataset para pré-treino (transfer learning)")
        print("   - Anotar manualmente com LabelImg ou Roboflow")
        print("   - Usar outro dataset que já tenha labels")
        
        return pasta_saida


def main():
    """Função principal."""
    import sys
    
    print("=" * 80)
    print("ORGANIZADOR DE DATASETS DO ZENODO")
    print("=" * 80)
    
    if len(sys.argv) < 2:
        print("\nVerificando datasets do Zenodo em F:/...\n")
        drive_f = Path("F:/")
        
        # Procurar por pastas do Zenodo
        zenodo_comum = [
            drive_f / "datasets_publicos_rgb/grandes_pretreino/deepstat_wp5_zenodo",
            drive_f / "datasets_publicos_rgb/grandes_pretreino/bdappv_zenodo",
            drive_f / "datasets_publicos/deepstat_wp5_zenodo",
            drive_f / "datasets_publicos/bdappv_zenodo",
        ]
        
        datasets_encontrados = [p for p in zenodo_comum if p.exists()]
        
        if not datasets_encontrados:
            # Buscar qualquer pasta com 'zenodo' no nome
            for pasta in drive_f.rglob("*zenodo*"):
                if pasta.is_dir():
                    datasets_encontrados.append(pasta)
        
        if not datasets_encontrados:
            print("Nenhum dataset do Zenodo encontrado em F:/")
            print("\nUso: python organizar_zenodo.py <caminho_para_dataset>")
            print("\nExemplo:")
            print("  python organizar_zenodo.py F:/datasets_publicos/deepstat_wp5_zenodo")
            return
        
        print(f"{len(datasets_encontrados)} dataset(s) do Zenodo encontrado(s):")
        for i, p in enumerate(datasets_encontrados, 1):
            print(f"   {i}. {p}")
        
        # Diagnosticar todos
        print("\n" + "=" * 80)
        for dataset_path in datasets_encontrados:
            organizador = OrganizadorDatasetZenodo(str(dataset_path))
            diagnostico = organizador.diagnosticar()
            print()  # Linha em branco entre datasets
    
    else:
        caminho = sys.argv[1]
        organizador = OrganizadorDatasetZenodo(caminho)
        diagnostico = organizador.diagnosticar()
        
        if diagnostico['conversivel_yolo']:
            print("\nDeseja organizar este dataset? (s/n): ", end="")
            resposta = input().lower()
            if resposta == 's':
                pasta_organizada = organizador.organizar_para_yolo()
                if pasta_organizada:
                    print(f"\nDataset organizado em: {pasta_organizada}")
    
    print("\n" + "=" * 80)
    print("RECOMENDAÇÃO")
    print("=" * 80)
    print("\nSe os datasets do Zenodo não tiverem labels YOLO:")
    print("\nMelhor opção: usar datasets do Roboflow")
    print("   - Já vêm no formato YOLO pronto")
    print("   - Têm labels de detecção de módulos")
    print("   - Alguns têm labels de sujidade")
    print("\nDatasets Roboflow recomendados:")
    print("   1. Solar PV Maintenance Combined (2.474 imgs)")
    print("   2. Aerial Solar Panels Brad (53 imgs)")
    print("   3. Soiling Detection Ammar (490 imgs)")
    print("\nExecute:")
    print("   python converter_datasets_para_yolo.py")


if __name__ == "__main__":
    main()
