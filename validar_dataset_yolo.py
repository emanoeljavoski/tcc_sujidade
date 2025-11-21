#!/usr/bin/env python3
"""
Script de validação rápida de datasets YOLO.
Verifica se um dataset tem a estrutura necessária para treinar YOLO11.

Uso:
    python validar_dataset_yolo.py F:/caminho/para/dataset
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

class ValidadorDatasetYOLO:
    """Valida se um dataset está pronto para treinar YOLO."""
    
    def __init__(self, caminho_dataset: str):
        """
        Inicializa validador.
        
        Args:
            caminho_dataset: Caminho para a pasta do dataset ou para o dataset.yaml
        """
        self.caminho = Path(caminho_dataset)
        self.erros = []
        self.avisos = []
        self.info = []
        
    def validar(self) -> bool:
        """
        Executa validação completa.
        
        Returns:
            True se dataset é válido, False caso contrário
        """
        print("=" * 80)
        print("VALIDADOR DE DATASET YOLO")
        print("=" * 80)
        print(f"\nVerificando: {self.caminho}")
        
        # 1. Verificar se caminho existe
        if not self.caminho.exists():
            self.erros.append(f"Erro: caminho não existe: {self.caminho}")
            return False
        
        # 2. Encontrar dataset.yaml
        yaml_path = self._encontrar_yaml()
        if not yaml_path:
            self.erros.append("Erro: dataset.yaml ou data.yaml não encontrado")
            return False
        
        self.info.append(f"YAML encontrado: {yaml_path.name}")
        
        # 3. Validar conteúdo do YAML
        yaml_valido = self._validar_yaml(yaml_path)
        if not yaml_valido:
            return False
        
        # 4. Validar estrutura de pastas
        estrutura_valida = self._validar_estrutura(yaml_path)
        if not estrutura_valida:
            return False
        
        # 5. Validar imagens e labels
        imgs_labels_ok = self._validar_imagens_labels(yaml_path)
        
        # 6. Exibir resultados
        self._exibir_resultados()
        
        return len(self.erros) == 0
    
    def _encontrar_yaml(self) -> Path:
        """Procura por dataset.yaml ou data.yaml."""
        # Se o caminho já é um YAML
        if self.caminho.suffix in ['.yaml', '.yml']:
            return self.caminho if self.caminho.exists() else None
        
        # Procurar na pasta
        for nome in ['dataset.yaml', 'data.yaml', 'config.yaml']:
            yaml_path = self.caminho / nome
            if yaml_path.exists():
                return yaml_path
        
        # Procurar recursivamente (até 2 níveis)
        for yaml_path in self.caminho.rglob('*.yaml'):
            if yaml_path.name in ['dataset.yaml', 'data.yaml']:
                return yaml_path
        
        return None
    
    def _validar_yaml(self, yaml_path: Path) -> bool:
        """Valida conteúdo do arquivo YAML."""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Verificar campos obrigatórios
            campos_obrigatorios = ['train', 'val', 'nc', 'names']
            campos_faltando = [c for c in campos_obrigatorios if c not in config]
            
            if campos_faltando:
                self.erros.append(f"Campos faltando no YAML: {', '.join(campos_faltando)}")
                return False
            
            # Verificar número de classes
            nc = config['nc']
            names = config['names']
            
            if len(names) != nc:
                self.avisos.append(f"Aviso: nc={nc} mas names tem {len(names)} elementos")
            
            self.info.append(f"Classes: {nc} ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})")
            
            # Verificar caminhos
            for split in ['train', 'val']:
                if split in config:
                    self.info.append(f"Split '{split}': {config[split]}")
            
            return True
            
        except Exception as e:
            self.erros.append(f"❌ Erro ao ler YAML: {e}")
            return False
    
    def _validar_estrutura(self, yaml_path: Path) -> bool:
        """Valida estrutura de pastas do dataset."""
        dataset_root = yaml_path.parent
        
        # Verificar splits esperados
        splits = ['train', 'val']
        splits_encontrados = []
        
        for split in splits:
            split_path = dataset_root / split
            # Também checar 'valid' como alternativa para 'val'
            if not split_path.exists() and split == 'val':
                split_path = dataset_root / 'valid'
            
            if split_path.exists():
                splits_encontrados.append(split)
                self.info.append(f"Pasta '{split}' encontrada")
                
                # Verificar subpastas images/ e labels/
                images_path = split_path / 'images'
                labels_path = split_path / 'labels'
                
                if not images_path.exists():
                    # Talvez as imagens estejam direto no split
                    # Verificar se há imagens diretamente
                    imagens_diretas = list(split_path.glob('*.jpg')) + \
                                     list(split_path.glob('*.png'))
                    if imagens_diretas:
                        self.avisos.append(f"Aviso: '{split}': imagens estão direto na pasta (sem 'images/')")
                    else:
                        self.erros.append(f"'{split}/images' não encontrada")
                else:
                    self.info.append(f"   '{split}/images' OK")
                
                if not labels_path.exists():
                    self.avisos.append(f"Aviso: '{split}/labels' não encontrada")
                else:
                    self.info.append(f"   '{split}/labels' OK")
            else:
                self.erros.append(f"❌ Pasta '{split}' não encontrada")
        
        return len([e for e in self.erros if 'Pasta' in e]) == 0
    
    def _validar_imagens_labels(self, yaml_path: Path) -> bool:
        """Valida presença de imagens e labels."""
        dataset_root = yaml_path.parent
        
        stats = {
            'train': {'images': 0, 'labels': 0},
            'val': {'images': 0, 'labels': 0},
            'test': {'images': 0, 'labels': 0}
        }
        
        for split in ['train', 'val', 'valid', 'test']:
            split_path = dataset_root / split
            if not split_path.exists():
                continue
            
            # Normalizar 'valid' para 'val'
            split_key = 'val' if split == 'valid' else split
            
            # Contar imagens
            images_path = split_path / 'images'
            if images_path.exists():
                extensoes = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                for ext in extensoes:
                    stats[split_key]['images'] += len(list(images_path.glob(f'*{ext}')))
            else:
                # Imagens podem estar direto no split
                for ext in ['.jpg', '.jpeg', '.png']:
                    stats[split_key]['images'] += len(list(split_path.glob(f'*{ext}')))
            
            # Contar labels
            labels_path = split_path / 'labels'
            if labels_path.exists():
                stats[split_key]['labels'] = len(list(labels_path.glob('*.txt')))
        
        # Exibir estatísticas
        print("\nESTATÍSTICAS:")
        print("-" * 40)
        for split in ['train', 'val', 'test']:
            n_imgs = stats[split]['images']
            n_labels = stats[split]['labels']
            
            if n_imgs > 0:
                percentual = (n_labels / n_imgs * 100) if n_imgs > 0 else 0
                status = "OK" if n_labels > 0 else "AVISO"
                print(f"{status} {split.upper():6s}: {n_imgs:5d} imagens | {n_labels:5d} labels ({percentual:.1f}%)")
                
                if n_labels == 0 and n_imgs > 0:
                    self.avisos.append(f"{split.upper()} tem imagens mas sem labels.")
                elif n_labels < n_imgs * 0.9:
                    self.avisos.append(f"{split.upper()} tem menos labels que imagens ({percentual:.1f}%).")
        
        print("-" * 40)
        total_imgs = sum(stats[s]['images'] for s in stats)
        total_labels = sum(stats[s]['labels'] for s in stats)
        print(f"TOTAL: {total_imgs} imagens | {total_labels} labels")
        
        # Validação final
        if stats['train']['images'] == 0:
            self.erros.append("Crítico: nenhuma imagem de treino encontrada.")
            return False
        
        if stats['val']['images'] == 0:
            self.avisos.append("Nenhuma imagem de validação (recomendado ter).")
        
        if total_labels == 0:
            self.erros.append("Crítico: nenhum label (.txt) encontrado.")
            return False
        
        return True
    
    def _exibir_resultados(self) -> None:
        """Exibe resumo da validação."""
        print("\n" + "=" * 80)
        print("RESULTADO DA VALIDAÇÃO")
        print("=" * 80)
        
        if self.info:
            print("\nINFORMAÇÕES:")
            for msg in self.info:
                print(f"   {msg}")
        
        if self.avisos:
            print("\nAVISOS:")
            for msg in self.avisos:
                print(f"   {msg}")
        
        if self.erros:
            print("\nERROS:")
            for msg in self.erros:
                print(f"   {msg}")
        
        print("\n" + "=" * 80)
        if len(self.erros) == 0:
            print("DATASET VÁLIDO. Pronto para treinar YOLO.")
        else:
            print("DATASET INVÁLIDO. Corrija os erros acima.")
        print("=" * 80)


def main():
    """Função principal."""
    if len(sys.argv) < 2:
        print("Uso: python validar_dataset_yolo.py <caminho_dataset>")
        print("\nExemplos:")
        print("  python validar_dataset_yolo.py F:/dataset_yolo")
        print("  python validar_dataset_yolo.py F:/dataset_yolo/dataset.yaml")
        print("\nProcurando datasets automaticamente em F:/...")
        
        # Procurar datasets em F:/
        drive_f = Path("F:/")
        if drive_f.exists():
            datasets_encontrados = []
            for yaml_path in drive_f.rglob('*.yaml'):
                if yaml_path.name in ['dataset.yaml', 'data.yaml']:
                    datasets_encontrados.append(yaml_path)
            
            if datasets_encontrados:
                print(f"\n{len(datasets_encontrados)} datasets YOLO encontrados:")
                for i, yaml_path in enumerate(datasets_encontrados[:10], 1):
                    print(f"   {i}. {yaml_path}")
                
                if len(datasets_encontrados) > 10:
                    print(f"   ... e mais {len(datasets_encontrados) - 10}")
                
                print("\nValidando o primeiro dataset encontrado...")
                caminho = datasets_encontrados[0].parent
            else:
                print("\nNenhum dataset YOLO encontrado em F:/")
                return
        else:
            print("\nDrive F:/ não encontrado")
            return
    else:
        caminho = sys.argv[1]
    
    # Validar
    validador = ValidadorDatasetYOLO(caminho)
    dataset_valido = validador.validar()
    
    if dataset_valido:
        print("\nExemplo de comando de treino:")
        print(f"""
from backend.aplicacao.servicos.treinamento_detector import TreinadorDetector

# Caminho validado
yaml_path = '{validador.caminho if validador.caminho.suffix in ['.yaml', '.yml'] else validador._encontrar_yaml()}'

# Treinar
treinador = TreinadorDetector(yaml_path)
resultados = treinador.treinar(
    epochs=200,
    imgsz=640,
    batch=16,
    device='mps'  # ou 'cuda' ou 'cpu'
)

# Métricas para TCC
print(f"mAP@0.5: {{resultados['metrics']['mAP50']:.4f}}")
print(f"Precision: {{resultados['metrics']['precision']:.4f}}")
print(f"Recall: {{resultados['metrics']['recall']:.4f}}")
""")
    else:
        print("\nSugestões:")
        print("   1. Baixar datasets do Roboflow em formato YOLOv8")
        print("   2. Executar: python converter_datasets_para_yolo.py")
    
    sys.exit(0 if dataset_valido else 1)


if __name__ == "__main__":
    main()
