#!/usr/bin/env python3
"""Conversão de múltiplos datasets públicos de painéis solares para formato YOLO.

Este script consolida diferentes conjuntos de dados em um único dataset
no formato YOLO, adequado para treinar modelos de detecção de módulos
fotovoltaicos (por exemplo, YOLO11).
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import random

class ConversorDatasetYOLO:
    """Converte datasets públicos para formato YOLO e prepara treinamento."""
    
    def __init__(self, drive_raiz: str = "F:/"):
        """
        Inicializa o conversor.
        
        Args:
            drive_raiz: Caminho raiz onde estão os datasets (padrão: F:/)
        """
        self.drive_raiz = Path(drive_raiz)
        self.pasta_saida = self.drive_raiz / "dataset_yolo_detector_modulos"
        
        # Criar estrutura de saída
        self.pasta_saida.mkdir(exist_ok=True)
        (self.pasta_saida / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.pasta_saida / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.pasta_saida / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.pasta_saida / "val" / "labels").mkdir(parents=True, exist_ok=True)
        (self.pasta_saida / "test" / "images").mkdir(parents=True, exist_ok=True)
        (self.pasta_saida / "test" / "labels").mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'train': 0,
            'val': 0,
            'test': 0,
            'total': 0
        }
    
    def verificar_datasets_disponiveis(self) -> Dict[str, Path]:
        """
        Verifica quais datasets estão disponíveis no drive.
        
        Returns:
            Dicionário com nome e caminho dos datasets encontrados
        """
        print("=" * 80)
        print("VERIFICANDO DATASETS DISPONÍVEIS")
        print("=" * 80)
        
        datasets_encontrados = {}
        
        # Locais comuns onde datasets podem estar
        locais_busca = [
            self.drive_raiz / "datasets_publicos_rgb",
            self.drive_raiz / "datasets_publicos",
            self.drive_raiz / "dataset",
            self.drive_raiz
        ]
        
        for local in locais_busca:
            if not local.exists():
                continue
                
            print(f"\nVerificando: {local}")
            
            # Procurar por datasets do Roboflow (formato YOLO)
            for pasta in local.rglob("*"):
                if not pasta.is_dir():
                    continue
                
                # Verificar se tem estrutura YOLO (train/, valid/ ou val/)
                tem_train = (pasta / "train").exists() or (pasta / "train" / "images").exists()
                tem_valid = (pasta / "valid").exists() or (pasta / "val").exists() or \
                           (pasta / "valid" / "images").exists() or (pasta / "val" / "images").exists()
                tem_data_yaml = (pasta / "data.yaml").exists() or (pasta / "dataset.yaml").exists()
                
                if tem_train and tem_valid:
                    nome_dataset = pasta.name
                    datasets_encontrados[nome_dataset] = pasta
                    print(f"  {nome_dataset} - formato YOLO válido")
                    print(f"     Caminho: {pasta}")
        
        print(f"\nResumo: {len(datasets_encontrados)} datasets YOLO encontrados")
        return datasets_encontrados
    
    def listar_roboflow_datasets(self) -> List[Path]:
        """
        Lista especificamente os datasets do Roboflow que devem estar em formato YOLO.
        
        Returns:
            Lista de caminhos para datasets Roboflow
        """
        roboflow_base = self.drive_raiz / "datasets_publicos_rgb"
        datasets = []
        
        if not roboflow_base.exists():
            print(f"Aviso: pasta {roboflow_base} não encontrada")
            return datasets
        
        # Datasets do Roboflow que devem ter detecção de módulos
        datasets_esperados = [
            "aereos_drone/solar_pv_maintenance_combined",
            "aereos_drone/aerial_solar_panels_brad",
            "aereos_drone/soiling_detection_ammar",
            "aereos_drone/solar_detection_tagus",
            "aereos_drone/bird_drop_anomalies",
            "aereos_drone/solar_panel_combine_zindi",
            "solo_binario/detection_soiling_tcs",
            "solo_binario/solar_panels_yolomodel"
        ]
        
        for dataset_rel in datasets_esperados:
            dataset_path = roboflow_base / dataset_rel
            if dataset_path.exists():
                datasets.append(dataset_path)
        
        return datasets
    
    def copiar_dataset_yolo(self, origem: Path, split: str = None) -> int:
        """
        Copia um dataset já no formato YOLO para a pasta de saída.
        
        Args:
            origem: Caminho do dataset origem
            split: Se especificado, força divisão (train/val/test). Se None, mantém splits originais.
        
        Returns:
            Número de imagens copiadas
        """
        imagens_copiadas = 0
        
        # Procurar por train/, valid/, val/, test/
        splits_origem = []
        for split_nome in ['train', 'valid', 'val', 'test']:
            split_path = origem / split_nome
            if split_path.exists():
                splits_origem.append((split_nome, split_path))
        
        if not splits_origem:
            print(f"  ⚠️  Sem splits encontrados em {origem.name}")
            return 0
        
        for split_nome_orig, split_path in splits_origem:
            # Mapear valid/val para 'val'
            if split_nome_orig in ['valid', 'val']:
                split_destino = 'val'
            else:
                split_destino = split_nome_orig
            
            # Verificar se tem images/ e labels/ ou se estão diretamente no split
            images_path = split_path / "images"
            labels_path = split_path / "labels"
            
            if not images_path.exists():
                # Imagens podem estar diretamente no split
                images_path = split_path
                # Procurar pasta de labels
                labels_path = split_path.parent / "labels" / split_nome_orig if (split_path.parent / "labels").exists() else None
            
            if not images_path.exists():
                print(f"  ⚠️  Pasta images não encontrada em {split_path}")
                continue
            
            # Listar imagens
            extensoes_imagem = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            imagens = [f for f in images_path.iterdir() 
                      if f.is_file() and f.suffix in extensoes_imagem]
            
            print(f"  📦 {split_destino.upper()}: {len(imagens)} imagens de {origem.name}/{split_nome_orig}")
            
            # Copiar imagens e labels
            for img_path in imagens:
                # Copiar imagem
                nome_unico = f"{origem.name}_{split_nome_orig}_{img_path.name}"
                img_destino = self.pasta_saida / split_destino / "images" / nome_unico
                shutil.copy2(img_path, img_destino)
                
                # Copiar label se existir
                if labels_path and labels_path.exists():
                    label_path = labels_path / f"{img_path.stem}.txt"
                    if label_path.exists():
                        label_destino = self.pasta_saida / split_destino / "labels" / f"{img_path.stem}.txt"
                        # Renomear label para corresponder à imagem única
                        label_destino = self.pasta_saida / split_destino / "labels" / f"{Path(nome_unico).stem}.txt"
                        shutil.copy2(label_path, label_destino)
                
                imagens_copiadas += 1
                self.stats[split_destino] += 1
        
        return imagens_copiadas
    
    def mesclar_datasets(self, datasets: List[Path]) -> None:
        """
        Mescla múltiplos datasets YOLO em um único dataset consolidado.
        
        Args:
            datasets: Lista de caminhos para datasets no formato YOLO
        """
        print("\n" + "=" * 80)
        print("MESCLANDO DATASETS")
        print("=" * 80)
        
        for dataset_path in datasets:
            print(f"\nProcessando: {dataset_path.name}")
            n_copiadas = self.copiar_dataset_yolo(dataset_path)
            print(f"  {n_copiadas} imagens copiadas")
        
        self.stats['total'] = sum([self.stats['train'], self.stats['val'], self.stats['test']])
    
    def balancear_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> None:
        """
        Balanceia os splits train/val/test se estiverem desbalanceados.
        
        Args:
            train_ratio: Proporção para treino (padrão: 70%)
            val_ratio: Proporção para validação (padrão: 15%)
            test_ratio: Proporção para teste (padrão: 15%)
        """
        print("\n" + "=" * 80)
        print("BALANCEANDO SPLITS")
        print("=" * 80)
        
        # Coletar todas as imagens
        todas_imagens = []
        for split in ['train', 'val', 'test']:
            images_dir = self.pasta_saida / split / "images"
            imagens = list(images_dir.glob("*.*"))
            todas_imagens.extend([(img, split) for img in imagens])
        
        # Embaralhar
        random.shuffle(todas_imagens)
        
        total = len(todas_imagens)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        n_test = total - n_train - n_val
        
        print(f"Total de imagens: {total}")
        print(f"   Train: {n_train} ({train_ratio*100:.0f}%)")
        print(f"   Val:   {n_val} ({val_ratio*100:.0f}%)")
        print(f"   Test:  {n_test} ({test_ratio*100:.0f}%)")
        
        # Redistribuir se necessário
        splits_atuais = {
            'train': self.stats['train'],
            'val': self.stats['val'],
            'test': self.stats['test']
        }
        
        # Verificar se está muito desbalanceado (>10% de diferença)
        diff_train = abs(splits_atuais['train'] - n_train) / total
        diff_val = abs(splits_atuais['val'] - n_val) / total
        diff_test = abs(splits_atuais['test'] - n_test) / total
        
        if max(diff_train, diff_val, diff_test) > 0.10:
            print("\nSplits desbalanceados; redistribuição recomendada.")
            # TODO: Implementar redistribuição se necessário
            # Por enquanto, apenas avisa
        else:
            print("\nSplits estão balanceados.")
    
    def criar_dataset_yaml(self, nome_classe: str = "solar_module") -> Path:
        """
        Cria arquivo dataset.yaml para o YOLO.
        
        Args:
            nome_classe: Nome da classe para detecção (padrão: "solar_module")
        
        Returns:
            Caminho para o arquivo dataset.yaml criado
        """
        yaml_path = self.pasta_saida / "dataset.yaml"
        
        config = {
            'path': str(self.pasta_saida.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            
            'nc': 1,  # Número de classes
            'names': [nome_classe]  # Nome das classes
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print("\n" + "=" * 80)
        print("DATASET.YAML CRIADO")
        print("=" * 80)
        print(f"Caminho: {yaml_path}")
        print("\nConteúdo:")
        print("-" * 40)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            print(f.read())
        print("-" * 40)
        
        return yaml_path
    
    def gerar_relatorio(self) -> None:
        """Gera relatório final da conversão."""
        print("\n" + "=" * 80)
        print("RELATÓRIO FINAL")
        print("=" * 80)
        
        print(f"\nDataset YOLO criado em: {self.pasta_saida}")
        print("\nEstatísticas:")
        print(f"   Total de imagens: {self.stats['total']}")
        print(f"   Train: {self.stats['train']} ({self.stats['train']/self.stats['total']*100:.1f}%)")
        print(f"   Val:   {self.stats['val']} ({self.stats['val']/self.stats['total']*100:.1f}%)")
        print(f"   Test:  {self.stats['test']} ({self.stats['test']/self.stats['total']*100:.1f}%)")
        
        print("\nEstrutura criada:")
        print(f"   {self.pasta_saida}/")
        print(f"   ├── train/")
        print(f"   │   ├── images/ ({self.stats['train']} imagens)")
        print(f"   │   └── labels/ ({self.stats['train']} arquivos .txt)")
        print(f"   ├── val/")
        print(f"   │   ├── images/ ({self.stats['val']} imagens)")
        print(f"   │   └── labels/ ({self.stats['val']} arquivos .txt)")
        print(f"   ├── test/")
        print(f"   │   ├── images/ ({self.stats['test']} imagens)")
        print(f"   │   └── labels/ ({self.stats['test']} arquivos .txt)")
        print(f"   └── dataset.yaml")
    
    def gerar_comando_treinamento(self, yaml_path: Path) -> str:
        """
        Gera comando pronto para treinar YOLO11.
        
        Args:
            yaml_path: Caminho para o dataset.yaml
        
        Returns:
            Comando Python para treinar o modelo
        """
        comando = f"""
# Comando de exemplo para treinar YOLO11 a partir do dataset gerado

# OPÇÃO 1: Script Python simples
from backend.aplicacao.servicos.treinamento_detector import TreinadorDetector

# Inicializar treinador
treinador = TreinadorDetector('{yaml_path}')

# Treinar modelo
treinador.treinar(
    epochs=200,
    imgsz=640,
    batch=16,  # Ajustar conforme GPU/RAM disponível
    device='mps',  # ou 'cuda' se tiver NVIDIA, ou 'cpu'
    patience=50,
    save_period=10,
    modelo_base='yolo11n.pt'  # ou yolo11s, yolo11m, yolo11l, yolo11x
)

## Alternativa: usar Ultralytics diretamente

from ultralytics import YOLO

# Carregar modelo pré-treinado
model = YOLO('yolo11n.pt')  # nano (mais rápido) ou yolo11s, yolo11m

# Treinar
results = model.train(
    data='{yaml_path}',
    epochs=200,
    imgsz=640,
    batch=16,
    device='mps',  # 'mps' para M1 Mac, 'cuda' para NVIDIA, 'cpu' para CPU
    patience=50,
    save_period=10,
    project='runs/detector_modulos',
    name='yolo11_modulos_v1',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,  # box loss gain
    cls=0.5,  # cls loss gain
    dfl=1.5,  # dfl loss gain
    verbose=True,
    seed=42
)

# Avaliar no conjunto de teste
metrics = model.val()

# Exibir métricas
print(f"mAP@0.5: {{metrics.box.map50:.4f}}")
print(f"mAP@0.5:0.95: {{metrics.box.map:.4f}}")
print(f"Precision: {{metrics.box.p:.4f}}")
print(f"Recall: {{metrics.box.r:.4f}}")

# Salvar métricas em JSON
import json
metricas_tcc = {{
    'mAP50': float(metrics.box.map50),
    'mAP50_95': float(metrics.box.map),
    'precision': float(metrics.box.p),
    'recall': float(metrics.box.r),
    'box_loss': float(results.results_dict.get('train/box_loss', 0)),
    'cls_loss': float(results.results_dict.get('train/cls_loss', 0)),
    'dfl_loss': float(results.results_dict.get('train/dfl_loss', 0))
}

with open('metricas_yolo11_detector.json', 'w') as f:
    json.dump(metricas_tcc, f, indent=2)

print("\nMétricas salvas em metricas_yolo11_detector.json")
"""
        
        print("\n" + "=" * 80)
        print("COMANDOS DE TREINAMENTO")
        print("=" * 80)
        print(comando)
        
        # Salvar comando em arquivo
        comando_path = self.pasta_saida / "COMO_TREINAR.py"
        with open(comando_path, 'w', encoding='utf-8') as f:
            f.write(comando)
        
        print(f"\nComando salvo em: {comando_path}")
        
        return comando


def main():
    """Função principal."""
    print("=" * 80)
    print("CONVERSOR DE DATASETS PARA YOLO11")
    print("=" * 80)
    
    # Inicializar conversor
    conversor = ConversorDatasetYOLO(drive_raiz="F:/")
    
    # Verificar datasets disponíveis
    datasets_disponiveis = conversor.verificar_datasets_disponiveis()
    
    if not datasets_disponiveis:
        print("\nERRO: nenhum dataset no formato YOLO encontrado.")
        print("\nSolução sugerida:")
        print("   1. Baixar datasets do Roboflow em formato YOLOv8")
        print("   2. Executar os scripts de download incluídos neste projeto")
        print("\n   Datasets recomendados:")
        print("   - Solar PV Maintenance Combined (2.474 imgs)")
        print("   - Aerial Solar Panels Brad Dwyer (53 imgs)")
        print("   - Soiling Detection Ammar (490 imgs)")
        return
    
    # Listar datasets Roboflow específicos
    datasets_roboflow = conversor.listar_roboflow_datasets()
    
    if datasets_roboflow:
        print(f"\nEncontrados {len(datasets_roboflow)} datasets Roboflow")
        
        # Mesclar todos os datasets
        conversor.mesclar_datasets(datasets_roboflow)
        
        # Balancear splits
        conversor.balancear_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # Criar dataset.yaml
        yaml_path = conversor.criar_dataset_yaml(nome_classe="solar_module")
        
        # Gerar relatório
        conversor.gerar_relatorio()
        
        # Gerar comando de treinamento
        conversor.gerar_comando_treinamento(yaml_path)
        
        print("\n" + "=" * 80)
        print("CONVERSÃO CONCLUÍDA COM SUCESSO")
        print("=" * 80)
        print(f"\nDataset YOLO pronto em: {conversor.pasta_saida}")
        print(f"dataset.yaml: {yaml_path}")
        
    else:
        print("\nNenhum dataset Roboflow encontrado.")
        print("   Use os datasets disponíveis encontrados:")
        for nome, caminho in datasets_disponiveis.items():
            print(f"   - {nome}: {caminho}")


if __name__ == "__main__":
    main()
