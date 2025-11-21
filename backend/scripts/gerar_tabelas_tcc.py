#!/usr/bin/env python3
"""
Gera artefatos (tabelas/figuras) para o TCC:
- Matriz de confusão
- Métricas por classe (CSV e LaTeX)
- Curva ROC (binário)
- Tabela de hiperparâmetros (LaTeX)
- Tabela de comparação (LaTeX)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch
from torchvision import transforms
from PIL import Image


def carregar_modelo(modelo_path: str, num_classes: int = 2):
    ckpt = torch.load(modelo_path, map_location='cpu')
    arch = ckpt.get('arquitetura', 'efficientnet_b4')
    # Import tardio para evitar dependência circular
    from aplicacao.modelos.classificador_sujidade import ClassificadorSujidade
    if arch == 'resnet50':
        from aplicacao.modelos.classificador_resnet import ClassificadorResNet
        clf = ClassificadorResNet(num_classes=num_classes)
    else:
        clf = ClassificadorSujidade(num_classes=num_classes, modelo_base=arch)
    clf.modelo.load_state_dict(ckpt['state_dict'], strict=False)
    clf.modelo.eval()
    return clf


def build_loader(image_paths, img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensors = []
    for p in image_paths:
        im = Image.open(p).convert('RGB')
        tensors.append(tf(im))
    X = torch.stack(tensors)
    return X


def avaliar_modelo_completo(modelo_path: str, test_dir: str):
    classes = ['limpo', 'sujo']
    paths = []
    y_true = []
    for idx, c in enumerate(classes):
        d = Path(test_dir) / c
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
            for p in d.glob(ext):
                paths.append(str(p))
                y_true.append(idx)
    X = build_loader(paths)
    clf = carregar_modelo(modelo_path, num_classes=2)
    # Garantir que os tensores estejam no mesmo dispositivo do modelo
    device = next(clf.modelo.parameters()).device
    with torch.no_grad():
        logits = []
        for i in range(0, len(X), 16):
            batch = X[i:i+16].to(device)
            logits.append(clf.modelo(batch))
        logits = torch.cat(logits, dim=0)
        probs = torch.softmax(logits, dim=1)
        y_pred = probs.argmax(dim=1).cpu().numpy()
    return np.array(y_true), y_pred, probs.cpu().numpy()


def gerar_tabelas(modelo_path: str, dataset_root: str, out_root: str):
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    test_dir = str(Path(dataset_root) / 'test')
    y_true, y_pred, y_prob = avaliar_modelo_completo(modelo_path, test_dir)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Limpo', 'Sujo'], yticklabels=['Limpo', 'Sujo'])
    plt.title('Matriz de Confusão (Teste)')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig(out / 'matriz_confusao.png', dpi=300)
    plt.close()

    # Métricas por classe
    report = classification_report(y_true, y_pred, target_names=['Limpo', 'Sujo'], output_dict=True)
    df_metrics = pd.DataFrame(report).transpose()
    df_metrics.to_csv(out / 'metricas_por_classe.csv')
    with open(out / 'tabela_metricas.tex', 'w') as f:
        f.write(df_metrics.to_latex(float_format='%.3f'))

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC (Teste)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out / 'curva_roc.png', dpi=300)
    plt.close()

    # Hiperparâmetros (exemplo; pode ajustar conforme relatório salvo)
    hiper = {
        'Modelo': 'EfficientNet-B4',
        'Epochs': 30,
        'Batch Size': 16,
        'Learning Rate': 0.001,
        'Optimizer': 'AdamW',
        'Loss Function': 'CrossEntropyLoss (weighted)',
        'Data Augmentation': 'torchvision (leve)',
        'Early Stopping': 'patience=3 por val_acc',
        'Hardware': 'Mac M1/MPS',
    }
    df_hiper = pd.DataFrame(list(hiper.items()), columns=['Hiperparâmetro', 'Valor'])
    df_hiper.to_latex(out / 'tabela_hiperparametros.tex', index=False)

    # Comparação (placeholder)
    comp = {
        'Trabalho': ['SolNet (2022)', 'DeepSolarEye (2018)', 'EffNet-B0 (2024)', 'Nosso (2025)'],
        'Dataset': ['2.231 imgs', '45.754 imgs', '2.231 imgs', '~46K imgs'],
        'Acurácia': ['98.2%', '96.0%', '98.5%', 'XX.X%']
    }
    pd.DataFrame(comp).to_latex(out / 'tabela_comparacao.tex', index=False)

    print('\n✅ Artefatos gerados em', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelo', required=True, help='Caminho para melhor_modelo.pth')
    ap.add_argument('--dataset', required=True, help='Raiz do dataset (com test/)')
    ap.add_argument('--out', default='/Volumes/Z Slim/modelos_salvos', help='Saída')
    args = ap.parse_args()
    gerar_tabelas(args.modelo, args.dataset, args.out)
