"""
Treinamento do Classificador EfficientNet para Sujidade em Painéis
Desenvolvido para TCC - Engenharia Mecatrônica

Este módulo implementa o pipeline de treinamento utilizado no TCC para o
classificador binário limpo/sujo baseado em EfficientNet-B4, com suporte a
outras variantes (B0, B1, ResNet50) quando necessário.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import json
import time
import logging
from typing import Dict, Callable, Optional, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from aplicacao.config import HARDWARE as CONFIG_HARDWARE
from .classificador_sujidade import ClassificadorSujidade
# ResNet opcional
try:
    from .classificador_resnet import ClassificadorResNet
except Exception:
    ClassificadorResNet = None

logger = logging.getLogger(__name__)

# Estado global do treinamento (para API polling)
estado_treinamento_classificador = {
    'treinando': False,
    'epoca_atual': 0,
    'total_epocas': 0,
    'progresso': 0,
    'metricas': {
        'train_loss': 0.0,
        'train_acc': 0.0,
        'val_loss': 0.0,
        'val_acc': 0.0
    },
    'historico': {
        'epocas': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    },
    'tempo_restante_seg': 0,
    'inicio_treinamento': None,
    'erro': None
}

class TreinadorClassificador:
    """
    Classe para treinamento do classificador de sujidade baseado em EfficientNet.
    
    Neste TCC, a configuração principal utiliza EfficientNet-B4 como backbone
    para classificação binária (limpo/sujo), explorando transfer learning a
    partir de pesos pré-treinados no ImageNet. O código permanece flexível para
    permitir outras variantes (por exemplo, EfficientNet-B0 ou ResNet50) quando
    necessário para comparação.
    """
    
    def __init__(self, diretorio_dataset: str, modelo_base: str = 'efficientnet_b4'):
        """
        Inicializa o treinador do classificador.
        
        Args:
            diretorio_dataset (str): Diretório com subpastas limpo/ e sujo/
            modelo_base (str): Arquitetura base (efficientnet_b0, efficientnet_b1, etc.)
        """
        self.diretorio_dataset = Path(diretorio_dataset)
        self.modelo_base = modelo_base

        # Seleciona dispositivo com base na prioridade configurada (mps > cuda > cpu por padrão)
        try:
            prioridades = CONFIG_HARDWARE.get("device_prioridade", ["mps", "cuda", "cpu"])
        except Exception:  # fallback seguro se config não estiver disponível
            prioridades = ["mps", "cuda", "cpu"]

        dispositivo_escolhido = torch.device("cpu")
        for nome in prioridades:
            if nome == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    dispositivo_escolhido = torch.device("mps")
                    break
            elif nome == "cuda":
                if torch.cuda.is_available():
                    dispositivo_escolhido = torch.device("cuda")
                    break
            elif nome == "cpu":
                dispositivo_escolhido = torch.device("cpu")
                break

        self.dispositivo = dispositivo_escolhido
        
        # Validar dataset
        if not self.diretorio_dataset.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {diretorio_dataset}")
        
        # Detectar classes a partir das subpastas presentes
        subdirs = [p.name for p in self.diretorio_dataset.iterdir() if p.is_dir()]
        if len(subdirs) < 2:
            raise FileNotFoundError("É necessário pelo menos duas classes (subpastas) dentro do diretório do dataset.")
        self.classes_detectadas = sorted(subdirs)
        
        logger.info(f"🧠 Treinador EfficientNet inicializado")
        logger.info(f"   Dataset: {diretorio_dataset}")
        logger.info(f"   Modelo base: {modelo_base}")
        logger.info(f"   Dispositivo: {self.dispositivo}")
        logger.info(f"   Classes detectadas: {self.classes_detectadas}")
    
    def preparar_datasets(
        self,
        val_size: float = 0.2,
        test_size: float = 0.15,
        batch_size: int = 16,
        img_size: int = 224,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepara datasets de treino, validação e teste.
        
        Args:
            val_size (float): Proporção para validação
            test_size (float): Proporção para teste
            batch_size (int): Tamanho do batch
            img_size (int): Tamanho da imagem
            seed (int): Semente para randomização
            
        Returns:
            tuple: DataLoaders (train, val, test)
        """
        
        # Ajustar tamanho de entrada conforme arquitetura
        auto_img_size = img_size
        try:
            if self.modelo_base == 'efficientnet_b4':
                auto_img_size = 380
            elif self.modelo_base == 'efficientnet_b5':
                auto_img_size = 456
            elif self.modelo_base == 'resnet50':
                auto_img_size = 224
        except Exception:
            auto_img_size = img_size

        # Data augmentation para treino (OTIMIZADO PARA VELOCIDADE)
        train_transform = transforms.Compose([
            transforms.Resize((auto_img_size, auto_img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Transform para val/test (sem augmentation)
        val_test_transform = transforms.Compose([
            transforms.Resize((auto_img_size, auto_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Detectar se dataset já está em formato pré-separado (train/val/test)
        pre_split = all((self.diretorio_dataset / d).exists() for d in ["train", "val", "test"])
        if pre_split:
            # Carregar conjuntos diretamente
            train_dataset = datasets.ImageFolder(
                root=str(self.diretorio_dataset / 'train'),
                transform=train_transform
            )
            val_dataset = datasets.ImageFolder(
                root=str(self.diretorio_dataset / 'val'),
                transform=val_test_transform
            )
            test_dataset = datasets.ImageFolder(
                root=str(self.diretorio_dataset / 'test'),
                transform=val_test_transform
            )
            # Remover arquivos lixo/ocultos (por exemplo '._*.jpg' do macOS) que quebram o PIL
            def _filtrar_amostras_invalidas(ds):
                if not hasattr(ds, 'samples'):
                    return
                amostras_orig = len(ds.samples)
                ds.samples = [
                    (p, c) for (p, c) in ds.samples
                    if not Path(p).name.startswith('._')
                ]
                if hasattr(ds, 'imgs'):
                    ds.imgs = ds.samples
                removidos = amostras_orig - len(ds.samples)
                if removidos > 0:
                    logger.info(f"Removidas {removidos} imagens com prefixo '._' em {ds.root}")

            _filtrar_amostras_invalidas(train_dataset)
            _filtrar_amostras_invalidas(val_dataset)
            _filtrar_amostras_invalidas(test_dataset)
            # Validar classes consistentes
            if not (train_dataset.classes == val_dataset.classes == test_dataset.classes):
                raise ValueError("As classes em train/val/test não são consistentes.")
            self.classes_detectadas = train_dataset.classes
            logger.info(f"📚 Dataset pré-separado detectado: classes={self.classes_detectadas}")
        else:
            # Carregar dataset completo e fazer split
            dataset_completo = datasets.ImageFolder(
                root=str(self.diretorio_dataset),
                transform=train_transform
            )
            
            # Configurar seed para reproducibilidade
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Calcular tamanhos
            total_size = len(dataset_completo)
            test_size_abs = int(total_size * test_size)
            val_size_abs = int(total_size * val_size)
            train_size_abs = total_size - val_size_abs - test_size_abs
            
            # Split dos dados
            train_dataset, val_dataset, test_dataset = random_split(
                dataset_completo,
                [train_size_abs, val_size_abs, test_size_abs],
                generator=torch.Generator().manual_seed(seed)
            )
            
            # Aplicar transform correto para val/test
            val_dataset.dataset.transform = val_test_transform
            test_dataset.dataset.transform = val_test_transform
        
        # Configuração adaptativa de DataLoader
        is_mps = (self.dispositivo.type == 'mps')
        dl_kwargs_train = {
            'batch_size': batch_size,
            'shuffle': True
        }
        dl_kwargs_eval = {
            'batch_size': batch_size,
            'shuffle': False
        }
        if is_mps:
            # Evitar múltiplos workers/pin_memory no MPS (pode travar no macOS)
            dl_kwargs_train.update({'num_workers': 0, 'pin_memory': False})
            dl_kwargs_eval.update({'num_workers': 0, 'pin_memory': False})
        else:
            # Em CUDA/CPU no Windows, usar num_workers=0 reduz risco de erros em DataLoader workers
            dl_kwargs_train.update({'num_workers': 0, 'pin_memory': True})
            dl_kwargs_eval.update({'num_workers': 0, 'pin_memory': True})

        # Criar DataLoaders
        train_loader = DataLoader(train_dataset, **dl_kwargs_train)
        val_loader = DataLoader(val_dataset, **dl_kwargs_eval)
        test_loader = DataLoader(test_dataset, **dl_kwargs_eval)
        
        # Atualizar classes detectadas a partir do dataset (ordem canônica)
        if pre_split:
            self.classes_detectadas = train_dataset.classes
        else:
            self.classes_detectadas = dataset_completo.classes

        # Calcular pesos das classes para balanceamento
        if pre_split:
            # Contar amostras por classe no conjunto de treino
            class_counts = {}
            for _, label in train_dataset.samples:
                class_name = train_dataset.classes[label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        else:
            # Para dataset não pré-separado
            class_counts = {}
            for _, label in train_dataset.dataset.samples:
                class_name = train_dataset.dataset.classes[label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Calcular pesos inversamente proporcionais à frequência
        total_samples = sum(class_counts.values())
        class_weights = {}
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (len(class_counts) * count)
        
        # Criar tensor de pesos na ordem das classes
        weight_tensor = torch.tensor(
            [class_weights[class_name] for class_name in self.classes_detectadas],
            dtype=torch.float32
        )
        
        logger.info(f"📊 Dataset preparado:")
        logger.info(f"   Treino: {len(train_dataset)} amostras")
        logger.info(f"   Validação: {len(val_dataset)} amostras")
        logger.info(f"   Teste: {len(test_dataset)} amostras")
        logger.info(f"   Classes: {self.classes_detectadas}")
        logger.info(f"   Distribuição treino: {class_counts}")
        logger.info(f"   Pesos das classes: {dict(zip(self.classes_detectadas, weight_tensor.tolist()))}")
        
        return train_loader, val_loader, test_loader, weight_tensor
    
    def treinar(
        self,
        epocas: int = 30,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 10,
        batch_size: int = 16,
        callback_progresso: Optional[Callable] = None,
        diretorio_saida: str = "modelos_salvos/classificador",
        use_focal: bool = False,
        unfreeze_epoch: int = 5,
        unfreeze_lr_factor: float = 0.1,
        resume_weights_path: Optional[str] = None
    ) -> Dict:
        """
        Executa treinamento completo do classificador.
        
        Args:
            epocas (int): Número de épocas
            learning_rate (float): Learning rate inicial
            weight_decay (float): Weight decay para regularização
            patience (int): Paciência para early stopping
            batch_size (int): Tamanho do batch
            callback_progresso (callable): Callback para atualizar progresso
            diretorio_saida (str): Diretório para salvar modelos
            
        Returns:
            dict: Resultados do treinamento
        """
        
        # Atualizar estado global
        global estado_treinamento_classificador
        estado_treinamento_classificador.update({
            'treinando': True,
            'epoca_atual': 0,
            'total_epocas': epocas,
            'progresso': 0,
            'inicio_treinamento': time.time(),
            'erro': None,
            'historico': {
                'epocas': [],
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
        })

        def _persistir_status_json(dir_out: str):
            try:
                p = Path(dir_out) / 'status_treinamento.json'
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(estado_treinamento_classificador, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        
        try:
            logger.info("🚀 Iniciando treinamento do classificador EfficientNet...")
            
            # Criar diretório de saída
            Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
            _persistir_status_json(diretorio_saida)
            
            # Preparar datasets
            train_loader, val_loader, test_loader, class_weights = self.preparar_datasets(
                batch_size=batch_size
            )
            
            # Criar modelo conforme arquitetura escolhida
            if self.modelo_base.startswith('efficientnet'):
                modelo = ClassificadorSujidade(num_classes=len(self.classes_detectadas), modelo_base=self.modelo_base)
            elif self.modelo_base == 'resnet50' and ClassificadorResNet is not None:
                modelo = ClassificadorResNet(num_classes=len(self.classes_detectadas))
            else:
                # fallback
                modelo = ClassificadorSujidade(num_classes=len(self.classes_detectadas), modelo_base='efficientnet_b4')

            # Se for retomada, carregar pesos antes de iniciar o loop
            if resume_weights_path:
                try:
                    ckpt = torch.load(resume_weights_path, map_location=self.dispositivo)
                    state = ckpt.get('state_dict', ckpt)
                    modelo.modelo.load_state_dict(state, strict=False)
                    logger.info(f"🔁 Retomando a partir de: {resume_weights_path}")
                except Exception as e:
                    logger.warning(f"Não foi possível carregar pesos de retomada ({resume_weights_path}): {e}")
            modelo.modelo.train()  # Modo treino
            
            # Configurar otimizador e loss
            otimizador = optim.AdamW(
                modelo.modelo.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Loss function (CrossEntropy ou Focal)
            class_weights_dev = class_weights.to(self.dispositivo)

            class FocalLoss(nn.Module):
                def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.reduction = reduction
                    self.ce = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')
                def forward(self, inputs, targets):
                    ce_loss = self.ce(inputs, targets)
                    pt = torch.exp(-ce_loss)
                    loss = ((1 - pt) ** self.gamma) * ce_loss
                    if self.reduction == 'mean':
                        return loss.mean()
                    elif self.reduction == 'sum':
                        return loss.sum()
                    return loss

            criterion = FocalLoss(alpha=class_weights_dev) if use_focal else nn.CrossEntropyLoss(weight=class_weights_dev)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                otimizador,
                mode='min',
                factor=0.5,
                patience=5
            )
            
            # Early stopping
            melhor_val_loss = float('inf')
            melhor_val_acc = 0.0
            epocas_sem_melhora = 0
            melhor_estado_modelo = None
            es_patience = 3  # early stopping agressivo por val_acc
            
            logger.info(f"🏋️ Iniciando treinamento com {epocas} épocas...")
            
            # Loop de treinamento
            for epoca in range(epocas):
                epoca_inicio = time.time()

                # Fine-tuning: descongelar tudo após unfreeze_epoch e reduzir LR
                if epoca == unfreeze_epoch:
                    try:
                        for p in modelo.modelo.parameters():
                            p.requires_grad = True
                        for g in otimizador.param_groups:
                            g['lr'] = max(1e-6, g['lr'] * unfreeze_lr_factor)
                        logger.info(f"🔓 Fine-tuning: camadas descongeladas e LR ajustado para {otimizador.param_groups[0]['lr']:.6f}")
                    except Exception as _:
                        pass
                
                # Fase de treino
                train_loss, train_acc = self._treinar_epoca(
                    modelo.modelo, train_loader, otimizador, criterion, self.dispositivo
                )
                
                # Fase de validação
                val_loss, val_acc = self._validar_epoca(
                    modelo.modelo, val_loader, criterion, self.dispositivo
                )
                
                # Atualizar learning rate
                scheduler.step(val_loss)
                lr_atual = otimizador.param_groups[0]['lr']
                
                # Atualizar estado global
                estado_treinamento_classificador.update({
                    'epoca_atual': epoca + 1,
                    'progresso': int(((epoca + 1) / epocas) * 100),
                    'metricas': {
                        'train_loss': float(train_loss),
                        'train_acc': float(train_acc),
                        'val_loss': float(val_loss),
                        'val_acc': float(val_acc)
                    }
                })
                _persistir_status_json(diretorio_saida)
                
                # Adicionar ao histórico
                estado_treinamento_classificador['historico']['epocas'].append(epoca + 1)
                estado_treinamento_classificador['historico']['train_loss'].append(float(train_loss))
                estado_treinamento_classificador['historico']['train_acc'].append(float(train_acc))
                estado_treinamento_classificador['historico']['val_loss'].append(float(val_loss))
                estado_treinamento_classificador['historico']['val_acc'].append(float(val_acc))
                
                # Estimar tempo restante
                tempo_epoca = time.time() - epoca_inicio
                if epoca > 0:
                    tempo_restante = tempo_epoca * (epocas - epoca - 1)
                    estado_treinamento_classificador['tempo_restante_seg'] = int(tempo_restante)
                
                # Salvar melhor por loss (para restore) e early stopping agressivo por acc
                if val_loss < melhor_val_loss:
                    melhor_val_loss = val_loss
                    melhor_estado_modelo = modelo.modelo.state_dict().copy()
                    checkpoint_path = Path(diretorio_saida) / f'checkpoint_epoch_{epoca+1}.pth'
                    modelo.salvar_modelo(str(checkpoint_path))

                if val_acc > melhor_val_acc:
                    melhor_val_acc = val_acc
                    epocas_sem_melhora = 0
                else:
                    epocas_sem_melhora += 1
                
                # Log de progresso
                logger.info(f"📊 Época {epoca+1}/{epocas} ({estado_treinamento_classificador['progresso']}%)")
                logger.info(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                logger.info(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                logger.info(f"   LR: {lr_atual:.6f} | Tempo: {tempo_epoca:.1f}s")
                
                # Chamar callback externo
                if callback_progresso:
                    callback_progresso(epoca + 1, epocas, estado_treinamento_classificador['metricas'])
                
                # Verificar early stopping agressivo por val_acc
                if epocas_sem_melhora >= es_patience:
                    logger.info(f"⏹️ Early stopping (agressivo) por val_acc na época {epoca+1}")
                    break
            
            # Restaurar melhor modelo
            if melhor_estado_modelo:
                modelo.modelo.load_state_dict(melhor_estado_modelo)
                logger.info("🏆 Melhor modelo restaurado")
            
            # Avaliação final no conjunto de teste (inclui matriz de confusão)
            test_loss, test_acc, test_report, cm = self._avaliar_final(
                modelo.modelo, test_loader, criterion, self.dispositivo
            )
            
            # Salvar modelo final
            modelo_final_path = Path(diretorio_saida) / 'melhor_modelo.pth'
            modelo.salvar_modelo(str(modelo_final_path))
            
            # Finalizar estado
            estado_treinamento_classificador['treinando'] = False
            estado_treinamento_classificador['progresso'] = 100
            _persistir_status_json(diretorio_saida)
            
            # Gerar relatório (inclui matriz de confusão)
            relatorio = self._gerar_relatorio_treinamento(
                modelo_final_path, diretorio_saida, test_report, cm
            )
            
            logger.info("✅ Treinamento concluído com sucesso!")
            logger.info(f"🏆 Melhor modelo salvo em: {modelo_final_path}")
            logger.info(f"📊 Acurácia final no teste: {test_acc:.4f}")
            
            return {
                'status': 'sucesso',
                'modelo_path': str(modelo_final_path),
                'relatorio': relatorio,
                'metricas_finais': {
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                    'val_acc': estado_treinamento_classificador['metricas']['val_acc'],
                    'val_loss': estado_treinamento_classificador['metricas']['val_loss']
                },
                'historico': estado_treinamento_classificador['historico'].copy()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            
            # Atualizar estado com erro
            estado_treinamento_classificador.update({
                'treinando': False,
                'erro': str(e),
                'progresso': 0
            })
            try:
                _persistir_status_json(diretorio_saida)
            except Exception:
                pass
            
            return {
                'status': 'erro',
                'erro': str(e),
                'metricas_finais': estado_treinamento_classificador['metricas'].copy()
            }
    
    def _treinar_epoca(self, modelo, train_loader, otimizador, criterion, dispositivo):
        """Executa uma época de treinamento."""
        modelo.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (dados, targets) in enumerate(train_loader):
            dados, targets = dados.to(dispositivo), targets.to(dispositivo)

            otimizador.zero_grad()
            # Para evitar conflitos de tipo (Half vs Float) em algumas combinações de GPU/versão,
            # usamos sempre forward "normal" sem autocast.
            saidas = modelo(dados)
            loss = criterion(saidas, targets)
            loss.backward()
            otimizador.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(saidas.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def _validar_epoca(self, modelo, val_loader, criterion, dispositivo):
        """Executa uma época de validação."""
        modelo.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for dados, targets in val_loader:
                dados, targets = dados.to(dispositivo), targets.to(dispositivo)
                # Mesmo raciocínio do treino: manter forward em precisão padrão para evitar erros de tipo.
                saidas = modelo(dados)
                loss = criterion(saidas, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(saidas.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def _avaliar_final(self, modelo, test_loader, criterion, dispositivo):
        """Avaliação final no conjunto de teste (inclui matriz de confusão)."""
        modelo.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for dados, targets in test_loader:
                dados, targets = dados.to(dispositivo), targets.to(dispositivo)
                # Manter forward em precisão padrão para evitar conflitos Half/Float na loss
                saidas = modelo(dados)
                loss = criterion(saidas, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(saidas.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calcular métricas
        all_predictions_np = np.array(all_predictions)
        all_targets_np = np.array(all_targets)
        accuracy = np.mean(all_predictions_np == all_targets_np)
        
        # Classification report
        class_names = self.classes_detectadas
        report = classification_report(
            all_targets_np,
            all_predictions_np,
            target_names=class_names,
            output_dict=True
        )

        # Matriz de confusão
        cm = confusion_matrix(
            all_targets_np,
            all_predictions_np,
            labels=list(range(len(class_names)))
        )
        
        return total_loss / len(test_loader), accuracy, report, cm
    
    def _gerar_relatorio_treinamento(self, modelo_path: Path, diretorio_saida: str, test_report: Dict, confusion_mat) -> Dict:
        """Gera relatório detalhado do treinamento (inclui matriz de confusão)."""
        try:
            relatorio = {
                'resumo': {
                    'status': 'concluido',
                    'modelo_base': self.modelo_base,
                    'modelo_treinado': str(modelo_path),
                    'dataset': str(self.diretorio_dataset),
                    'dispositivo': str(self.dispositivo),
                    'epocas_treinadas': estado_treinamento_classificador['epoca_atual'],
                    'tempo_total_seg': round(time.time() - estado_treinamento_classificador['inicio_treinamento'], 2)
                },
                'metricas_finais': estado_treinamento_classificador['metricas'].copy(),
                'historico_completo': estado_treinamento_classificador['historico'].copy(),
                'metricas_teste': test_report,
                'matriz_confusao_teste': confusion_mat.tolist(),
                'hiperparametros': {
                    'batch_size': 16,
                    'learning_rate': 0.001,
                    'weight_decay': 1e-4,
                    'optimizer': 'AdamW',
                    'scheduler': 'ReduceLROnPlateau',
                    'early_stopping_patience': 10
                }
            }
            
            # Salvar relatório em JSON
            relatorio_path = Path(diretorio_saida) / 'relatorio_treinamento.json'
            with open(relatorio_path, 'w', encoding='utf-8') as f:
                json.dump(relatorio, f, indent=2, ensure_ascii=False)
            
            # Gerar gráficos de loss/acc
            self._gerar_graficos_treinamento(diretorio_saida)

            # Gerar figura da matriz de confusão
            self._gerar_matriz_confusao(diretorio_saida, confusion_mat)
            
            logger.info(f"📄 Relatório salvo: {relatorio_path}")
            return relatorio
        
        except Exception as e:
            logger.error(f"❌ Erro ao gerar relatório: {e}")
            return {'erro': str(e)}
    
    def _gerar_graficos_treinamento(self, diretorio_saida: str):
        """Gera gráficos de treinamento."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Backend não-interativo
            
            historico = estado_treinamento_classificador['historico']
            
            # Gráfico de Loss
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(historico['epocas'], historico['train_loss'], label='Train Loss', color='red')
            plt.plot(historico['epocas'], historico['val_loss'], label='Val Loss', color='blue')
            plt.title('Loss por Época')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Gráfico de Acurácia
            plt.subplot(1, 2, 2)
            plt.plot(historico['epocas'], historico['train_acc'], label='Train Acc', color='green')
            plt.plot(historico['epocas'], historico['val_acc'], label='Val Acc', color='orange')
            plt.title('Acurácia por Época')
            plt.xlabel('Época')
            plt.ylabel('Acurácia')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(Path(diretorio_saida) / 'graficos_treinamento.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Gráficos de treinamento gerados")
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar gráficos: {e}")
    
    def _gerar_matriz_confusao(self, diretorio_saida: str, confusion_mat):
        """Gera e salva a matriz de confusão do conjunto de teste."""
        try:
            import matplotlib
            matplotlib.use('Agg')

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                confusion_mat,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.classes_detectadas,
                yticklabels=self.classes_detectadas
            )
            plt.title('Matriz de confusão - teste')
            plt.xlabel('Classe predita')
            plt.ylabel('Classe verdadeira')
            plt.tight_layout()

            output_path = Path(diretorio_saida) / 'matriz_confusao_teste.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"📊 Matriz de confusão salva em: {output_path}")
        except Exception as e:
            logger.error(f"❌ Erro ao gerar matriz de confusão: {e}")

def obter_status_treinamento_classificador() -> Dict:
    """Retorna status atual do treinamento do classificador."""
    return estado_treinamento_classificador.copy()

def resetar_status_treinamento_classificador():
    """Reseta o status de treinamento do classificador."""
    global estado_treinamento_classificador
    estado_treinamento_classificador = {
        'treinando': False,
        'epoca_atual': 0,
        'total_epocas': 0,
        'progresso': 0,
        'metricas': {
            'train_loss': 0.0,
            'train_acc': 0.0,
            'val_loss': 0.0,
            'val_acc': 0.0
        },
        'historico': {
            'epocas': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        },
        'tempo_restante_seg': 0,
        'inicio_treinamento': None,
        'erro': None
    }

def iniciar_treinamento_classificador_async(
    diretorio_dataset: str,
    epocas: int = 30,
    learning_rate: float = 0.001,
    batch_size: int = 16
) -> Dict:
    """Inicia treinamento do classificador em background."""
    import threading
    
    def treinar_background():
        try:
            treinador = TreinadorClassificador(diretorio_dataset)
            resultado = treinador.treinar(
                epocas=epocas,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            logger.info("✅ Treinamento classificador background concluído")
        except Exception as e:
            logger.error(f"❌ Erro no treinamento classificador background: {e}")
            estado_treinamento_classificador['erro'] = str(e)
            estado_treinamento_classificador['treinando'] = False
    
    # Resetar estado
    resetar_status_treinamento_classificador()
    
    # Iniciar thread
    thread = threading.Thread(target=treinar_background, daemon=True)
    thread.start()
    
    return {
        'status': 'iniciado',
        'mensagem': 'Treinamento do classificador iniciado em background',
        'detalhes': {
            'dataset': diretorio_dataset,
            'epocas': epocas,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
    }

# Função fábrica
def criar_treinador(diretorio_dataset: str):
    """Função fábrica para criar treinador."""
    return TreinadorClassificador(diretorio_dataset)

# Teste rápido
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Teste do Treinador EfficientNet")
    print("Para usar em produção:")
    print("1. treinador = TreinadorClassificador('dados/modulos_individuais')")
    print("2. resultado = treinador.treinar(epocas=30)")
    
    print("✅ Treinador importado com sucesso!")
