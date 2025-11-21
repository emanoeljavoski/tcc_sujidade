"""
Aplicação Principal FastAPI - Sistema de Inspeção de Painéis Solares
Desenvolvido para TCC - Engenharia Mecatrônica

Pipeline Completo: Detecção (YOLOv8) + Classificação (EfficientNet)
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio
import json
import time
import logging
from datetime import datetime
import cv2
import numpy as np
import base64

# Imports dos modelos
from aplicacao.modelos.detector_modulos import DetectorModulos, criar_detector
from aplicacao.modelos.classificador_sujidade import ClassificadorSujidade, criar_classificador
from aplicacao.modelos.pipeline_completo import PipelineInspecao, criar_pipeline
from aplicacao.modelos.treinamento_detector import (
    iniciar_treinamento_async as treinar_detector_async,
    obter_status_treinamento as status_treinamento_detector,
    resetar_status_treinamento
)
from aplicacao.modelos.treinamento_classificador import (
    iniciar_treinamento_classificador_async as treinar_classificador_async,
    obter_status_treinamento_classificador as status_treinamento_classificador,
    resetar_status_treinamento_classificador
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração
from aplicacao.config import DIRETORIOS, LIMITES, CORS_CONFIG, MODELO

# Criar instância do FastAPI
app = FastAPI(
    title="API de Inspeção de Painéis Solares",
    description="""
    Sistema completo para inspeção automatizada de usinas solares.
    
    ## Pipeline de 2 Estágios:
    1. **Detecção**: YOLOv8 localiza todos os módulos fotovoltaicos
    2. **Classificação**: EfficientNet classifica sujidade de cada módulo
    
    Desenvolvido como TCC - Engenharia Mecatrônica
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_CONFIG["allow_origins"],
    allow_credentials=CORS_CONFIG["allow_credentials"],
    allow_methods=CORS_CONFIG["allow_methods"],
    allow_headers=CORS_CONFIG["allow_headers"],
)

# Instâncias globais dos modelos
detector_global = None
classificador_global = None
pipeline_global = None
gerador_ortomosaico_global = None
detector_sahi_global = None

# Histórico de inspeções
historico_inspecoes = []
# Status global de preparação de dataset
status_preparacao = {
    'rodando': False,
    'total_imagens': 0,
    'imagens_processadas': 0,
    'modulos_gerados': 0,
    'erros': 0,
    'inicio': None,
    'fim': None,
    'mensagem': ''
}

def criar_diretorios():
    """Cria estrutura de diretórios necessária."""
    for diretorio in DIRETORIOS.values():
        try:
            Path(diretorio).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # No Windows, WinError 183 pode ocorrer mesmo com exist_ok=True
            # Verificar se o caminho já existe e é um diretório
            p = Path(diretorio)
            if p.exists() and p.is_dir():
                pass  # Diretório já existe, OK
            else:
                logger.error(f"❌ Erro ao criar diretório {diretorio}: já existe um arquivo com esse nome")
                raise
        except PermissionError as e:
            # Ex.: drive externo/HD não pronto (D:\modelos_salvos). Não derrubar a API.
            logger.warning(f"⚠️ Permissão negada ao criar diretório {diretorio}: {e}. Pulando este diretório.")
        except OSError as e:
            # Outros erros de SO (dispositivo não pronto, etc.)
            logger.warning(f"⚠️ Erro de SO ao criar diretório {diretorio}: {e}. Pulando este diretório.")
    
    # Subdiretórios específicos
    subdirs = [
        Path(DIRETORIOS["plantas_completas"]) / "imagens" / "train",
        Path(DIRETORIOS["plantas_completas"]) / "imagens" / "val",
        Path(DIRETORIOS["plantas_completas"]) / "anotacoes",
        Path(DIRETORIOS["modulos_individuais"]) / "limpo",
        Path(DIRETORIOS["modulos_individuais"]) / "pouco sujo",
        Path(DIRETORIOS["modulos_individuais"]) / "sujo",
        Path(DIRETORIOS["modulos_individuais"]) / "muito sujo",
    ]
    for subdir in subdirs:
        try:
            subdir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            if subdir.exists() and subdir.is_dir():
                pass
            else:
                logger.error(f"❌ Erro ao criar subdiretório {subdir}: já existe um arquivo com esse nome")
                raise

def inicializar_modelos():
    """Inicializa os modelos globais."""
    global detector_global, classificador_global, pipeline_global
    
    try:
        # Verificar se existem modelos treinados
        detector_path = Path(DIRETORIOS["modelos_salvos"]) / "detector_yolo" / "best.pt"
        classificador_path = Path(DIRETORIOS["modelos_salvos"]) / "classificador" / "melhor_modelo.pth"
        usando_classificador_binario = False

        # Priorizar modelos já treinados neste Dell (paths absolutos em F:) quando os caminhos padrão ainda não existem
        alt_detector_path = Path(r"F:\\modelos_salvos\\yolo\\yolo11n_solar_dust_roboflow_v3\\weights\\best.pt")
        if not detector_path.exists() and alt_detector_path.exists():
            detector_path = alt_detector_path

        alt_classificador_path = Path(r"F:\\modelos_salvos\\classificador_2classes\\run_20251114_1620\\checkpoint_epoch_3.pth")
        if alt_classificador_path.exists():
            classificador_path = alt_classificador_path
            usando_classificador_binario = True
        
        # Inicializar detector
        if detector_path.exists():
            detector_global = criar_detector(str(detector_path))
            logger.info(f"✅ Detector YOLOv8 carregado: {detector_path}")
        else:
            detector_global = criar_detector()
            logger.info("📥 Detector YOLOv8 inicializado (modelo pré-treinado COCO)")
        
        # Inicializar classificador
        if classificador_path.exists():
            if usando_classificador_binario:
                # Checkpoint treinado com 2 classes (limpo/sujo); o mapeamento para 4 níveis
                # de sujidade é feito internamente no classificador a partir da probabilidade de "sujo".
                classificador_global = criar_classificador(str(classificador_path), num_classes=2)
            else:
                classificador_global = criar_classificador(str(classificador_path), num_classes=MODELO["classificador"]["num_classes"])
            logger.info(f"✅ Classificador EfficientNet carregado: {classificador_path}")
        else:
            classificador_global = criar_classificador(num_classes=MODELO["classificador"]["num_classes"])
            logger.info("📥 Classificador EfficientNet inicializado (ImageNet)")
        
        # Inicializar pipeline
        pipeline_global = criar_pipeline(detector_global, classificador_global)
        logger.info("🚀 Pipeline completo inicializado")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar modelos: {e}")
        return False

# Inicialização ao startup
@app.on_event("startup")
async def startup_event():
    """Executado ao iniciar a API."""
    global gerador_ortomosaico_global, detector_sahi_global
    logger.info("🌟 Iniciando API de Inspeção de Painéis Solares...")
    
    # Criar diretórios
    criar_diretorios()
    logger.info("📁 Estrutura de diretórios criada")
    
    # Inicializar modelos (pular se variável de ambiente solicitar)
    try:
        skip = os.getenv("SKIP_MODEL_INIT", "0") == "1"
    except Exception:
        skip = False
    if skip:
        logger.info("⏭️ SKIP_MODEL_INIT=1 — pulando inicialização de modelos pesados no startup")
    else:
        if inicializar_modelos():
            logger.info("✅ Modelos inicializados com sucesso")
        else:
            logger.warning("⚠️ Erro na inicialização dos modelos - API funcionará no modo limitado")
    
    # Inicializar componentes de ortomosaico v2 (OpenStitching + SAHI)
    try:
        from aplicacao.modelos.gerador_ortomosaico import GeradorOrtomosaico
        from aplicacao.modelos.detector_sahi import DetectorSAHI
        import torch

        if gerador_ortomosaico_global is None:
            gerador_ortomosaico_global = GeradorOrtomosaico()

        if detector_sahi_global is None:
            if torch.cuda.is_available():
                device_sahi = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_sahi = "mps"
            else:
                device_sahi = "cpu"

            detector_path = Path(DIRETORIOS["modelos_salvos"]) / "detector_yolo" / "best.pt"
            alt_detector_path = Path(r"F:\\modelos_salvos\\yolo\\yolo11n_solar_dust_roboflow_v3\\weights\\best.pt")
            if not detector_path.exists() and alt_detector_path.exists():
                detector_path = alt_detector_path

            if detector_path.exists():
                caminho_modelo_sahi = str(detector_path)
            else:
                caminho_modelo_sahi = "yolo11n.pt"

            detector_sahi_global = DetectorSAHI(
                caminho_modelo=caminho_modelo_sahi,
                confidence_threshold=0.25,
                device=device_sahi,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )

        logger.info("✅ Componentes de ortomosaico v2 inicializados (OpenStitching + SAHI)")
    except Exception as e:
        logger.warning(f"⚠️ Não foi possível inicializar componentes de ortomosaico v2: {e}")

# ============ FUNÇÕES AUXILIARES PARA TILING ============

def aplicar_nms_global(deteccoes: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Aplica Non-Maximum Suppression global para remover detecções duplicadas entre tiles.
    
    Args:
        deteccoes: Lista de detecções com 'bbox' e 'confianca_deteccao'
        iou_threshold: Threshold de IoU para considerar duplicatas
    
    Returns:
        Lista de detecções filtradas
    """
    if len(deteccoes) == 0:
        return []
    
    # Converter para arrays numpy
    boxes = np.array([d['bbox'] for d in deteccoes])
    # Alguns dicionários usam 'confianca', outros podem ter 'confianca_deteccao'
    scores = np.array([d.get('confianca', d.get('confianca_deteccao', 0.5)) for d in deteccoes])
    
    # Calcular áreas
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Ordenar por confiança (maior primeiro)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calcular IoU com as demais boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Manter apenas boxes com IoU < threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return [deteccoes[i] for i in keep]


def processar_imagem_com_tiling(
    caminho_imagem: str,
    confianca_min: float = 0.15,
    tile_size: int = 1280,
    overlap: int = 192
) -> Dict[str, Any]:
    """Processa imagem grande dividindo em tiles sobrepostos."""

    # Carregar imagem
    img = cv2.imread(str(caminho_imagem))
    if img is None:
        raise ValueError(f"Não foi possível carregar: {caminho_imagem}")
    
    h, w = img.shape[:2]
    
    # Se imagem pequena (<4000px), usar pipeline normal
    if max(h, w) <= 4000:
        return pipeline_global.inspecionar_planta(str(caminho_imagem), confianca_min)
    
    logger.info(f"Imagem grande ({w}x{h}). Aplicando tiling {tile_size}x{tile_size} com overlap {overlap}px")
    
    # Gerar tiles com overlap
    tiles_info = []
    stride = tile_size - overlap
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x1, y1 = x, y
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
            
            # Pular tiles muito pequenos (ajustado para tile_size menor)
            if (y2 - y1) < 320 or (x2 - x1) < 320:
                continue
            
            tile = img[y1:y2, x1:x2]
            tiles_info.append({
                'tile': tile,
                'offset': (x1, y1),
                'coords': (x1, y1, x2, y2)
            })
    
    logger.info(f"Dividido em {len(tiles_info)} tiles")
    
    # Processar cada tile
    todas_deteccoes = []
    temp_dir = Path(caminho_imagem).parent
    
    for idx, info in enumerate(tiles_info):
        tile_path = temp_dir / f"temp_tile_{idx}_{int(time.time()*1000)}.jpg"
        
        try:
            # Salvar tile
            cv2.imwrite(str(tile_path), info['tile'])
            
            # Detectar módulos no tile com parâmetros otimizados
            deteccoes_tile = detector_global.detectar(
                str(tile_path),
                confianca_min=0.15,
                imgsz=1280,
            )

            # Log detalhado por tile
            if len(deteccoes_tile) > 0:
                logger.info(f"  Tile {idx}: {len(deteccoes_tile)} módulos detectados")
            
            # Ajustar coordenadas para imagem original
            x_offset, y_offset = info['offset']
            for det in deteccoes_tile:
                # Ajustar bbox
                det['bbox'][0] += x_offset
                det['bbox'][1] += y_offset
                det['bbox'][2] += x_offset
                det['bbox'][3] += y_offset
                
                # Garantir que tenha a chave 'confianca' para NMS
                if 'confianca' not in det and 'confianca_deteccao' not in det:
                    det['confianca'] = 0.5
            
            todas_deteccoes.extend(deteccoes_tile)
            
        finally:
            if tile_path.exists():
                tile_path.unlink()
    
    logger.info(f"Detecções brutas: {len(todas_deteccoes)}")
    
    # Se não encontrou nada, retornar vazio (mas com status de sucesso para o agregador)
    if len(todas_deteccoes) == 0:
        return {
            'status': 'sucesso',
            'total_modulos': 0,
            'modulos_limpos': 0,
            'modulos_sujos': 0,
            'percentual_limpos': 0,
            'percentual_sujos': 0,
            'deteccoes': [],
            'metodo': 'tiling',
            'num_tiles': len(tiles_info),
            'contagem_classes': {},
            'distribuicao_classes': {},
            'classe_predominante': None,
            'indice_homogeneidade': 0
        }
    
    # Aplicar NMS para remover duplicatas
    deteccoes_unicas = aplicar_nms_global(todas_deteccoes, iou_threshold=0.5)
    logger.info(f"Após NMS: {len(deteccoes_unicas)}")
    
    # Classificar cada módulo
    modulos_classificados = []
    for det in deteccoes_unicas:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        
        # Garantir que bbox está dentro da imagem
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = img[y1:y2, x1:x2]
        
        # Verificar se crop é válido
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            logger.warning(f"Crop inválido: bbox=({x1},{y1},{x2},{y2})")
            continue
        
        crop_path = temp_dir / f"temp_crop_{int(time.time()*1000)}.jpg"
        try:
            cv2.imwrite(str(crop_path), crop)
            resultado = classificador_global.classificar(str(crop_path))
            
            # Mesclar resultado da classificação com detecção
            det_com_classe = det.copy()
            det_com_classe.update(resultado)
            # Normalizar chave de confiança usada pelo agregador
            if 'confianca_classificacao' not in det_com_classe:
                det_com_classe['confianca_classificacao'] = det_com_classe.get('confianca', 0.0)
            modulos_classificados.append(det_com_classe)
            
        except Exception as e:
            logger.error(f"Erro ao classificar crop: {e}")
        finally:
            if crop_path.exists():
                crop_path.unlink()
    
    # Calcular estatísticas
    if len(modulos_classificados) == 0:
        return {
            'status': 'sucesso',
            'total_modulos': 0,
            'modulos_limpos': 0,
            'modulos_sujos': 0,
            'percentual_limpos': 0,
            'percentual_sujos': 0,
            'deteccoes': [],
            'metodo': 'tiling',
            'num_tiles': len(tiles_info),
            'contagem_classes': {},
            'distribuicao_classes': {},
            'classe_predominante': None,
            'indice_homogeneidade': 0
        }
    
    limpos = sum(1 for m in modulos_classificados if m.get('classe_binaria') == 'limpo')
    sujos = len(modulos_classificados) - limpos
    
    # Contagem por nível de sujidade (4 níveis)
    contagem = {}
    for m in modulos_classificados:
        nivel = m.get('nivel_sujidade', m.get('classe', 'desconhecido'))
        contagem[nivel] = contagem.get(nivel, 0) + 1
    
    # Distribuição percentual
    distribuicao = {k: (v / len(modulos_classificados) * 100) for k, v in contagem.items()}
    
    # Classe predominante e índice de homogeneidade
    classe_predominante = max(contagem.items(), key=lambda x: x[1])[0] if contagem else None
    indice_homogeneidade = max(contagem.values()) / len(modulos_classificados) if contagem else 0
    
    return {
        'status': 'sucesso',
        'total_modulos': len(modulos_classificados),
        'modulos_limpos': limpos,
        'modulos_sujos': sujos,
        'percentual_limpos': (limpos / len(modulos_classificados) * 100),
        'percentual_sujos': (sujos / len(modulos_classificados) * 100),
        'deteccoes': modulos_classificados,
        'metodo': 'tiling',
        'num_tiles': len(tiles_info),
        'contagem_classes': contagem,
        'distribuicao_classes': distribuicao,
        'classe_predominante': classe_predominante,
        'indice_homogeneidade': indice_homogeneidade * 100
    }

# ============ ENDPOINTS PRINCIPAIS ============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Interface Web para Upload de Datasets."""
    html_content = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistema de Inspeção de Painéis Solares</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: #f3f4f6;
                min-height: 100vh;
                padding: 32px 16px;
            }
            .container {
                max-width: 1080px;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 16px;
                padding: 32px 32px 28px 32px;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
                border: 1px solid #e5e7eb;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2em;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .upload-section {
                background: #f9fafb;
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 20px;
                border: 1px solid #e5e7eb;
            }
            .upload-section h2 {
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
            }
            .file-input-wrapper {
                position: relative;
                overflow: hidden;
                display: inline-block;
                width: 100%;
            }
            .file-input-wrapper input[type=file] {
                position: absolute;
                left: -9999px;
            }
            .file-input-label {
                display: block;
                padding: 15px 30px;
                background: #667eea;
                color: white;
                border-radius: 10px;
                cursor: pointer;
                text-align: center;
                font-weight: 600;
                transition: all 0.3s;
            }
            .file-input-label:hover {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .upload-btn {
                width: 100%;
                padding: 15px;
                background: #28a745;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                margin-top: 15px;
                transition: all 0.3s;
            }
            .upload-btn:hover {
                background: #218838;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4);
            }
            .upload-btn:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                display: none;
            }
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .file-count {
                margin-top: 10px;
                color: #666;
                font-size: 0.9em;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 30px;
            }
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
            }
            .stat-card h3 {
                font-size: 2em;
                margin-bottom: 5px;
            }
            .stat-card p {
                opacity: 0.9;
            }
            .progress-container {
                margin-top: 15px;
            }
            .progress-bar-outer {
                width: 100%;
                height: 10px;
                background: #e9ecef;
                border-radius: 999px;
                overflow: hidden;
            }
            .progress-bar-inner {
                height: 100%;
                width: 0%;
                background: #667eea;
                transition: width 0.2s ease;
            }
            .progress-label {
                margin-top: 6px;
                font-size: 0.9em;
                color: #555;
                text-align: right;
            }
            .links {
                margin-top: 30px;
                padding-top: 30px;
                border-top: 2px solid #eee;
                text-align: center;
            }
            .links a {
                display: inline-block;
                margin: 0 10px;
                padding: 10px 20px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                transition: all 0.3s;
            }
            .links a:hover {
                background: #5568d3;
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sistema de Inspeção de Painéis Solares</h1>
            <p class="subtitle">Trabalho de Conclusão de Curso – Engenharia Mecatrônica | Pipeline: YOLOv8 + EfficientNet</p>
            
            <!-- Inspeção de planta (interface principal do TCC) -->
            <div class="upload-section">
                <h2>Inspeção de Planta</h2>
                <p>Envie uma ortofoto ou um conjunto de imagens de drone. O sistema gera o ortomosaico (quando necessário),
                detecta os módulos fotovoltaicos, classifica o nível de sujidade (quatro níveis) e gera o mapa de homogeneidade da planta.</p>
                <div class="file-input-wrapper">
                    <input type="file" id="fileInspecao" accept="image/*" multiple>
                    <label for="fileInspecao" class="file-input-label">
                        Selecionar imagem da planta (uma ou várias do drone)
                    </label>
                </div>
                <button class="upload-btn" onclick="inspecionar()" id="btnInspecionar" disabled>
                    Executar inspeção
                </button>
                <div class="status" id="statusInspecao"></div>
                <div class="progress-container" id="progressContainer" style="display:none">
                    <div class="progress-bar-outer">
                        <div class="progress-bar-inner" id="progressBar"></div>
                    </div>
                    <div class="progress-label" id="progressLabel">0%</div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:15px">
                    <div>
                        <h3>Imagem Anotada</h3>
                        <img id="imgAnotada" style="width:100%;border-radius:10px" />
                    </div>
                    <div>
                        <h3>Mapa de Homogeneidade</h3>
                        <img id="imgMapa" style="width:100%;border-radius:10px" />
                    </div>
                </div>
                <div style="margin-top:20px;border-top:1px solid #eee;padding-top:15px">
                    <h3>Resumo Numérico da Inspeção</h3>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:10px;font-size:0.95em">
                        <div>
                            <strong>Total de módulos</strong>
                            <div id="resTotalModulos">--</div>
                        </div>
                        <div>
                            <strong>Distribuição por classe</strong>
                            <div id="resDistribuicao">--</div>
                        </div>
                        <div>
                            <strong>Índice de homogeneidade</strong>
                            <div id="resIndiceHomog">--</div>
                        </div>
                        <div>
                            <strong>Classe predominante</strong>
                            <div id="resClassePredominante">--</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="links">
                <a href="/docs" target="_blank">Documentação da API</a>
                <a href="/health" target="_blank">Status do sistema</a>
            </div>
        </div>
        
        <script>
            // Inspeção — única funcionalidade da interface principal
            document.getElementById('fileInspecao').addEventListener('change', e => {
                document.getElementById('btnInspecionar').disabled = e.target.files.length === 0;
            });

            let progressInterval = null;

            function startProgress() {
                const container = document.getElementById('progressContainer');
                const bar = document.getElementById('progressBar');
                const label = document.getElementById('progressLabel');
                if (!container || !bar || !label) return;

                let value = 0;
                container.style.display = 'block';
                bar.style.width = '0%';
                label.textContent = '0%';

                if (progressInterval) clearInterval(progressInterval);
                progressInterval = setInterval(() => {
                    if (value < 90) {
                        value += 2;
                        bar.style.width = value + '%';
                        label.textContent = value + '%';
                    }
                }, 200);
            }

            function finishProgress(success) {
                const container = document.getElementById('progressContainer');
                const bar = document.getElementById('progressBar');
                const label = document.getElementById('progressLabel');
                if (progressInterval) {
                    clearInterval(progressInterval);
                    progressInterval = null;
                }
                if (!container || !bar || !label) return;

                if (success) {
                    bar.style.width = '100%';
                    label.textContent = '100%';
                } else {
                    bar.style.width = '0%';
                    label.textContent = '0%';
                }
            }

            async function inspecionar() {
                const input = document.getElementById('fileInspecao');
                const statusDiv = document.getElementById('statusInspecao');
                const btn = document.getElementById('btnInspecionar');
                const files = input.files;
                if (!files || !files.length) return;

                const fd = new FormData();
                let url = '/inspecionar-planta';
                let sucesso = false;

                if (files.length === 1) {
                    // Uma única ortofoto/planta completa
                    fd.append('file', files[0]);
                } else {
                    // Várias imagens de drone → gerar ortomosaico no backend (pipeline v2: OpenStitching + SAHI)
                    for (let f of files) {
                        fd.append('files', f);
                    }
                    url = '/inspecionar-planta-ortomosaico-v2';
                }

                fd.append('confianca_min', '0.5');
                btn.disabled = true;
                btn.textContent = 'Inspecionando...';
                statusDiv.style.display = 'none';
                startProgress();

                try {
                    const res = await fetch(url, { method: 'POST', body: fd });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.detail || 'Erro na inspeção');
                    sucesso = true;
                    statusDiv.className = 'status success';
                    statusDiv.textContent = `Inspeção concluída: ${data.total_modulos} módulos`;
                    statusDiv.style.display = 'block';
                    if (data.imagem_anotada) document.getElementById('imgAnotada').src = data.imagem_anotada;
                    if (data.mapa_homogeneidade) document.getElementById('imgMapa').src = data.mapa_homogeneidade;

                    // Atualizar resumo numérico
                    const total = data.total_modulos ?? 0;
                    const contagem = data.contagem_classes || {};
                    const dist = data.distribuicao_classes || {};
                    const classePred = data.classe_predominante ?? '--';
                    const indiceHom = data.indice_homogeneidade ?? 0;

                    const elTotal = document.getElementById('resTotalModulos');
                    const elDist = document.getElementById('resDistribuicao');
                    const elIdx = document.getElementById('resIndiceHomog');
                    const elClasse = document.getElementById('resClassePredominante');

                    if (elTotal) elTotal.textContent = `${total}`;

                    if (elDist) {
                        const partes = [];
                        const ordem = ['limpo', 'pouco sujo', 'sujo', 'muito sujo'];
                        for (const cls of ordem) {
                            const n = contagem[cls] ?? 0;
                            const p = dist[cls] ?? 0;
                            partes.push(`${cls}: ${n} (${p}%)`);
                        }
                        elDist.textContent = partes.join(' | ');
                    }

                    if (elIdx) elIdx.textContent = `${(indiceHom * 100).toFixed(1)}%`;
                    if (elClasse) elClasse.textContent = `${classePred || '--'}`;
                } catch (e) {
                    statusDiv.className = 'status error';
                    statusDiv.textContent = 'Erro: ' + e.message;
                    statusDiv.style.display = 'block';
                } finally {
                    finishProgress(sucesso);
                    btn.disabled = false;
                    btn.textContent = 'Executar inspeção';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Verificação de saúde da API."""
    status_modelos = {
        "detector_inicializado": detector_global is not None,
        "classificador_inicializado": classificador_global is not None,
        "pipeline_pronto": pipeline_global is not None
    }
    
    return {
        "status": "saudavel" if all(status_modelos.values()) else "parcial",
        "modelos": status_modelos,
        "timestamp": datetime.now().isoformat()
    }

# ============ DASHBOARD DE TREINAMENTO ============

@app.get("/dashboard-treinamento", response_class=HTMLResponse)
async def dashboard_treinamento():
    """Página dedicada de monitoramento do treinamento do classificador."""
    html = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Dashboard de Treinamento</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:20px }
        .wrap { max-width: 900px; margin: 0 auto; }
        h1 { margin: 10px 0 20px; font-size: 28px }
        .card { background:#111827; border-radius:14px; padding:20px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); margin-bottom:16px }
        .row { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px }
        .stat { background: linear-gradient(135deg,#1f2937,#111827); border-radius:12px; padding:16px; text-align:center }
        .stat h3 { margin:0 0 6px; font-size:22px; color:#fff }
        .stat p { margin:0; color:#94a3b8 }
        .bar { height:14px; background:#334155; border-radius:9px; overflow:hidden }
        .bar>div { height:100%; width:0%; background: linear-gradient(90deg,#22c55e,#84cc16); transition: width .3s ease }
        .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px }
        .muted { color:#94a3b8; font-size: 13px }
        .sep { display:flex; justify-content:space-between; margin-top:6px }
        a { color:#60a5fa }
      </style>
    </head>
    <body>
      <div class="wrap">
        <h1>🏋️ Dashboard de Treinamento — Classificador</h1>

        <div class="card">
          <div class="bar"><div id="progress"></div></div>
          <div class="sep">
            <div id="epoch" class="muted">Época: 0/0</div>
            <div id="eta" class="muted">ETA: --</div>
          </div>
        </div>

        <div class="row">
          <div class="stat"><h3 id="train_loss">--</h3><p>Train Loss</p></div>
          <div class="stat"><h3 id="train_acc">--</h3><p>Train Acc</p></div>
          <div class="stat"><h3 id="val_loss">--</h3><p>Val Loss</p></div>
          <div class="stat"><h3 id="val_acc">--</h3><p>Val Acc</p></div>
        </div>

        <div class="card">
          <div class="grid2">
            <div>
              <div class="muted">Status</div>
              <div id="status_txt">--</div>
            </div>
            <div>
              <div class="muted">Erros</div>
              <div id="err_txt">--</div>
            </div>
          </div>
          <div class="muted" style="margin-top:10px">API: <a href="/status-treinamento-classificador" target="_blank">/status-treinamento-classificador</a></div>
        </div>
      </div>

      <script>
        const fmt = (v, d=4) => (typeof v === 'number' ? v.toFixed(d) : '--');
        const pct = v => (typeof v === 'number' ? (v*100).toFixed(2)+'%' : '--');
        function setTxt(id,val){ const el=document.getElementById(id); if(el) el.textContent = val; }
        function setProgress(p){ const el=document.getElementById('progress'); if(el) el.style.width = Math.max(0,Math.min(100,p||0))+'%'; }
        function setEpoch(cur,total){ setTxt('epoch', `Época: ${cur||0}/${total||0}`); }
        function setETA(sec){ if(!sec){ setTxt('eta','ETA: --'); return; } const h=String(Math.floor(sec/3600)).padStart(2,'0'); const m=String(Math.floor((sec%3600)/60)).padStart(2,'0'); const s=String(Math.floor(sec%60)).padStart(2,'0'); setTxt('eta', `ETA: ${h}:${m}:${s}`); }

        async function poll(){
          try{
            const r = await fetch('/status-treinamento-classificador');
            const s = await r.json();
            if(s && typeof s==='object'){
              setProgress(s.progresso||0);
              setEpoch(s.epoca_atual, s.total_epocas);
              setETA(s.tempo_restante_seg);
              const m = s.metricas || {};
              setTxt('train_loss', fmt(m.train_loss));
              setTxt('train_acc', pct(m.train_acc));
              setTxt('val_loss', fmt(m.val_loss));
              setTxt('val_acc', pct(m.val_acc));
              setTxt('status_txt', (s.treinando ? 'Treinando' : 'Inativo'));
              setTxt('err_txt', s.erro || '—');
            }
          }catch(e){
            setTxt('status_txt','Falha ao consultar status');
            setTxt('err_txt', String(e));
          }
        }
        poll();
        setInterval(poll, 4000);
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# ============ ENDPOINTS DE UPLOAD ============

@app.post("/upload-plantas")
async def upload_plantas(
    files: List[UploadFile] = File(...),
    condicao: str = Form(...)
):
    """
    Upload de fotos de plantas completas (UFV, telhados) para treinamento.
    
    Args:
        files: Lista de arquivos de imagem
        condicao: Condição das plantas ('sujo' ou 'lavado')
        
    Returns:
        Dict: Status do upload
    """
    try:
        # Validar condição
        if condicao not in ['sujo', 'lavado']:
            raise HTTPException(status_code=400, detail="Condição deve ser 'sujo' ou 'lavado'")
        
        arquivos_salvos = []
        erros = []
        
        # Salvar em subpastas por condição
        destino_dir = Path(DIRETORIOS["plantas_completas"]) / "imagens" / condicao
        destino_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            # Validar tipo de arquivo
            if not file.content_type.startswith('image/'):
                erros.append(f"{file.filename}: tipo inválido")
                continue
            
            # Validar tamanho (se disponível)
            tam = getattr(file, 'size', None)
            if tam and tam > LIMITES["tamanho_maximo_arquivo"]:
                erros.append(f"{file.filename}: arquivo muito grande")
                continue
            
            # Salvar arquivo
            caminho_destino = destino_dir / file.filename
            
            # Evitar sobreescrever
            contador = 1
            while caminho_destino.exists():
                nome_base = Path(file.filename).stem
                extensao = Path(file.filename).suffix
                caminho_destino = destino_dir / f"{nome_base}_{contador}{extensao}"
                contador += 1
            
            with open(caminho_destino, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            arquivos_salvos.append(file.filename)
        
        logger.info(f"📸 Upload de plantas {condicao}: {len(arquivos_salvos)} arquivos salvos")
        
        return {
            "mensagem": f"{len(arquivos_salvos)} imagens de plantas '{condicao}' carregadas",
            "condicao": condicao,
            "arquivos_salvos": arquivos_salvos,
            "erros": erros,
            "destino": str(destino_dir)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no upload de plantas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-plantas-auto")
async def upload_plantas_auto(
    files: List[UploadFile] = File(...),
    iniciar_treino: bool = Form(True)
):
    """
    Upload de plantas sem separação manual. Usa heurística de nome do arquivo para
    rotular automaticamente como 'sujo' ou 'lavado' e, opcionalmente, inicia
    preparação de módulos e treinamento do classificador.
    """
    try:
        base_dir = Path(DIRETORIOS["plantas_completas"]) / "imagens"
        unificado_dir = base_dir / "unificado"
        sujo_dir = base_dir / "sujo"
        lavado_dir = base_dir / "lavado"
        indef_dir = base_dir / "indefinido"
        for d in [unificado_dir, sujo_dir, lavado_dir, indef_dir]:
            d.mkdir(parents=True, exist_ok=True)

        sujo_kw = ["sujo", "dirty", "poeira", "dust", "antes", "before", "pre_"]
        lavado_kw = ["lavado", "limpo", "clean", "depois", "after", "post_"]

        cont = {"sujo": 0, "lavado": 0, "indefinido": 0}
        erros = []

        # Salvar e classificar por nome
        for file in files:
            if not file.content_type.startswith('image/'):
                erros.append(f"{file.filename}: tipo inválido")
                continue

            tam = getattr(file, 'size', None)
            if tam and tam > LIMITES["tamanho_maximo_arquivo"]:
                erros.append(f"{file.filename}: arquivo muito grande")
                continue

            destino = unificado_dir / file.filename
            contador = 1
            while destino.exists():
                nome_base = Path(file.filename).stem
                extensao = Path(file.filename).suffix
                destino = unificado_dir / f"{nome_base}_{contador}{extensao}"
                contador += 1

            with open(destino, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            low = destino.name.lower()
            if any(k in low for k in sujo_kw):
                alvo = sujo_dir / destino.name
                destino.replace(alvo)
                cont["sujo"] += 1
            elif any(k in low for k in lavado_kw):
                alvo = lavado_dir / destino.name
                destino.replace(alvo)
                cont["lavado"] += 1
            else:
                alvo = indef_dir / destino.name
                destino.replace(alvo)
                cont["indefinido"] += 1

        logger.info(f"📸 Upload AUTO: {cont} (indefinidos serão ignorados na preparação)")

        # Pipeline automático: preparar módulos e treinar
        iniciou_preparacao = False
        iniciou_treino = False

        def _tarefa_preparacao_local():
            try:
                status_preparacao.update({
                    'rodando': True,
                    'inicio': time.time(),
                    'imagens_processadas': 0,
                    'modulos_gerados': 0,
                    'erros': 0,
                    'mensagem': 'Iniciando preparação...'
                })
                imagens = []
                for cond in ['sujo', 'lavado']:
                    d = base_dir / cond
                    if d.exists():
                        imagens.extend([str(p) for p in d.glob("**/*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
                status_preparacao['total_imagens'] = len(imagens)
                for idx, img_path in enumerate(imagens):
                    try:
                        cond = 'sujo' if '/sujo/' in img_path or img_path.endswith('/sujo') else 'lavado'
                        alvo = 'sujo' if cond == 'sujo' else 'limpo'
                        deteccoes = detector_global.detectar(img_path, LIMITES.get('confianca_minima_padrao', 0.5))
                        salvar_dir = Path(DIRETORIOS["modulos_individuais"]) / alvo
                        antes = len(list(salvar_dir.glob('*.jpg')))
                        detector_global.recortar_modulos(img_path, deteccoes, str(salvar_dir))
                        depois = len(list(salvar_dir.glob('*.jpg')))
                        status_preparacao['modulos_gerados'] += max(0, depois - antes)
                    except Exception as e:
                        status_preparacao['erros'] += 1
                        logger.error(f"Erro preparando {img_path}: {e}")
                    finally:
                        status_preparacao['imagens_processadas'] = idx + 1
                status_preparacao.update({
                    'rodando': False,
                    'fim': time.time(),
                    'mensagem': 'Preparação concluída'
                })
            except Exception as e:
                status_preparacao.update({'rodando': False, 'mensagem': f'Erro: {e}'})
                logger.error(f"❌ Erro na preparação: {e}")

        def _tarefa_treino_apos_preparacao():
            try:
                # Esperar preparação terminar
                while status_preparacao.get('rodando', False):
                    time.sleep(2)
                # Checar classes e iniciar treino
                modulos_dir = Path(DIRETORIOS["modulos_individuais"])
                classes = [p.name for p in modulos_dir.iterdir() if p.is_dir()]
                contagens = {c: len(list((modulos_dir / c).glob("*.jpg"))) for c in classes}
                insuficientes = {c: n for c, n in contagens.items() if n < 10}
                if insuficientes:
                    logger.warning(f"⚠️ Amostras insuficientes, não iniciando treino: {insuficientes}")
                    return
                _ = treinar_classificador_async(str(modulos_dir))
                logger.info("✅ Treinamento do classificador iniciado automaticamente")
            except Exception as e:
                logger.error(f"❌ Erro ao iniciar treino automático: {e}")

        if iniciar_treino:
            # Disparar em background
            from threading import Thread
            Thread(target=_tarefa_preparacao_local, daemon=True).start()
            iniciou_preparacao = True
            Thread(target=_tarefa_treino_apos_preparacao, daemon=True).start()
            iniciou_treino = True

        return {
            "mensagem": "Upload auto concluído",
            "contagem": cont,
            "indefinidos": cont["indefinido"],
            "iniciou_preparacao": iniciou_preparacao,
            "iniciou_treino": iniciou_treino
        }
    
    except Exception as e:
        logger.error(f"❌ Erro no upload auto: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-modulos")
async def upload_modulos(
    files: List[UploadFile] = File(...),
    classe: str = Form(...)
):
    """
    Upload de módulos individuais para treinamento do classificador.
    
    Args:
        files: Lista de arquivos de imagem
        classe: Classe dos módulos ('limpo' ou 'sujo')
        
    Returns:
        Dict: Status do upload
    """
    try:
        # Validar classe
        classes_validas = ['limpo', 'pouco sujo', 'sujo', 'muito sujo']
        if classe not in classes_validas:
            raise HTTPException(status_code=400, detail=f"Classe deve ser uma de: {classes_validas}")
        
        arquivos_salvos = []
        erros = []
        
        destino_dir = Path(DIRETORIOS["modulos_individuais"]) / classe
        
        for file in files:
            # Validar tipo de arquivo
            if not file.content_type.startswith('image/'):
                erros.append(f"{file.filename}: tipo inválido")
                continue
            
            # Validar tamanho (se disponível)
            tam = getattr(file, 'size', None)
            if tam and tam > LIMITES["tamanho_maximo_arquivo"]:
                erros.append(f"{file.filename}: arquivo muito grande")
                continue
            
            # Salvar arquivo
            caminho_destino = destino_dir / file.filename
            
            # Evitar sobreescrever
            contador = 1
            while caminho_destino.exists():
                nome_base = Path(file.filename).stem
                extensao = Path(file.filename).suffix
                caminho_destino = destino_dir / f"{nome_base}_{contador}{extensao}"
                contador += 1
            
            with open(caminho_destino, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            arquivos_salvos.append(file.filename)
        
        logger.info(f"🖼️ Upload de módulos {classe}: {len(arquivos_salvos)} arquivos salvos")
        
        return {
            "mensagem": f"{len(arquivos_salvos)} módulos '{classe}' carregados",
            "classe": classe,
            "arquivos_salvos": arquivos_salvos,
            "erros": erros,
            "destino": str(destino_dir)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no upload de módulos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/salvar-anotacao")
async def salvar_anotacao(
    arquivo_imagem: str = Form(...),
    bounding_boxes: str = Form(...)
):
    """
    Salva anotações de bounding boxes em formato YOLO.
    
    Args:
        arquivo_imagem: Nome do arquivo de imagem
        bounding_boxes: JSON com lista de bounding boxes
        
    Returns:
        Dict: Status do salvamento
    """
    try:
        # Parsear bounding boxes
        boxes = json.loads(bounding_boxes)
        
        # Diretório de anotações
        anotacoes_dir = Path(DIRETORIOS["plantas_completas"]) / "anotacoes"
        arquivo_anotacao = anotacoes_dir / f"{Path(arquivo_imagem).stem}.txt"
        
        # Converter para formato YOLO
        yolo_lines = []
        for box in boxes:
            # Format: classe x_center y_center width height (normalizados 0-1)
            linha = f"0 {box['x']:.6f} {box['y']:.6f} {box['width']:.6f} {box['height']:.6f}"
            yolo_lines.append(linha)
        
        # Salvar anotação
        with open(arquivo_anotacao, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        logger.info(f"✏️ Anotação salva: {arquivo_anotacao} ({len(yolo_lines)} boxes)")
        
        return {
            "mensagem": "Anotação salva com sucesso",
            "arquivo_anotacao": str(arquivo_anotacao),
            "num_boxes": len(yolo_lines)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao salvar anotação: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plantas-anotadas")
async def listar_plantas_anotadas():
    """Lista plantas enviadas (sujas e lavadas)."""
    try:
        plantas_dir = Path(DIRETORIOS["plantas_completas"]) / "imagens"
        
        resultado = {
            "sujo": [],
            "lavado": [],
            "total_sujo": 0,
            "total_lavado": 0
        }
        
        # Listar plantas sujas
        sujo_dir = plantas_dir / "sujo"
        if sujo_dir.exists():
            for img_path in sujo_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    resultado["sujo"].append({
                        "nome": img_path.name,
                        "caminho": str(img_path),
                        "tamanho_mb": round(img_path.stat().st_size / (1024*1024), 2)
                    })
            resultado["total_sujo"] = len(resultado["sujo"])
        
        # Listar plantas lavadas
        lavado_dir = plantas_dir / "lavado"
        if lavado_dir.exists():
            for img_path in lavado_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    resultado["lavado"].append({
                        "nome": img_path.name,
                        "caminho": str(img_path),
                        "tamanho_mb": round(img_path.stat().st_size / (1024*1024), 2)
                    })
            resultado["total_lavado"] = len(resultado["lavado"])
        
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Erro ao listar plantas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preparar-modulos")
async def preparar_modulos(background_tasks: BackgroundTasks):
    """Prepara dataset de módulos recortando as plantas (sujo→sujo, lavado→limpo)."""
    if detector_global is None:
        raise HTTPException(status_code=503, detail="Detector não inicializado")

    def tarefa_preparacao():
        try:
            status_preparacao.update({
                'rodando': True,
                'inicio': time.time(),
                'imagens_processadas': 0,
                'modulos_gerados': 0,
                'erros': 0,
                'mensagem': 'Iniciando preparação...'
            })
            base = Path(DIRETORIOS["plantas_completas"]) / "imagens"
            imagens = []
            for cond in ['sujo', 'lavado']:
                d = base / cond
                if d.exists():
                    imagens.extend([str(p) for p in d.glob("**/*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
            status_preparacao['total_imagens'] = len(imagens)
            for idx, img_path in enumerate(imagens):
                try:
                    cond = 'sujo' if '/sujo/' in img_path or img_path.endswith('/sujo') else 'lavado'
                    alvo = 'sujo' if cond == 'sujo' else 'limpo'
                    deteccoes = detector_global.detectar(img_path, LIMITES.get('confianca_minima_padrao', 0.5))
                    salvar_dir = Path(DIRETORIOS["modulos_individuais"]) / alvo
                    antes = len(list(salvar_dir.glob('*.jpg')))
                    detector_global.recortar_modulos(img_path, deteccoes, str(salvar_dir))
                    depois = len(list(salvar_dir.glob('*.jpg')))
                    status_preparacao['modulos_gerados'] += max(0, depois - antes)
                except Exception as e:
                    status_preparacao['erros'] += 1
                    logger.error(f"Erro preparando {img_path}: {e}")
                finally:
                    status_preparacao['imagens_processadas'] = idx + 1
            status_preparacao.update({
                'rodando': False,
                'fim': time.time(),
                'mensagem': 'Preparação concluída'
            })
        except Exception as e:
            status_preparacao.update({'rodando': False, 'mensagem': f'Erro: {e}'})
            logger.error(f"❌ Erro na preparação: {e}")

    background_tasks.add_task(tarefa_preparacao)
    return {"status": "iniciado", "mensagem": "Preparação iniciada em background"}

@app.get("/status-preparacao")
async def obter_status_preparacao():
    """Retorna status da preparação de módulos."""
    return status_preparacao

@app.get("/listar-modulos")
async def listar_modulos():
    """Lista contagem de módulos por classe."""
    try:
        base = Path(DIRETORIOS["modulos_individuais"]) 
        classes = ['limpo', 'pouco sujo', 'sujo', 'muito sujo']
        contagem = {}
        for c in classes:
            d = base / c
            contagem[c] = len([p for p in d.glob('*.jpg')]) if d.exists() else 0
        return {
            'classes': classes,
            'contagem': contagem,
            'total': sum(contagem.values())
        }
    except Exception as e:
        logger.error(f"❌ Erro ao listar módulos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ ENDPOINTS DE TREINAMENTO ============

@app.post("/treinar-detector")
async def treinar_detector(
    background_tasks: BackgroundTasks,
    epocas: int = Form(50),
    batch_size: int = Form(16),
    learning_rate: float = Form(0.01)
):
    """Inicia treinamento do detector YOLOv8 em background."""
    try:
        # Verificar se dataset está preparado
        dataset_yaml = Path(DIRETORIOS["plantas_completas"]) / "dataset.yaml"
        if not dataset_yaml.exists():
            raise HTTPException(
                status_code=400,
                detail="Dataset YAML não encontrado. Execute upload de plantas e anotações primeiro."
            )
        
        # Verificar se não há treinamento em andamento
        status_atual = status_treinamento_detector()
        if status_atual['treinando']:
            raise HTTPException(
                status_code=409,
                detail="Treinamento do detector já em andamento"
            )
        
        # Iniciar treinamento em background
        resultado = treinar_detector_async(
            str(dataset_yaml),
            epocas=epocas,
            batch_size=batch_size,
            lr=learning_rate
        )
        
        logger.info(f"🎯 Treinamento do detector iniciado: {epocas} épocas")
        
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar treinamento do detector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status-treinamento-detector")
async def obter_status_treinamento_detector():
    """Retorna status atual do treinamento do detector."""
    try:
        status = status_treinamento_detector()
        return status
    except Exception as e:
        logger.error(f"❌ Erro ao obter status do detector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/treinar-classificador")
async def treinar_classificador(
    background_tasks: BackgroundTasks,
    epocas: int = Form(30),
    learning_rate: float = Form(0.001),
    batch_size: int = Form(16),
    dataset_path: Optional[str] = Form(None)
):
    """Inicia treinamento do classificador EfficientNet em background."""
    try:
        # Se dataset_path fornecido, usar diretório informado (pré-separado train/val/test) e pular validações locais
        if dataset_path:
            ds_path = Path(dataset_path)
            if not ds_path.exists():
                raise HTTPException(status_code=400, detail=f"Dataset informado não existe: {dataset_path}")
            base_treino = ds_path
        else:
            # Caso contrário, usar dados/modulos_individuais (N classes) com checagens mínimas
            modulos_dir = Path(DIRETORIOS["modulos_individuais"])
            if not modulos_dir.exists():
                raise HTTPException(status_code=400, detail="Diretório de módulos não encontrado.")
            classes = [p.name for p in modulos_dir.iterdir() if p.is_dir()]
            if len(classes) < 2:
                raise HTTPException(status_code=400, detail="É necessário pelo menos 2 classes (subpastas) em 'modulos_individuais'.")
            contagens = {c: len(list((modulos_dir / c).glob("*.jpg"))) for c in classes}
            insuficientes = {c: n for c, n in contagens.items() if n < 10}
            if insuficientes:
                raise HTTPException(status_code=400, detail=f"Amostras insuficientes: {insuficientes} (mínimo: 10 por classe)")
            base_treino = modulos_dir
        
        # Verificar se não há treinamento em andamento
        status_atual = status_treinamento_classificador()
        if status_atual['treinando']:
            raise HTTPException(
                status_code=409,
                detail="Treinamento do classificador já em andamento"
            )
        
        # Iniciar treinamento em background
        resultado = treinar_classificador_async(
            str(base_treino),
            epocas=epocas,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        logger.info(f"🧠 Treinamento do classificador iniciado: {epocas} épocas")
        
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar treinamento do classificador: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status-treinamento-classificador")
async def obter_status_treinamento_classificador():
    """Retorna status atual do treinamento do classificador."""
    try:
        status = status_treinamento_classificador()

        # Garantir compatibilidade de chave para ETA
        if isinstance(status, dict):
            try:
                if 'tempo_restante_sec' not in status and 'tempo_restante_seg' in status:
                    status['tempo_restante_sec'] = status.get('tempo_restante_seg')
            except Exception:
                pass

        # Se status em memória estiver "vazio", tentar ler do arquivo JSON persistido
        def _parece_vazio(s: dict) -> bool:
            try:
                return not s.get('treinando') and s.get('total_epocas', 0) == 0 and s.get('progresso', 0) == 0
            except Exception:
                return False

        if _parece_vazio(status):
            import json
            from pathlib import Path

            candidatos = []
            # 1) Variável de ambiente explícita
            p_env = os.getenv('MODEL_STATUS_PATH')
            if p_env:
                candidatos.append(Path(p_env))
            # 2) Caminho absoluto conhecido do projeto (Z Slim)
            candidatos.append(Path('/Volumes/Z Slim/modelos_salvos/classificador_augmented/status_treinamento.json'))
            # 3) Caminho relativo padrão do repo
            candidatos.append(Path('modelos_salvos/classificador_augmented/status_treinamento.json'))

            for p in candidatos:
                try:
                    if p.is_file():
                        with open(p, 'r', encoding='utf-8') as f:
                            arquivo = json.load(f)
                        if isinstance(arquivo, dict):
                            if 'tempo_restante_sec' not in arquivo and 'tempo_restante_seg' in arquivo:
                                arquivo['tempo_restante_sec'] = arquivo.get('tempo_restante_seg')
                            return arquivo
                except Exception:
                    continue

        return status
    except Exception as e:
        logger.error(f"❌ Erro ao obter status do classificador: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ ENDPOINTS DE INSPEÇÃO ============

@app.post("/inspecionar-planta")
async def inspecionar_planta(file: UploadFile = File(...), confianca_min: float = Form(0.5)):
    """
    Executa pipeline completo de inspeção em uma planta.
    
    Args:
        file: Imagem da planta para inspecionar
        confianca_min: Confiança mínima para detecção
        
    Returns:
        Dict: Resultado completo da inspeção
    """
    try:
        # Verificar se pipeline está pronto
        if pipeline_global is None:
            raise HTTPException(
                status_code=503,
                detail="Pipeline não inicializado. Aguarde inicialização complete."
            )
        
        # Validar arquivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        # Salvar cópia permanente para aprendizado
        base_dir = Path(DIRETORIOS["resultado_inspecoes"]) / "entradas_planta"
        base_dir.mkdir(parents=True, exist_ok=True)
        caminho_entrada = base_dir / f"planta_{int(time.time())}_{file.filename}"
        
        with open(caminho_entrada, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Executar inspeção usando a imagem persistida
        resultado = pipeline_global.inspecionar_planta(str(caminho_entrada), confianca_min)
        
        # Adicionar ao histórico
        if resultado['status'] == 'sucesso':
            registro_historico = {
                "id": f"inspec_{int(time.time())}",
                "timestamp": resultado['timestamp'],
                "imagem_original": file.filename,
                "total_modulos": resultado['total_modulos'],
                "modulos_sujos": resultado['modulos_sujos'],
                "percentual_sujos": resultado['percentual_sujos'],
                "confianca_media": resultado['confianca_media'],
                "tempo_processamento": resultado['tempo_processamento']
            }
            historico_inspecoes.append(registro_historico)
            
            # Manter apenas últimas 50 inspeções no histórico
            if len(historico_inspecoes) > 50:
                historico_inspecoes.pop(0)
        
        logger.info(f"🔍 Inspeção concluída: {resultado['total_modulos']} módulos")
        
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na inspeção: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inspecionar-planta-ortomosaico")
async def inspecionar_planta_ortomosaico(
    files: List[UploadFile] = File(...),
    confianca_min: float = Form(0.5)
):
    """Gera ortomosaico a partir de múltiplas imagens e executa o pipeline completo.

    Fluxo:
        1. Salva temporariamente as imagens enviadas.
        2. Usa o PipelineInspecao para gerar um ortomosaico simples (panorama).
        3. Executa inspecionar_planta() do pipeline sobre o ortomosaico.
    """
    try:
        if pipeline_global is None:
            raise HTTPException(status_code=503, detail="Pipeline não inicializado. Aguarde inicialização completa.")

        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="Envie pelo menos uma imagem.")

        # Diretório permanente de entradas (para aprendizado)
        base_dir = Path(DIRETORIOS["resultado_inspecoes"]) / "entradas_planta"
        base_dir.mkdir(parents=True, exist_ok=True)

        caminhos_imagens = []
        ts = int(time.time())
        try:
            # Salvar todas as imagens de entrada de forma permanente
            for idx, file in enumerate(files):
                if not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail=f"Arquivo não é imagem: {file.filename}")

                destino = base_dir / f"lote_{ts}_{idx:03d}_{file.filename}"
                with open(destino, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                caminhos_imagens.append(str(destino))

            if len(caminhos_imagens) == 1:
                # Se só tem 1 imagem, cai no fluxo normal de inspeção
                logger.info("Apenas 1 imagem enviada, usando fluxo padrão de inspeção")
                resultado = pipeline_global.inspecionar_planta(caminhos_imagens[0], confianca_min)
            else:
                # Gerar ortomosaico para visualização
                caminho_orto = Path(DIRETORIOS["resultado_inspecoes"]) / f"ortomosaico_{int(time.time())}.jpg"
                logger.info(f"Gerando ortomosaico a partir de {len(caminhos_imagens)} imagens (para visualização)...")
                try:
                    caminho_orto_str = pipeline_global.gerar_ortomosaico(caminhos_imagens, str(caminho_orto))
                except Exception as e:
                    logger.warning(f"Falha ao gerar ortomosaico: {e}. Usando primeira imagem como base visual.")
                    caminho_orto_str = caminhos_imagens[0]

                # Detectar e classificar nas IMAGENS ORIGINAIS com TILING (representação fiel)
                logger.info(f"Inspecionando {len(caminhos_imagens)} imagens originais individualmente para obter métricas fiéis...")
                resultados_individuais = []
                todas_deteccoes = []
                for idx, img_path in enumerate(caminhos_imagens):
                    logger.info(f"  Processando imagem {idx+1}/{len(caminhos_imagens)}: {Path(img_path).name}")
                    # USAR TILING para imagens grandes (8192x6144)
                    res = processar_imagem_com_tiling(img_path, confianca_min)
                    if res.get('status') == 'sucesso':
                        resultados_individuais.append(res)
                        if res.get('total_modulos', 0) > 0:
                            logger.info(
                                f"  -> Detectados {res.get('total_modulos', 0)} módulos "
                                f"({res.get('num_tiles', 0)} tiles)"
                            )
                            todas_deteccoes.extend(res.get('deteccoes', []))
                        else:
                            logger.warning("  -> Nenhum módulo detectado nesta imagem")

                # Agregar resultados
                total_modulos = sum(r['total_modulos'] for r in resultados_individuais)
                total_limpos = sum(r['modulos_limpos'] for r in resultados_individuais)
                total_sujos = sum(r['modulos_sujos'] for r in resultados_individuais)

                # Agregar detecções (já preenchidas no loop acima)
                todas_deteccoes = todas_deteccoes

                # Garantir que todas as detecções tenham chaves de sujidade consistentes
                for det in todas_deteccoes:
                    if 'nivel_sujidade' not in det:
                        det['nivel_sujidade'] = det.get('classe', 'desconhecido')
                    if 'classe_binaria' not in det:
                        nivel = det.get('nivel_sujidade', '')
                        det['classe_binaria'] = 'sujo' if 'sujo' in str(nivel).lower() else 'limpo'

                # Calcular estatísticas agregadas
                if total_modulos > 0:
                    percentual_limpos = round((total_limpos / total_modulos) * 100, 1)
                    percentual_sujos = round((total_sujos / total_modulos) * 100, 1)

                    # Contagem por classe
                    contagem_classes = {
                        'limpo': sum(1 for d in todas_deteccoes if d.get('classe') == 'limpo'),
                        'pouco sujo': sum(1 for d in todas_deteccoes if d.get('classe') == 'pouco sujo'),
                        'sujo': sum(1 for d in todas_deteccoes if d.get('classe') == 'sujo'),
                        'muito sujo': sum(1 for d in todas_deteccoes if d.get('classe') == 'muito sujo'),
                    }

                    # Distribuição percentual
                    distribuicao_classes = {k: round((v / total_modulos) * 100, 1) for k, v in contagem_classes.items()}

                    # Classe predominante e índice de homogeneidade
                    classe_predominante = max(distribuicao_classes.items(), key=lambda x: x[1])[0]
                    indice_homogeneidade = round(max(distribuicao_classes.values()) / 100.0, 3)

                    confianca_media = round(sum(d['confianca_classificacao'] for d in todas_deteccoes) / len(todas_deteccoes), 3)
                else:
                    percentual_limpos = 0
                    percentual_sujos = 0
                    contagem_classes = {'limpo': 0, 'pouco sujo': 0, 'sujo': 0, 'muito sujo': 0}
                    distribuicao_classes = {'limpo': 0, 'pouco sujo': 0, 'sujo': 0, 'muito sujo': 0}
                    classe_predominante = None
                    indice_homogeneidade = 0
                    confianca_media = 0

                # Gerar imagem anotada e mapa de homogeneidade usando o ortomosaico
                # (visualização aproximada, números vêm das originais)
                import base64
                img_orto_cv = cv2.imread(caminho_orto_str)
                imagem_anotada_base64 = None
                mapa_homog_base64 = None
                if img_orto_cv is not None:
                    # Converter ortomosaico para base64 sem anotações (ou com anotações placeholder)
                    _, buffer = cv2.imencode('.jpg', img_orto_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    imagem_anotada_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                    mapa_homog_base64 = imagem_anotada_base64  # Placeholder: usar mesmo ortomosaico

                # Montar resultado final
                resultado = {
                    'status': 'sucesso',
                    'imagem_original': f'{len(caminhos_imagens)} imagens do drone',
                    'imagem_ortomosaico': Path(caminho_orto_str).name,
                    'total_modulos': total_modulos,
                    'modulos_limpos': total_limpos,
                    'modulos_sujos': total_sujos,
                    'percentual_limpos': percentual_limpos,
                    'percentual_sujos': percentual_sujos,
                    'confianca_media': confianca_media,
                    'contagem_classes': contagem_classes,
                    'distribuicao_classes': distribuicao_classes,
                    'classe_predominante': classe_predominante,
                    'indice_homogeneidade': indice_homogeneidade,
                    'deteccoes': todas_deteccoes,
                    'imagem_anotada': imagem_anotada_base64,
                    'mapa_homogeneidade': mapa_homog_base64,
                    'tempo_processamento': 0,  # placeholder
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'nota': 'Detecção realizada nas imagens originais para fidelidade. Ortomosaico usado como visualização.',
                }

            logger.info(f"Inspeção concluída: {resultado.get('total_modulos', 0)} módulos")
            return resultado

        finally:
            # Imagens de entrada são mantidas para aprendizado futuro
            pass

    except HTTPException:
        # Propagar erros HTTP explícitos
        raise
    except Exception as e:
        logger.error(f"Erro na inspeção com ortomosaico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inspecionar-planta-ortomosaico-v2")
async def inspecionar_planta_ortomosaico_v2(
    files: List[UploadFile] = File(...),
    confianca_min: float = Form(0.5)
):
    """
    Nova versão do pipeline de ortomosaico usando OpenStitching + SAHI.
    Gera um ortomosaico robusto com OpenStitching e executa detecção
    com tiling inteligente (SAHI) diretamente sobre o ortomosaico,
    seguida de classificação com EfficientNet.
    """
    try:
        if gerador_ortomosaico_global is None:
            raise HTTPException(status_code=503, detail="Gerador de ortomosaico (OpenStitching) não inicializado.")
        if detector_sahi_global is None:
            raise HTTPException(status_code=503, detail="Detector SAHI não inicializado.")
        if classificador_global is None:
            raise HTTPException(status_code=503, detail="Classificador não inicializado.")

        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="Envie pelo menos uma imagem.")

        base_dir = Path(DIRETORIOS["resultado_inspecoes"]) / "entradas_planta"
        base_dir.mkdir(parents=True, exist_ok=True)

        caminhos_imagens = []
        ts = int(time.time())
        for idx, file in enumerate(files):
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"Arquivo não é imagem: {file.filename}")
            destino = base_dir / f"v2_{ts}_{idx:03d}_{file.filename}"
            with open(destino, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            caminhos_imagens.append(str(destino))

        # Gerar ortomosaico com OpenStitching
        if len(caminhos_imagens) == 1:
            caminho_orto_str = caminhos_imagens[0]
            logger.info("Apenas 1 imagem enviada, usando diretamente sem stitching para o pipeline v2")
        else:
            caminho_orto = Path(DIRETORIOS["resultado_inspecoes"]) / f"ortomosaico_v2_{int(time.time())}.jpg"
            logger.info(f"[v2] Gerando ortomosaico (OpenStitching) a partir de {len(caminhos_imagens)} imagens...")
            res_orto = gerador_ortomosaico_global.gerar_ortomosaico(caminhos_imagens, str(caminho_orto))
            if not res_orto.get("sucesso"):
                logger.warning(f"[v2] Falha ao gerar ortomosaico: {res_orto.get('erro')}. Usando primeira imagem como fallback.")
                caminho_orto_str = caminhos_imagens[0]
            else:
                caminho_orto_str = res_orto.get("caminho_ortomosaico", str(caminho_orto))

        # Detecção com SAHI sobre o ortomosaico
        logger.info(f"[v2] Detectando módulos no ortomosaico com SAHI: {Path(caminho_orto_str).name}")
        res_sahi = detector_sahi_global.detectar(caminho_orto_str)
        deteccoes = res_sahi.get("deteccoes", [])
        logger.info(f"[v2] SAHI retornou {len(deteccoes)} detecções brutas")

        img_orto_cv = cv2.imread(caminho_orto_str)
        if img_orto_cv is None:
            raise HTTPException(status_code=500, detail="Falha ao carregar ortomosaico para classificação.")
        h, w = img_orto_cv.shape[:2]

        # Classificar cada módulo detectado
        modulos_classificados = []
        temp_dir = Path(DIRETORIOS["resultado_inspecoes"]) / "temp_crops_v2"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for det in deteccoes:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img_orto_cv[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            crop_path = temp_dir / f"crop_{int(time.time()*1000)}.jpg"
            try:
                cv2.imwrite(str(crop_path), crop)
                res_cls = classificador_global.classificar(str(crop_path))
            except Exception as e:
                logger.error(f"[v2] Erro ao classificar módulo: {e}")
                res_cls = {"classe": "erro", "confianca": 0.0, "nivel_sujidade": None, "probabilidades": {}}
            finally:
                if crop_path.exists():
                    crop_path.unlink()

            det_com_classe = {
                "bbox": [x1, y1, x2, y2],
                "confianca_deteccao": float(det.get("confianca", 0.0)),
                "classe_yolo": det.get("classe"),
                "classe_yolo_id": det.get("classe_id"),
                "classe": res_cls.get("classe"),
                "confianca_classificacao": float(res_cls.get("confianca", 0.0)),
                "nivel_sujidade": res_cls.get("nivel_sujidade"),
                "probabilidades": res_cls.get("probabilidades", {}),
            }

            nivel = det_com_classe.get("nivel_sujidade")
            if nivel is None:
                classe_binaria = "sujo" if str(det_com_classe.get("classe", "")).lower().startswith("sujo") else "limpo"
            else:
                classe_binaria = "sujo" if "sujo" in str(nivel).lower() and "limpo" not in str(nivel).lower() else "limpo"
            det_com_classe["classe_binaria"] = classe_binaria

            modulos_classificados.append(det_com_classe)

        total_modulos = len(modulos_classificados)
        if total_modulos == 0:
            return {
                "status": "sucesso",
                "imagem_original": f"{len(caminhos_imagens)} imagens do drone",
                "imagem_ortomosaico": Path(caminho_orto_str).name,
                "total_modulos": 0,
                "modulos_limpos": 0,
                "modulos_sujos": 0,
                "percentual_limpos": 0,
                "percentual_sujos": 0,
                "confianca_media": 0,
                "contagem_classes": {},
                "distribuicao_classes": {},
                "classe_predominante": None,
                "indice_homogeneidade": 0,
                "deteccoes": [],
                "imagem_anotada": None,
                "mapa_homogeneidade": None,
                "tempo_processamento": res_sahi.get("tempo_processamento", 0.0),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "nota": "Nenhum módulo detectado no pipeline v2 (OpenStitching + SAHI).",
            }

        limpos = sum(1 for m in modulos_classificados if m.get("classe_binaria") == "limpo")
        sujos = total_modulos - limpos

        contagem_classes = {"limpo": 0, "pouco sujo": 0, "sujo": 0, "muito sujo": 0}
        for m in modulos_classificados:
            nivel = m.get("nivel_sujidade")
            if nivel == "limpo":
                chave = "limpo"
            elif nivel == "pouco_sujo":
                chave = "pouco sujo"
            elif nivel == "sujo":
                chave = "sujo"
            elif nivel == "muito_sujo":
                chave = "muito sujo"
            else:
                chave = m.get("classe", "desconhecido")
            contagem_classes[chave] = contagem_classes.get(chave, 0) + 1

        distribuicao_classes = {k: round((v / total_modulos) * 100, 1) for k, v in contagem_classes.items() if total_modulos > 0}

        if contagem_classes:
            classe_predominante = max(contagem_classes.items(), key=lambda x: x[1])[0]
            indice_homogeneidade = round(max(contagem_classes.values()) / float(total_modulos), 3)
        else:
            classe_predominante = None
            indice_homogeneidade = 0.0

        confianca_media = round(
            sum(m.get("confianca_classificacao", 0.0) for m in modulos_classificados) / float(total_modulos),
            3,
        )

        # Gerar imagens de visualização (anotada e mapa de homogeneidade)
        anotada = img_orto_cv.copy()
        mapa = img_orto_cv.copy()
        for m in modulos_classificados:
            x1, y1, x2, y2 = m["bbox"]
            cor = (0, 255, 0) if m.get("classe_binaria") == "limpo" else (0, 0, 255)
            cv2.rectangle(anotada, (x1, y1), (x2, y2), cor, 2)
            cv2.rectangle(mapa, (x1, y1), (x2, y2), cor, -1)

        mapa = cv2.addWeighted(mapa, 0.3, img_orto_cv, 0.7, 0)

        _, buffer_anotada = cv2.imencode(".jpg", anotada, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, buffer_mapa = cv2.imencode(".jpg", mapa, [cv2.IMWRITE_JPEG_QUALITY, 90])
        imagem_anotada_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_anotada).decode('utf-8')}"
        mapa_homog_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_mapa).decode('utf-8')}"

        resultado = {
            "status": "sucesso",
            "imagem_original": f"{len(caminhos_imagens)} imagens do drone",
            "imagem_ortomosaico": Path(caminho_orto_str).name,
            "total_modulos": total_modulos,
            "modulos_limpos": limpos,
            "modulos_sujos": sujos,
            "percentual_limpos": round((limpos / total_modulos) * 100, 1),
            "percentual_sujos": round((sujos / total_modulos) * 100, 1),
            "confianca_media": confianca_media,
            "contagem_classes": contagem_classes,
            "distribuicao_classes": distribuicao_classes,
            "classe_predominante": classe_predominante,
            "indice_homogeneidade": indice_homogeneidade,
            "deteccoes": modulos_classificados,
            "imagem_anotada": imagem_anotada_base64,
            "mapa_homogeneidade": mapa_homog_base64,
            "tempo_processamento": res_sahi.get("tempo_processamento", 0.0),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nota": "Pipeline v2: OpenStitching (ortomosaico) + SAHI (tiling) + EfficientNet.",
        }

        logger.info(f"[v2] Inspeção concluída: {resultado.get('total_modulos', 0)} módulos")
        return resultado

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[v2] Erro na inspeção com ortomosaico v2: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historico-inspecoes")
async def obter_historico_inspecoes():
    """Retorna histórico de inspeções realizadas."""
    try:
        return {
            "total": len(historico_inspecoes),
            "inspecoes": historico_inspecoes[-20:]  # Últimas 20
        }
    except Exception as e:
        logger.error(f"❌ Erro ao obter histórico: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ ENDPOINTS DE MÉTRICAS ============

@app.get("/metricas-detector")
async def obter_metricas_detector():
    """Retorna métricas do treinamento do detector."""
    try:
        # Tentar carregar relatório se existir
        relatorio_path = Path(DIRETORIOS["modelos_salvos"]) / "detector_yolo" / "relatorio_treinamento.json"
        
        if relatorio_path.exists():
            with open(relatorio_path, 'r', encoding='utf-8') as f:
                relatorio = json.load(f)
            return relatorio
        else:
            # Retornar status atual se não há relatório
            return {
                "status": "sem_dados",
                "mensagem": "Nenhum treinamento concluído encontrado",
                "status_atual": status_treinamento_detector()
            }
            
    except Exception as e:
        logger.error(f"❌ Erro ao obter métricas do detector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metricas-classificador")
async def obter_metricas_classificador():
    """Retorna métricas do treinamento do classificador."""
    try:
        # Tentar carregar relatório se existir
        relatorio_path = Path(DIRETORIOS["modelos_salvos"]) / "classificador" / "relatorio_treinamento.json"
        
        if relatorio_path.exists():
            with open(relatorio_path, 'r', encoding='utf-8') as f:
                relatorio = json.load(f)
            return relatorio
        else:
            # Retornar status atual se não há relatório
            return {
                "status": "sem_dados",
                "mensagem": "Nenhum treinamento concluído encontrado",
                "status_atual": status_treinamento_classificador()
            }
            
    except Exception as e:
        logger.error(f"❌ Erro ao obter métricas do classificador: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/informacoes-sistema")
async def obter_informacoes_sistema():
    """Retorna informações detalhadas do sistema."""
    try:
        info = {
            "sistema": {
                "nome": "Sistema de Inspeção de Painéis Solares",
                "versao": "2.0.0",
                "desenvolvedor": "TCC - Engenharia Mecatrônica",
                "pipeline": "YOLOv8 + EfficientNet"
            },
            "modelos": {},
            "datasets": {},
            "status_api": {
                "rodando": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Informações dos modelos
        if detector_global:
            info["modelos"]["detector"] = detector_global.obter_informacoes_modelo()
        
        if classificador_global:
            info["modelos"]["classificador"] = classificador_global.obter_informacoes_modelo()
        
        # Informações dos datasets
        plantas_dir = Path(DIRETORIOS["plantas_completas"]) / "imagens" / "train"
        modulos_dir = Path(DIRETORIOS["modulos_individuais"])
        
        info["datasets"]["plantas_completas"] = {
            "total_imagens": len(list(plantas_dir.glob("*.jpg"))),
            "diretorio": str(plantas_dir)
        }
        
        info["datasets"]["modulos_individuais"] = {
            "limpos": len(list((modulos_dir / "limpo").glob("*.jpg"))),
            "sujos": len(list((modulos_dir / "sujo").glob("*.jpg"))),
            "total": len(list(modulos_dir.rglob("*.jpg"))),
            "diretorio": str(modulos_dir)
        }
        
        return info
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter informações do sistema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ ENDPOINTS DE TESTE ============

@app.post("/testar-deteccao")
async def testar_deteccao(file: UploadFile = File(...)):
    """Testa apenas a detecção (sem classificação)."""
    try:
        if detector_global is None:
            raise HTTPException(status_code=503, detail="Detector não inicializado")
        
        # Salvar temporariamente
        temp_path = Path(DIRETORIOS["resultado_inspecoes"]) / f"teste_det_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Executar detecção
        deteccoes = detector_global.detectar(str(temp_path))
        
        # Gerar imagem anotada
        imagem_anotada = detector_global.desenhar_deteccoes(str(temp_path), deteccoes)
        
        # Limpar
        temp_path.unlink()
        
        return {
            "status": "sucesso",
            "deteccoes": deteccoes,
            "num_modulos": len(deteccoes),
            "imagem_anotada": imagem_anotada
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de detecção: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/testar-classificacao")
async def testar_classificacao(file: UploadFile = File(...)):
    """Testa apenas a classificação (um módulo por vez)."""
    try:
        if classificador_global is None:
            raise HTTPException(status_code=503, detail="Classificador não inicializado")
        
        # Salvar temporariamente
        temp_path = Path(DIRETORIOS["resultado_inspecoes"]) / f"teste_cls_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Executar classificação
        resultado = classificador_global.classificar(str(temp_path))
        
        # Limpar
        temp_path.unlink()
        
        return {
            "status": "sucesso",
            "classificacao": resultado
        }
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de classificação: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("🌞 Iniciando API de Inspeção de Painéis Solares")
    print("📍 Documentação disponível em: http://localhost:8000/docs")
    print("🚀 Sistema pronto para uso!")
    
    uvicorn.run(
        "principal:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
