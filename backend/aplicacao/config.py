"""
Configurações do Sistema de Inspeção de Painéis Solares
Desenvolvido para TCC - Engenharia Mecatrônica
"""

# Diretórios do sistema
DIRETORIOS = {
    "raiz": "dados",
    "plantas_completas": "dados/plantas_completas",
    "modulos_individuais": "dados/modulos_individuais",
    "datasets_publicos": "dados/datasets_publicos",
    "resultado_inspecoes": "dados/resultado_inspecoes",
    "modelos_salvos": r"F:\\modelos_salvos",
    "logs": "logs",
    "temp": "temp"
}

# Limites e configurações
LIMITES = {
    "tamanho_maximo_arquivo": 20 * 1024 * 1024 * 1024,  # 20GB por arquivo
    "tamanho_total_maximo_gb": 20,  # 20GB total recomendado
    "maximo_arquivos_upload": 5000,
    "confianca_minima_padrao": 0.5,
    "timeout_inspecao": 300,  # 5 minutos
}

# Configurações de CORS
CORS_CONFIG = {
    "allow_origins": [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Configurações de treinamento
TREINAMENTO = {
    "detector": {
        "epocas_padrao": 30,  # Reduzido
        "batch_size_padrao": 8,  # Reduzido para 8GB RAM
        "learning_rate_padrao": 0.01,
        "imgsz_padrao": 512,  # Reduzido de 640
        "patience_padrao": 8,  # Reduzido
        "modelo_base_padrao": "yolov8n.pt"
    },
    "classificador": {
        "epocas_padrao": 20,  # Reduzido
        "batch_size_padrao": 8,  # Reduzido para 8GB RAM
        "learning_rate_padrao": 0.001,
        "weight_decay_padrao": 1e-4,
        "patience_padrao": 8,  # Reduzido
        "img_size_padrao": 180  # Reduzido de 224
    }
}

# Configurações de modelo
MODELO = {
    "detector": {
        "classes": ["modulo"],
        "num_classes": 1,
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45
    },
    "classificador": {
        "classes": ["limpo", "pouco sujo", "sujo", "muito sujo"],
        "num_classes": 4,
        "confidence_threshold": 0.7
    }
}

# Formatos de arquivo suportados
FORMATOS_ARQUIVO = {
    "imagens": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "anotacoes": [".txt", ".json"],
    "modelos": [".pt", ".pth", ".onnx"]
}

# Configurações de logging
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/api.log"
}

# Configurações de hardware
HARDWARE = {
    "device_prioridade": ["mps", "cuda", "cpu"],  # Apple Silicon > NVIDIA > CPU
    "num_workers": 0,  # MPS não suporta múltiplos workers no macOS
    "pin_memory": False
}

# Configurações de API
API = {
    "title": "API de Inspeção de Painéis Solares",
    "description": "Sistema completo para inspeção automatizada de usinas solares",
    "version": "2.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "host": "0.0.0.0",
    "port": 8000
}

# Configurações de frontend (para referência)
FRONTEND = {
    "url_default": "http://localhost:5173",
    "porta_dev": 5173,
    "build_dir": "dist"
}
