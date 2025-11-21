"""
Script para baixar datasets públicos automaticamente.
Desenvolvido para TCC - Engenharia Mecatrônica
"""
import os
import requests
import zipfile
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def baixar_zenodo_pv01():
    """Baixa dataset Zenodo PV01 (drone images) - RECOMENDADO"""
    logger.info("📥 Baixando Zenodo PV01 Dataset...")
    
    url = "https://zenodo.org/record/5171712/files/PV01.zip"
    destino = Path("dados/datasets_publicos/zenodo_pv01.zip")
    destino.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Fazendo download de {url}...")
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        
        if total > 0:
            logger.info(f"Tamanho do arquivo: {total / (1024*1024):.1f} MB")
        
        with open(destino, 'wb') as f:
            downloaded = 0
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
                downloaded += len(data)
                if total > 0:
                    percent = (downloaded / total) * 100
                    print(f"\rProgresso: {percent:.1f}%", end="", flush=True)
        
        print()  # Nova linha após progress bar
        logger.info(f"✅ Download completo: {destino}")
        
        # Extrair
        logger.info("📦 Extraindo arquivos...")
        with zipfile.ZipFile(destino, 'r') as zip_ref:
            zip_ref.extractall(destino.parent / "pv01")
        
        logger.info("✅ Extraído com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao baixar Zenodo PV01: {e}")
        return False

def baixar_kaggle_dust_detection():
    """Baixa dataset Kaggle de detecção de poeira"""
    logger.info("📥 Baixando Kaggle Dust Detection...")
    
    try:
        # Verificar se kaggle CLI está configurado
        import subprocess
        
        # Testar autenticação Kaggle
        result = subprocess.run(['kaggle', 'datasets', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("⚠️ Kaggle CLI não configurado. Configure com: kaggle config")
            logger.info("📝 Passos para configurar:")
            logger.info("1. Ir em kaggle.com/settings")
            logger.info("2. Criar API token (baixa kaggle.json)")
            logger.info("3. Mover para ~/.kaggle/kaggle.json")
            logger.info("4. Definir permissões: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        # Criar diretório de destino
        destino_dir = Path("dados/datasets_publicos")
        destino_dir.mkdir(parents=True, exist_ok=True)
        
        # Download via Kaggle CLI
        logger.info("Baixando via Kaggle CLI...")
        os.system("kaggle datasets download -d hemanthsai7/solar-panel-dust-detection -p dados/datasets_publicos/")
        
        # Extrair
        zip_path = Path("dados/datasets_publicos/solar-panel-dust-detection.zip")
        if zip_path.exists():
            logger.info("📦 Extraindo arquivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("dados/datasets_publicos/kaggle_dust")
            
            # Remover zip após extração
            zip_path.unlink()
            logger.info("✅ Kaggle dataset baixado e extraído!")
            return True
        else:
            logger.error("❌ Arquivo zip não encontrado após download")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao baixar Kaggle dataset: {e}")
        return False

def baixar_roboflow_solar_panels(api_key: str = None):
    """Baixa dataset Roboflow via API (opcional)"""
    if not api_key:
        logger.info("🔑 Pulei download Roboflow (API key não fornecida)")
        return False
    
    logger.info("📥 Baixando Roboflow Solar Panels...")
    
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("roboflow-100").project("solar-panels-taxvb")
        dataset = project.version(2).download("yolov8", 
                                              location="dados/datasets_publicos/roboflow_solar")
        
        logger.info("✅ Roboflow dataset baixado!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao baixar Roboflow dataset: {e}")
        return False

def converter_zenodo_para_yolo():
    """Converte máscaras de segmentação do Zenodo para formato YOLO"""
    logger.info("🔄 Convertendo Zenodo para formato YOLO...")
    
    try:
        import cv2
        import numpy as np
        from pathlib import Path
        
        src_dir = Path("dados/datasets_publicos/pv01")
        if not src_dir.exists():
            logger.error("❌ Diretório PV01 não encontrado. Execute download primeiro.")
            return False
        
        dst_dir = Path("dados/plantas_completas/imagens")
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        (dst_dir / "train").mkdir(exist_ok=True)
        (dst_dir / "val").mkdir(exist_ok=True)
        
        annotations_dir = Path("dados/plantas_completas/anotacoes")
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Processando imagens...")
        processed = 0
        
        for img_path in list(src_dir.glob("*.bmp"))[:50]:  # Limitar para teste
            label_path = img_path.parent / f"{img_path.stem}_label.bmp"
            
            if not label_path.exists():
                continue
            
            # Ler imagem e máscara
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                continue
            
            # Encontrar contornos (cada painel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = img.shape[:2]
            yolo_annotations = []
            
            for contour in contours:
                # Bounding box do contorno
                x, y, bw, bh = cv2.boundingRect(contour)
                
                # Ignorar boxes muito pequenos
                if bw < 20 or bh < 20:
                    continue
                
                # Converter para formato YOLO (normalizado)
                x_center = (x + bw/2) / w
                y_center = (y + bh/2) / h
                width = bw / w
                height = bh / h
                
                # Classe 0 = módulo fotovoltaico
                yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            if yolo_annotations:  # Apenas se encontrou módulos
                # Salvar imagem
                dst_path = dst_dir / "train" / img_path.name
                cv2.imwrite(str(dst_path), img)
                
                # Salvar anotações YOLO
                annotation_path = annotations_dir / f"{img_path.stem}.txt"
                with open(annotation_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                processed += 1
                if processed % 10 == 0:
                    print(f"\rProcessadas: {processed} imagens", end="", flush=True)
        
        print()  # Nova linha
        logger.info(f"✅ Conversão completa! {processed} imagens processadas.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na conversão: {e}")
        return False

def criar_dataset_yaml():
    """Cria arquivo dataset.yaml para YOLOv8"""
    logger.info("📝 Criando dataset.yaml...")
    
    try:
        import os
        
        # Obter path absoluto
        current_dir = os.path.abspath(".")
        data_dir = os.path.join(current_dir, "dados", "plantas_completas")
        
        yaml_content = f"""# Dataset de painéis solares para YOLOv8
# TCC - Engenharia Mecatrônica
path: {data_dir}
train: imagens/train
val: imagens/val

nc: 1  # número de classes
names: ['modulo']  # painel fotovoltaico
"""
        
        with open("dados/plantas_completas/dataset.yaml", 'w') as f:
            f.write(yaml_content)
        
        logger.info("✅ dataset.yaml criado!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar dataset.yaml: {e}")
        return False

def organizar_kaggle_para_classificacao():
    """Organiza dataset Kaggle para classificação EfficientNet"""
    logger.info("🗂️ Organizando dataset Kaggle para classificação...")
    
    try:
        src_dir = Path("dados/datasets_publicos/kaggle_dust")
        dst_limpo = Path("dados/modulos_individuais/limpo")
        dst_sujo = Path("dados/modulos_individuais/sujo")
        
        if not src_dir.exists():
            logger.warning("⚠️ Dataset Kaggle não encontrado")
            return False
        
        # Criar diretórios de destino
        dst_limpo.mkdir(parents=True, exist_ok=True)
        dst_sujo.mkdir(parents=True, exist_ok=True)
        
        # Mover arquivos (assumindo estrutura padrão)
        logger.info("Organizando arquivos...")
        
        # Procurar pastas clean/dusty ou similar
        for folder in src_dir.iterdir():
            if folder.is_dir():
                if 'clean' in folder.name.lower():
                    for img in folder.glob("*.jpg"):
                        dst_path = dst_limpo / img.name
                        shutil.copy2(img, dst_path)
                
                elif 'dust' in folder.name.lower() or 'dirty' in folder.name.lower():
                    for img in folder.glob("*.jpg"):
                        dst_path = dst_sujo / img.name
                        shutil.copy2(img, dst_path)
        
        # Contar arquivos
        limpo_count = len(list(dst_limpo.glob("*.jpg")))
        sujo_count = len(list(dst_sujo.glob("*.jpg")))
        
        logger.info(f"✅ Organização completa!")
        logger.info(f"   Limpos: {limpo_count} imagens")
        logger.info(f"   Sujos: {sujo_count} imagens")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao organizar dataset: {e}")
        return False

def main():
    """Função principal de download e preparação"""
    print("=" * 80)
    print("🚀 DOWNLOAD AUTOMÁTICO DE DATASETS PÚBLICOS")
    print("   TCC - Engenharia Mecatrônica")
    print("=" * 80)
    
    success_count = 0
    total_tasks = 4
    
    # 1. Baixar Zenodo PV01 (detecção)
    if baixar_zenodo_pv01():
        success_count += 1
    
    # 2. Converter Zenodo para YOLO
    if converter_zenodo_para_yolo():
        success_count += 1
    
    # 3. Criar dataset.yaml
    if criar_dataset_yaml():
        success_count += 1
    
    # 4. Baixar Kaggle (classificação)
    if baixar_kaggle_dust_detection():
        success_count += 1
        organizar_kaggle_para_classificacao()
    
    # 5. Roboflow (opcional)
    # api_key = os.getenv("ROBOFLOW_API_KEY")
    # baixar_roboflow_solar_panels(api_key)
    
    print("\n" + "=" * 80)
    print(f"📊 RESUMO: {success_count}/{total_tasks} tarefas concluídas")
    
    if success_count == total_tasks:
        print("✅ TODOS OS DATASETS BAIXADOS E PREPARADOS COM SUCESSO!")
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("1. cd backend && pip install -r requirements.txt")
        print("2. python -m aplicacao.principal")
        print("3. cd ../frontend && npm install && npm run dev")
        print("4. Acesse http://localhost:5173")
    else:
        print("⚠️ Algumas tarefas falharam. Verifique os logs acima.")
    
    print("=" * 80)

if __name__ == "__main__":
    import shutil  # Import aqui para evitar conflito
    main()
