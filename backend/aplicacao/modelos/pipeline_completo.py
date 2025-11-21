"""
Pipeline Completo: Detecção + Classificação de Painéis Solares
Desenvolvido para TCC - Engenharia Mecatrônica
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import time
import json
from pathlib import Path
import logging
import base64

from aplicacao.modelos.drone_mosaicing import DroneMosaicing

logger = logging.getLogger(__name__)

class PipelineInspecao:
    """
    Pipeline completo que integra detecção (YOLOv8) e classificação (EfficientNet)
    para inspeção automatizada de plantas solares.
    
    Fluxo:
    1. Upload da imagem da planta completa
    2. YOLOv8 detecta todos os módulos fotovoltaicos
    3. Para cada módulo detectado:
       - Recorta a região do módulo
       - EfficientNet classifica em limpo/sujo
    4. Gera relatório visual com bounding boxes coloridos
    5. Calcula estatísticas da planta
    """
    
    def __init__(self, detector, classificador):
        """
        Inicializa o pipeline de inspeção.
        
        Args:
            detector: Instância de DetectorModulos (YOLOv8)
            classificador: Instancia de ClassificadorSujidade (EfficientNet)
        """
        self.detector = detector
        self.classificador = classificador
        # Mosaico de drone dedicado para ortomosaico
        self.mosaicer = DroneMosaicing(max_resolution=2000, match_ratio=0.75, ransac_threshold=5.0)
        
        logger.info("Pipeline de inspeção inicializado")
        logger.info(f"   Detector: {type(detector).__name__}")
        logger.info(f"   Classificador: {type(classificador).__name__}")
    
    def inspecionar_planta(self, caminho_imagem: str, confianca_min: float = 0.5):
        """
        Executa pipeline completo de inspeção.
        
        Args:
            caminho_imagem (str): Caminho da imagem da planta
            confianca_min (float): Confiança mínima para detecção
            
        Returns:
            dict: Resultado completo da inspeção
        """
        inicio = time.time()
        
        try:
            logger.info(f"Iniciando inspeção da planta: {Path(caminho_imagem).name}")

            # Carregar imagem uma única vez e usar suas dimensões para ajustar parâmetros
            imagem_original = cv2.imread(caminho_imagem)
            if imagem_original is None:
                raise ValueError(f"Imagem não encontrada ou inválida: {caminho_imagem}")

            h, w = imagem_original.shape[:2]
            maior_dim = max(h, w)

            # Ajuste automático para ortomosaicos grandes
            if maior_dim >= 2500:
                imgsz_det = 1280
                conf_det = min(confianca_min, 0.35)
                logger.info(
                    "Imagem grande detectada (%dx%d). Usando parâmetros de detecção para ortomosaico (imgsz=%d, conf_min=%.2f)",
                    w,
                    h,
                    imgsz_det,
                    conf_det,
                )
            else:
                imgsz_det = 640
                conf_det = confianca_min
                logger.info(
                    "Imagem em escala padrão (%dx%d). Usando parâmetros de detecção padrão (imgsz=%d, conf_min=%.2f)",
                    w,
                    h,
                    imgsz_det,
                    conf_det,
                )

            # Etapa 1: Detecção de módulos na planta
            logger.info("Etapa 1/4: detectando módulos com YOLOv8...")
            deteccoes = self.detector.detectar(caminho_imagem, conf_det, imgsz=imgsz_det)
            
            if not deteccoes:
                logger.warning("Nenhum módulo detectado na imagem")
                return self._criar_resultado_vazio(caminho_imagem, time.time() - inicio)
            
            logger.info(f"   {len(deteccoes)} módulos detectados")
            
            # Etapa 2: Classificar cada módulo detectado
            logger.info("Etapa 2/4: classificando sujidade com EfficientNet...")
            resultados = []
            
            for i, det in enumerate(deteccoes):
                # Recortar módulo
                x1, y1, x2, y2 = map(int, det['bbox'])
                modulo_recortado = imagem_original[y1:y2, x1:x2]
                
                # Converter para PIL
                modulo_pil = Image.fromarray(cv2.cvtColor(modulo_recortado, cv2.COLOR_BGR2RGB))
                
                # Classificar sujidade
                classificacao = self.classificador.classificar(modulo_pil)
                # Mapear saída do classificador binário (limpo/sujo) para 4 níveis de sujidade
                nivel_sujidade = classificacao.get('nivel_sujidade')
                classe_binaria = classificacao.get('classe')

                if nivel_sujidade:
                    if nivel_sujidade == 'pouco_sujo':
                        classe_nivel = 'pouco sujo'
                    elif nivel_sujidade == 'muito_sujo':
                        classe_nivel = 'muito sujo'
                    else:
                        classe_nivel = nivel_sujidade
                else:
                    classe_nivel = classificacao.get('classe', 'desconhecido')

                resultado = {
                    'indice': i,
                    'bbox': det['bbox'],
                    'confianca_deteccao': det['confianca'],
                    'classe': classe_nivel,
                    'classe_binaria': classe_binaria,
                    'nivel_sujidade': nivel_sujidade,
                    'confianca_classificacao': classificacao['confianca'],
                    'probabilidades': classificacao['probabilidades']
                }
                
                resultados.append(resultado)
                
                # Progresso
                if (i + 1) % 5 == 0:
                    print(f"\r   Classificando: {i+1}/{len(deteccoes)} módulos", end="", flush=True)
            
            print()  # Nova linha
            logger.info(f"   {len(resultados)} módulos classificados")
            
            # Etapa 3: Gerar imagem anotada
            logger.info("Etapa 3/4: gerando relatório visual...")
            imagem_anotada = self._desenhar_anotacoes(imagem_original, resultados)
            mapa_homogeneidade = self._gerar_mapa_homogeneidade(imagem_original, resultados)
            
            # Etapa 4: Calcular estatísticas
            logger.info("Etapa 4/4: calculando estatísticas...")
            estatisticas = self._calcular_estatisticas(resultados)
            
            # 5️⃣ Montar resultado final
            tempo_total = time.time() - inicio
            
            resultado = {
                'status': 'sucesso',
                'imagem_original': Path(caminho_imagem).name,
                'total_modulos': len(resultados),
                'modulos_limpos': estatisticas['limpos'],
                'modulos_sujos': estatisticas['sujos'],
                'percentual_limpos': estatisticas['percentual_limpos'],
                'percentual_sujos': estatisticas['percentual_sujos'],
                'confianca_media': estatisticas['confianca_media'],
                'contagem_classes': estatisticas['contagem_classes'],
                'distribuicao_classes': estatisticas['distribuicao_classes'],
                'classe_predominante': estatisticas['classe_predominante'],
                'indice_homogeneidade': estatisticas['indice_homogeneidade'],
                'deteccoes': resultados,
                'imagem_anotada': imagem_anotada,
                'mapa_homogeneidade': mapa_homogeneidade,
                'tempo_processamento': round(tempo_total, 2),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metricas_detecao': self._calcular_metricas_detecao(deteccoes),
                'metricas_classificacao': self._calcular_metricas_classificacao(resultados),
            }
            
            logger.info(f"Inspeção concluída em {tempo_total:.2f}s")
            logger.info(f"   {estatisticas['limpos']} módulos limpos ({estatisticas['percentual_limpos']:.1f}%)")
            logger.info(f"   {estatisticas['sujos']} módulos sujos ({estatisticas['percentual_sujos']:.1f}%)")
            
            return resultado
            
        except Exception as e:
            logger.error(f"Erro na inspeção: {e}")
            return {
                'status': 'erro',
                'erro': str(e),
                'tempo_processamento': round(time.time() - inicio, 2)
            }
    
    def inspecionar_lote(self, caminhos_imagens: List[str], confianca_min: float = 0.5):
        """
        Processa múltiplas imagens em lote.
        
        Args:
            caminhos_imagens (list): Lista de caminhos das imagens
            confianca_min (float): Confiança mínima para detecção
            
        Returns:
            list: Lista de resultados de inspeção
        """
        logger.info(f"Iniciando inspeção em lote: {len(caminhos_imagens)} imagens")
        
        resultados = []
        for i, caminho in enumerate(caminhos_imagens):
            logger.info(f"Processando imagem {i+1}/{len(caminhos_imagens)}: {Path(caminho).name}")
            
            resultado = self.inspecionar_planta(caminho, confianca_min)
            resultado['indice_lote'] = i
            resultados.append(resultado)
        
        # Estatísticas do lote
        total_modulos = sum(r.get('total_modulos', 0) for r in resultados)
        total_limpos = sum(r.get('modulos_limpos', 0) for r in resultados)
        total_sujos = sum(r.get('modulos_sujos', 0) for r in resultados)
        
        logger.info("Lote processado:")
        logger.info(f"   Total de módulos: {total_modulos}")
        logger.info(f"   Total limpos: {total_limpos} ({total_limpos/total_modulos*100:.1f}%)" if total_modulos > 0 else "   Total limpos: 0")
        logger.info(f"   Total sujos: {total_sujos} ({total_sujos/total_modulos*100:.1f}%)" if total_modulos > 0 else "   Total sujos: 0")
        
        return resultados

    def gerar_ortomosaico(self, caminhos_imagens: List[str], caminho_saida: str) -> str:
        """Gera um ortomosaico utilizando o módulo DroneMosaicing.

        Esta implementação delega a lógica de mosaico para a classe DroneMosaicing,
        que usa SIFT + FLANN + homografia (RANSAC) para compor as imagens.

        Args:
            caminhos_imagens: Lista de caminhos das imagens de entrada.
            caminho_saida: Caminho de arquivo onde o ortomosaico será salvo (JPEG).

        Returns:
            str: Caminho do ortomosaico salvo.
        """
        try:
            if not caminhos_imagens:
                raise ValueError("Nenhuma imagem fornecida para gerar ortomosaico")

            logger.info("Gerando ortomosaico com DroneMosaicing")
            caminho_final = self.mosaicer.gerar_ortomosaico(caminhos_imagens, caminho_saida)
            logger.info("Ortomosaico gerado com sucesso em %s", caminho_final)
            return caminho_final

        except Exception as e:
            logger.error(f"Erro ao gerar ortomosaico com DroneMosaicing: {e}")
            raise
    
    def _desenhar_anotacoes(self, imagem: np.ndarray, resultados: List[Dict]) -> str:
        """
        Desenha bounding boxes coloridos com labels na imagem.
        
        Args:
            imagem: Imagem OpenCV original
            resultados: Lista de resultados da classificação
            
        Returns:
            str: Imagem anotada em base64
        """
        try:
            imagem_copia = imagem.copy()
            
            for resultado in resultados:
                x1, y1, x2, y2 = map(int, resultado['bbox'])
                classe = resultado['classe']
                confianca = resultado['confianca_classificacao']
                conf_detecao = resultado['confianca_deteccao']
                
                # Cores por classe (BGR): limpo=verde, pouco sujo=amarelo, sujo=laranja, muito sujo=vermelho
                if classe == 'limpo':
                    cor = (0, 200, 0)            # Verde
                elif classe == 'pouco sujo':
                    cor = (0, 255, 255)          # Amarelo
                elif classe == 'sujo':
                    cor = (0, 165, 255)          # Laranja
                elif classe == 'muito sujo':
                    cor = (0, 0, 255)            # Vermelho
                else:
                    cor = (255, 255, 255)        # Branco (desconhecido)
                
                # Desenhar bounding box
                cv2.rectangle(imagem_copia, (x1, y1), (x2, y2), cor, 3)
                
                # Label com informações
                label_classe = classe.upper()
                # 'confianca' já está em porcentagem (0-100)
                label_conf = f"{confianca:.0f}%"
                label = f"{label_classe} {label_conf}"
                
                # Informações adicionais em label menor
                label_info = f"Det: {conf_detecao*100:.0f}%"
                
                # Tamanhos de texto
                (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                (w2, h2), _ = cv2.getTextSize(label_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Fundo para label principal
                cv2.rectangle(imagem_copia, (x1, y1-h1-10), (x1+w1, y1), cor, -1)
                cv2.putText(imagem_copia, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Fundo para label de informação
                cv2.rectangle(imagem_copia, (x1, y1+h1-10), (x1+w2, y1+h1), cor, -1)
                cv2.putText(imagem_copia, label_info, (x1, y1+h1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Converter para base64
            _, buffer = cv2.imencode('.jpg', imagem_copia, [cv2.IMWRITE_JPEG_QUALITY, 90])
            imagem_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return f"data:image/jpeg;base64,{imagem_base64}"
            
        except Exception as e:
            logger.error(f"Erro ao desenhar anotações: {e}")
            return None
    
    def _calcular_estatisticas(self, resultados: List[Dict]) -> Dict:
        """
        Calcula estatísticas da inspeção.
        
        Args:
            resultados: Lista de resultados da classificação
            
        Returns:
            dict: Estatísticas calculadas
        """
        if not resultados:
            return {
                'limpos': 0,
                'sujos': 0,
                'percentual_limpos': 0,
                'percentual_sujos': 0,
                'distribuicao_classes': {c: 0 for c in ['limpo', 'pouco sujo', 'sujo', 'muito sujo']},
                'confianca_media': 0,
                'confianca_detecao_media': 0
            }
        
        # Contar classes (4 níveis)
        total = len(resultados)
        contagem = {
            'limpo': sum(1 for r in resultados if r['classe'] == 'limpo'),
            'pouco sujo': sum(1 for r in resultados if r['classe'] == 'pouco sujo'),
            'sujo': sum(1 for r in resultados if r['classe'] == 'sujo'),
            'muito sujo': sum(1 for r in resultados if r['classe'] == 'muito sujo'),
        }
        limpos = contagem['limpo']
        sujos = contagem['pouco sujo'] + contagem['sujo'] + contagem['muito sujo']
        # Percentuais gerais
        percentual_limpos = round((limpos / total) * 100, 1) if total > 0 else 0
        percentual_sujos = round((sujos / total) * 100, 1) if total > 0 else 0
        # Distribuição por classe
        distribuicao = {k: (round(v / total * 100, 1) if total > 0 else 0) for k, v in contagem.items()}

        # Índice de homogeneidade: fração da classe predominante (0-1)
        if total > 0:
            classe_predominante = max(distribuicao.items(), key=lambda kv: kv[1])[0]
            indice_homogeneidade = round(max(distribuicao.values()) / 100.0, 3)
        else:
            classe_predominante = None
            indice_homogeneidade = 0
        
        # Calcular confianças médias
        conf_classificacao = np.mean([r['confianca_classificacao'] for r in resultados])
        conf_deteccao = np.mean([r['confianca_deteccao'] for r in resultados])
        
        return {
            'limpos': limpos,
            'sujos': sujos,
            'percentual_limpos': percentual_limpos,
            'percentual_sujos': percentual_sujos,
            'distribuicao_classes': distribuicao,
            'contagem_classes': contagem,
            'confianca_media': round(float(conf_classificacao), 3),
            'confianca_detecao_media': round(float(conf_deteccao), 3),
            'classe_predominante': classe_predominante,
            'indice_homogeneidade': indice_homogeneidade
        }
    
    def _calcular_metricas_detecao(self, deteccoes: List[Dict]) -> Dict:
        """
        Calcula métricas específicas da detecção.
        
        Args:
            deteccoes: Lista de detecções do YOLO
            
        Returns:
            dict: Métricas de detecção
        """
        if not deteccoes:
            return {'confianca_min': 0, 'confianca_max': 0, 'confianca_media': 0, 'confianca_std': 0}
        
        confiancas = [d['confianca'] for d in deteccoes]
        
        return {
            'confianca_min': round(float(np.min(confiancas)), 3),
            'confianca_max': round(float(np.max(confiancas)), 3),
            'confianca_media': round(float(np.mean(confiancas)), 3),
            'confianca_std': round(float(np.std(confiancas)), 3)
        }
    
    def _calcular_metricas_classificacao(self, resultados: List[Dict]) -> Dict:
        """
        Calcula métricas específicas da classificação.
        
        Args:
            resultados: Lista de resultados completos
            
        Returns:
            dict: Métricas de classificação
        """
        if not resultados:
            return {
                'confianca_limpos_media': 0,
                'confianca_sujos_media': 0,
                'distribuicao_probabilidades': {}
            }
        
        # Separar por classe (4 níveis)
        limpos = [r for r in resultados if r['classe'] == 'limpo']
        sujos_all = [r for r in resultados if r['classe'] in ['pouco sujo', 'sujo', 'muito sujo']]
        # Calcular confianças médias agregadas
        conf_limpos = np.mean([r['confianca_classificacao'] for r in limpos]) if limpos else 0
        conf_sujos = np.mean([r['confianca_classificacao'] for r in sujos_all]) if sujos_all else 0
        # Distribuição de probabilidades médias por classe
        classes_padrao = ['limpo', 'pouco sujo', 'sujo', 'muito sujo']
        dist_probs = {}
        for classe in classes_padrao:
            probs = [r['probabilidades'].get(classe, 0) for r in resultados]
            dist_probs[classe] = round(float(np.mean(probs) if len(probs) > 0 else 0), 3)
        
        return {
            'confianca_limpos_media': round(float(conf_limpos), 3),
            'confianca_sujos_media': round(float(conf_sujos), 3),
            'distribuicao_probabilidades': dist_probs
        }

    def _gerar_mapa_homogeneidade(self, imagem: np.ndarray, resultados: List[Dict]) -> str:
        """
        Gera um mapa de homogeneidade (heatmap) baseado na severidade da sujidade.
        Limpo=0.0, Pouco sujo=0.33, Sujo=0.66, Muito sujo=1.0
        
        Returns:
            str: Imagem heatmap em base64 (overlay na imagem original)
        """
        try:
            import cv2
            import base64
            h, w = imagem.shape[:2]
            mapa = np.zeros((h, w), dtype=np.float32)
            peso = np.zeros((h, w), dtype=np.float32)
            # Mapear classe -> severidade
            severidade = {
                'limpo': 0.0,
                'pouco sujo': 0.33,
                'sujo': 0.66,
                'muito sujo': 1.0
            }
            for r in resultados:
                x1, y1, x2, y2 = map(int, r['bbox'])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                val = severidade.get(r['classe'], 0.0)
                mapa[y1:y2, x1:x2] += val
                peso[y1:y2, x1:x2] += 1.0
            # Média nas regiões cobertas
            mask = peso > 0
            mapa[mask] = mapa[mask] / peso[mask]
            # Normalizar 0-255 e aplicar colormap
            mapa_vis = (mapa * 255).astype(np.uint8)
            heat = cv2.applyColorMap(mapa_vis, cv2.COLORMAP_JET)
            # Overlay
            overlay = imagem.copy()
            alpha = 0.4
            overlay = cv2.addWeighted(heat, alpha, overlay, 1 - alpha, 0)
            _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            logger.error(f"Erro ao gerar mapa de homogeneidade: {e}")
            return None
    
    def _criar_resultado_vazio(self, caminho_imagem: str, tempo_processamento: float) -> Dict:
        """
        Cria resultado para caso sem detecções.
        
        Args:
            caminho_imagem (str): Caminho da imagem
            tempo_processamento (float): Tempo de processamento
            
        Returns:
            dict: Resultado vazio
        """
        return {
            'status': 'sem_deteccao',
            'imagem_original': Path(caminho_imagem).name,
            'total_modulos': 0,
            'modulos_limpos': 0,
            'modulos_sujos': 0,
            'percentual_limpos': 0,
            'percentual_sujos': 0,
            'confianca_media': 0,
            'deteccoes': [],
            'imagem_anotada': None,
            'tempo_processamento': round(tempo_processamento, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mensagem': 'Nenhum módulo fotovoltaico detectado na imagem'
        }
    
    def gerar_relatorio_json(self, resultado_inspecao: Dict, caminho_salvar: str = None):
        """
        Gera relatório em formato JSON.
        
        Args:
            resultado_inspecao (dict): Resultado da inspeção
            caminho_salvar (str): Caminho para salvar relatório
            
        Returns:
            str: JSON do relatório
        """
        try:
            # Preparar relatório
            relatorio = {
                'relatorio_inspecao_planta_solar': {
                    'metadados': {
                        'sistema': 'Pipeline Inspeção Painéis Solares',
                        'versao': '1.0.0',
                        'desenvolvedor': 'TCC - Engenharia Mecatrônica',
                        'timestamp': resultado_inspecao.get('timestamp'),
                        'tempo_processamento_seg': resultado_inspecao.get('tempo_processamento')
                    },
                    'resultados': resultado_inspecao
                }
            }
            
            # Converter para JSON
            relatorio_json = json.dumps(relatorio, indent=2, ensure_ascii=False)
            
            # Salvar se caminho fornecido
            if caminho_salvar:
                Path(caminho_salvar).parent.mkdir(parents=True, exist_ok=True)
                with open(caminho_salvar, 'w', encoding='utf-8') as f:
                    f.write(relatorio_json)
                logger.info(f"📄 Relatório JSON salvo: {caminho_salvar}")
            
            return relatorio_json
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório JSON: {e}")
            return None
    
    def validar_resultado(self, resultado: Dict) -> Dict:
        """
        Valida e retorna informações sobre o resultado.
        
        Args:
            resultado (dict): Resultado da inspeção
            
        Returns:
            dict: Informações de validação
        """
        try:
            validacoes = {
                'status': resultado.get('status', 'desconhecido'),
                'tem_modulos': resultado.get('total_modulos', 0) > 0,
                'tem_imagem_anotada': resultado.get('imagem_anotada') is not None,
                'tempo_aceitavel': resultado.get('tempo_processamento', 999) < 10.0,
                'confianca_aceitavel': resultado.get('confianca_media', 0) > 0.7,
            }
            
            # Calcular score de qualidade (0-100)
            score = 0
            if validacoes['status'] == 'sucesso':
                score += 30
            if validacoes['tem_modulos']:
                score += 20
            if validacoes['tem_imagem_anotada']:
                score += 20
            if validacoes['tempo_aceitavel']:
                score += 15
            if validacoes['confianca_aceitavel']:
                score += 15
            
            validacoes['score_qualidade'] = score
            
            return validacoes
            
        except Exception as e:
            return {
                'status': 'erro_validacao',
                'erro': str(e),
                'score_qualidade': 0
            }

# Função fábrica
def criar_pipeline(detector, classificador):
    """
    Função fábrica para criar pipeline de inspeção.
    
    Args:
        detector: Instância do detector YOLO
        classificador: Instância do classificador EfficientNet
        
    Returns:
        PipelineInspecao: Instância do pipeline
    """
    return PipelineInspecao(detector, classificador)

# Teste rápido
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Teste do Pipeline de Inspeção")
    print("Este é um teste de inicialização do pipeline completo.")
    
    # Nota: Este teste requer detector e classificador já inicializados
    print("✅ Pipeline importado com sucesso!")
    print("Para usar: pipeline = PipelineInspecao(detector, classificador)")
