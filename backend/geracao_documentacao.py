"""
Geração de Documentação LaTeX para TCC
Gera todas as tabelas no padrão Igor Dalavechia (2024)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class GeradorDocumentacaoTCC:
    """Gera documentação LaTeX para TCC"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classes = ['Limpo', 'Sujo']
    
    def gerar_todas_tabelas(self, resultados: Dict):
        """Gera todas as tabelas LaTeX"""
        
        print("\nGerando tabelas LaTeX...")
        
        tabelas = []
        
        # Tabela 1: Classes
        tabelas.append(self._gerar_tabela_classes())
        
        # Tabela 2: Dataset
        tabelas.append(self._gerar_tabela_dataset(resultados))
        
        # Tabela 3: Configuração
        tabelas.append(self._gerar_tabela_configuracao(resultados))
        
        # Tabela 4: Resultados
        tabelas.append(self._gerar_tabela_resultados(resultados))
        
        # Tabela 5: Desempenho por classe
        tabelas.append(self._gerar_tabela_desempenho_classes(resultados))
        
        # Salvar todas as tabelas
        conteudo_completo = "% Tabelas geradas automaticamente pelo gerador de documentação\n\n"
        conteudo_completo += "\n\n".join(tabelas)
        
        caminho = self.output_dir / 'tabelas_latex.tex'
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(conteudo_completo)
        
        print(f"   Tabelas LaTeX salvas em: {caminho}")
    
    def _gerar_tabela_classes(self) -> str:
        """Tabela 1: Classes identificadas"""
        
        return r"""\begin{table}[htb]
\caption{Classes a serem identificadas pelo sistema}
\label{tab:classes}
\centering
\begin{tabular}{|c|l|}
\hline
\textbf{Classe} & \textbf{Descrição} \\
\hline
Limpo & Painel sem sujidade visível, superfície limpa \\
\hline
Sujo & Painel com sujidade (poeira, dejetos, manchas) \\
\hline
\end{tabular}
\fonte{Do Autor (2025)}
\end{table}"""
    
    def _gerar_tabela_dataset(self, resultados: Dict) -> str:
        """Tabela 2: Banco de dados de treinamento"""
        
        # Extrair informações do primeiro fold
        fold_0 = resultados['folds'][0]
        y_true = np.array(fold_0['y_true'])
        
        limpos = int(np.sum(y_true == 0))
        sujos = int(np.sum(y_true == 1))
        total = limpos + sujos
        
        # Com augmentation 15x
        limpos_aug = limpos * 15
        sujos_aug = sujos * 15
        total_aug = total * 15
        
        perc_limpo = (limpos / total) * 100
        perc_sujo = (sujos / total) * 100
        
        return f"""\\begin{{table}}[htb]
\\caption{{Banco de dados de treinamento}}
\\label{{tab:dataset_treino}}
\\centering
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\textbf{{Classe}} & \\textbf{{Nº Original}} & \\textbf{{Após Augmentation}} & \\textbf{{Porcentagem}} \\\\
\\hline
Limpo & {limpos} & {limpos_aug} & {perc_limpo:.1f}\\% \\\\
\\hline
Sujo & {sujos} & {sujos_aug} & {perc_sujo:.1f}\\% \\\\
\\hline
\\textbf{{Total}} & \\textbf{{{total}}} & \\textbf{{{total_aug}}} & \\textbf{{100\\%}} \\\\
\\hline
\\end{{tabular}}
\\fonte{{Do Autor (2025)}}
\\end{{table}}"""
    
    def _gerar_tabela_configuracao(self, resultados: Dict) -> str:
        """Tabela 3: Configuração do treinamento"""
        
        config = resultados['configuracao']
        
        return f"""\\begin{{table}}[htb]
\\caption{{Configuração do treinamento}}
\\label{{tab:configuracao}}
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Parâmetro}} & \\textbf{{Valor}} \\\\
\\hline
Arquitetura & {config['arquitetura']} \\\\
\\hline
Número de Classes & {config['num_classes']} \\\\
\\hline
Batch Size & {config['batch_size']} \\\\
\\hline
Learning Rate & {config['learning_rate']} \\\\
\\hline
Dropout & {config['dropout']} \\\\
\\hline
Número de Épocas & {config['num_epocas']} \\\\
\\hline
Data Augmentation & {config['data_augmentation']} \\\\
\\hline
Validação Cruzada & {config['n_folds']}-fold \\\\
\\hline
\\end{{tabular}}
\\fonte{{Do Autor (2025)}}
\\end{{table}}"""
    
    def _gerar_tabela_resultados(self, resultados: Dict) -> str:
        """Tabela 4: Resultados da validação cruzada"""
        
        metricas = resultados['metricas_agregadas']
        
        # Calcular intervalo de confiança 95%
        ic_lower = metricas['acuracia_media'] - 1.96 * metricas['acuracia_std']
        ic_upper = metricas['acuracia_media'] + 1.96 * metricas['acuracia_std']
        
        return f"""\\begin{{table}}[htb]
\\caption{{Resultados da validação cruzada 5-fold}}
\\label{{tab:resultados}}
\\centering
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\textbf{{Métrica}} & \\textbf{{Média}} & \\textbf{{Desvio Padrão}} & \\textbf{{IC 95\\%}} \\\\
\\hline
Acurácia & {metricas['acuracia_media']:.4f} & {metricas['acuracia_std']:.4f} & [{ic_lower:.4f}, {ic_upper:.4f}] \\\\
\\hline
Precisão & {metricas['precision_media']:.4f} & - & - \\\\
\\hline
Recall & {metricas['recall_media']:.4f} & - & - \\\\
\\hline
F1-Score & {metricas['f1_media']:.4f} & - & - \\\\
\\hline
\\end{{tabular}}
\\fonte{{Do Autor (2025)}}
\\end{{table}}"""
    
    def _gerar_tabela_desempenho_classes(self, resultados: Dict) -> str:
        """Tabela 5: Desempenho por classe"""
        
        # Calcular matriz de confusão agregada
        cm_total = np.zeros((2, 2))
        for fold in resultados['folds']:
            cm_total += np.array(fold['confusion_matrix'])
        
        # Precisão por classe
        prec_limpo = cm_total[0,0] / (cm_total[0,0] + cm_total[1,0]) if (cm_total[0,0] + cm_total[1,0]) > 0 else 0
        prec_sujo = cm_total[1,1] / (cm_total[1,1] + cm_total[0,1]) if (cm_total[1,1] + cm_total[0,1]) > 0 else 0
        
        # Recall por classe
        rec_limpo = cm_total[0,0] / (cm_total[0,0] + cm_total[0,1]) if (cm_total[0,0] + cm_total[0,1]) > 0 else 0
        rec_sujo = cm_total[1,1] / (cm_total[1,1] + cm_total[1,0]) if (cm_total[1,1] + cm_total[1,0]) > 0 else 0
        
        # F1-Score
        f1_limpo = 2 * (prec_limpo * rec_limpo) / (prec_limpo + rec_limpo) if (prec_limpo + rec_limpo) > 0 else 0
        f1_sujo = 2 * (prec_sujo * rec_sujo) / (prec_sujo + rec_sujo) if (prec_sujo + rec_sujo) > 0 else 0
        
        # Médias
        metricas = resultados['metricas_agregadas']
        
        return f"""\\begin{{table}}[htb]
\\caption{{Desempenho de classificação por classe}}
\\label{{tab:desempenho_classe}}
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\textbf{{Classe}} & \\textbf{{Precisão}} & \\textbf{{Recall}} & \\textbf{{F1-Score}} & \\textbf{{Support}} \\\\
\\hline
Limpo & {prec_limpo:.4f} & {rec_limpo:.4f} & {f1_limpo:.4f} & {int(cm_total[0].sum())} \\\\
\\hline
Sujo & {prec_sujo:.4f} & {rec_sujo:.4f} & {f1_sujo:.4f} & {int(cm_total[1].sum())} \\\\
\\hline
\\textbf{{Média/Total}} & {metricas['precision_media']:.4f} & {metricas['recall_media']:.4f} & {metricas['f1_media']:.4f} & {int(cm_total.sum())} \\\\
\\hline
\\end{{tabular}}
\\fonte{{Do Autor (2025)}}
\\end{{table}}"""
    
    def salvar_json_completo(self, resultados: Dict):
        """Salva JSON completo com todos os resultados"""
        
        caminho = self.output_dir / 'resultados_completos.json'
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   ✅ JSON completo salvo: {caminho}")
    
    def gerar_resumo_markdown(self, resultados: Dict):
        """Gera resumo em Markdown"""
        
        metricas = resultados['metricas_agregadas']
        config = resultados['configuracao']
        
        resumo = f"""# Resumo do treinamento

## Resultados finais

**Acurácia Média:** {metricas['acuracia_media']:.2%} ± {metricas['acuracia_std']:.2%}

**Métricas:**
- Precisão: {metricas['precision_media']:.4f}
- Recall: {metricas['recall_media']:.4f}
- F1-Score: {metricas['f1_media']:.4f}

## Configuração

- Arquitetura: EfficientNet-B4
- Número de folds: {config['num_folds']}
- Épocas por fold: {config['num_epochs']}
- Lote (batch size): {config['batch_size']}
- Otimizador: {config['optimizer']}
- Taxa de aprendizado inicial: {config['lr_inicial']}

## Tempo de treinamento

- Tempo total: {resultados['tempo_total_horas']:.2f} horas
- Tempo médio por fold: {resultados['tempo_medio_por_fold_min']:.1f} minutos

## Arquivos gerados

- `matriz_confusao.png` - Matriz de confusão agregada
- `curvas_treinamento.png` - Curvas de loss e accuracy
- `comparacao_folds.png` - Comparação entre folds
- `tabelas_latex.tex` - Todas as tabelas em LaTeX
- `resultados_completos.json` - Resultados detalhados
- `modelo_fold*.pth` - Modelos treinados de cada fold

---

**Gerado em:** {resultados['inicio']}
"""
        
        caminho = self.output_dir / 'RESUMO.md'
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(resumo)
        
        print(f"   Resumo salvo em: {caminho}")
