# Sistema de Inspeção de Painéis Solares

Repositório do backend e dos scripts de treinamento/avaliação utilizados no Trabalho de Conclusão de Curso em Engenharia Mecatrônica sobre detecção de módulos fotovoltaicos com YOLO11n e classificação binária de sujidade com EfficientNet-B4.

O foco deste repositório é o núcleo de aprendizado de máquina descrito no capítulo de Metodologia e Resultados do TCC:

- detecção de módulos em imagens aéreas com YOLO11n (Ultralytics),
- classificação de cada módulo em "limpo" ou "sujo" com EfficientNet-B4,
- scripts de preparação de dados, treinamento, avaliação e geração das tabelas/figuras do TCC.

As ideias de aplicação web completa (frontend com anotador de bounding boxes, painel de métricas, etc.) são apresentadas no texto do TCC como proposta de interface e trabalhos futuros. Este repositório publica principalmente o backend em FastAPI e os pipelines de treinamento.

---

## Estrutura do repositório

Na raiz do projeto:

```text
sistema-paineis-solares/
├── backend/                       # Backend FastAPI e scripts de treinamento/avaliação
├── figuras/                       # Figuras geradas (curvas de treino, matrizes de confusão, esquemas)
├── outputs/                       # Relatórios de treinamento, JSONs e métricas agregadas
├── runs/                          # Saídas de treino do YOLO11 (results.csv, results.png, etc.)
├── yolo11n.pt                     # Peso base do YOLO11n utilizado como ponto de partida
├── baixar_multiplos_roboflow.py   # Download em lote de vários datasets do Roboflow Universe
├── baixar_roboflow_rapido.py      # Download rápido do dataset "Aerial Solar Panels" (Brad Dwyer)
├── converter_datasets_para_yolo.py# Conversão/unificação de datasets em um único conjunto YOLO
├── gerar_figura3_2_matriz_confusao_esquematica.py
├── organizar_dataset_zenodo.py    # Organização de datasets do Zenodo (BDAPPV, DeepStat)
├── solucao_completa_yolo11.py     # Script integrador do fluxo de detecção com YOLO11n
├── treinar_yolo11_contornando_erro.py
├── validar_dataset_yolo.py        # Validação de anotações YOLO
└── README.md
```

Dentro de `backend/`, os arquivos mais importantes para o TCC incluem:

- `requirements.txt` – dependências do backend.
- `treinar_yolo11_gpu_otimizado.py` – treinamento estável do YOLO11n no dataset consolidado.
- `treinar_yolo11_roboflow_rapido.py` / `treinar_yolo11_aerial_rapido.py` – experimentos rápidos com datasets específicos do Roboflow.
- `treinar_2classes_meus_public_50_50.py` – treinamento principal do classificador EfficientNet-B4 binário no dataset balanceado 50/50.
- `treinar_5fold_dell_otimizado.py` – validação cruzada 5-fold estratificada do EfficientNet-B4.
- `avaliar_classificador_2classes.py` / `avaliar_2classes_meus_public_50_50.py` – avaliação do classificador em teste.
- `montar_dataset_meus_public_50_50.py` – montagem do dataset balanceado combinando dados próprios com públicos.
- `gerar_metricas_tcc_yolo11_teste.py` – cálculo das métricas do YOLO11n no conjunto de teste.
- `gerar_tabelas_tcc.py` e `geracao_documentacao.py` – geração automática de tabelas e relatórios usados no TCC.

---

## Requisitos e instalação

### Ambiente recomendado

- Python 3.10+ em Windows, Linux ou macOS.
- GPU NVIDIA com suporte a CUDA (ou execução em CPU, com tempos de treino maiores).
- Armazenamento adicional (por exemplo, um SSD externo) para os datasets públicos e datasets consolidados.

### Instalação do backend

```bash
cd backend

# Criar ambiente virtual (exemplo Windows)
python -m venv .venv
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

Os scripts de treinamento utilizam caminhos absolutos para os datasets (por exemplo, `F:\dataset_yolo_completo`, `F:\dataset_2classes_meus_public_50_50`). Ajuste esses caminhos nos scripts conforme o seu ambiente, se necessário.

---

## Datasets e reprodutibilidade

Este repositório **não** inclui os datasets de imagens. O TCC documenta detalhadamente, no Capítulo 3 e na seção 4.9, como obter os dados públicos e como o dataset misto foi construído.

Resumo das principais fontes de dados:

- **Roboflow Universe** – diversos datasets RGB de painéis solares ("Solar PV Maintenance Combined", "Aerial Solar Panels", "Soiling Detection", etc.).
- **Zenodo** – BDAPPV e DeepStat WP5 (mais de 100 mil recortes aéreos), utilizados para experimentos de detecção em larga escala.
- **Kaggle** – datasets de classificação binária e multiclasse (por exemplo, `hemanthsai7/solar-panel-dust-detection`).
- **Dados locais** – imagens de drone coletadas especificamente para o TCC, utilizadas para compor o dataset misto 50/50.

Para facilitar a reprodução dos experimentos, o TCC disponibiliza uma pasta pública no Google Drive com os datasets já organizados e modelos treinados:

- Google Drive (datasets preparados e modelos): consulte o link descrito na seção 4.9 do TCC.

Os scripts `baixar_multiplos_roboflow.py`, `baixar_roboflow_rapido.py`, `converter_datasets_para_yolo.py` e `organizar_dataset_zenodo.py` ilustram como automatizar o download e a organização dos dados públicos.

---

## Como reproduzir os experimentos do TCC

1. **Preparar o ambiente Python**
   - Criar o ambiente virtual em `backend/` e instalar as dependências.

2. **Obter os datasets**
   - Baixar os datasets públicos nas fontes originais (Roboflow, Zenodo, Kaggle) ou utilizar os arquivos já preparados do Google Drive.
   - Ajustar, se necessário, os caminhos de entrada nos scripts de treinamento para apontarem para os seus diretórios locais.

3. **Treinar o detector YOLO11n**
   - Usar `backend/treinar_yolo11_gpu_otimizado.py` com o dataset consolidado em formato YOLO (`dataset_yolo_completo`).
   - As saídas do treino (incluindo `results.csv` e `results.png`) ficam em `runs/`.

4. **Treinar o classificador EfficientNet-B4 binário (dataset 50/50)**
   - Usar `backend/treinar_2classes_meus_public_50_50.py` apontando para `dataset_2classes_meus_public_50_50`.
   - O script gera um `relatorio_treinamento.json` com as métricas de treino/validação e o checkpoint final do modelo.

5. **Executar a validação cruzada 5-fold**
   - Usar `backend/treinar_5fold_dell_otimizado.py` para reproduzir os resultados de 5-fold descritos no TCC.

6. **Avaliar os modelos e gerar métricas**
   - `backend/avaliar_classificador_2classes.py` e `backend/avaliar_2classes_meus_public_50_50.py` geram matrizes de confusão e métricas sobre o conjunto de teste.
   - `backend/gerar_metricas_tcc_yolo11_teste.py` calcula mAP, precision e recall do YOLO11n no conjunto de teste.
   - `backend/gerar_tabelas_tcc.py` e `backend/geracao_documentacao.py` consolidam os resultados em tabelas e figuras usadas no capítulo de resultados.

---

## Resultados principais (resumo)

Os valores a seguir estão detalhados e justificados no capítulo de Resultados do TCC. De forma resumida:

- **Detector YOLO11n** (dataset consolidado):
  - mAP@0.5 ≈ 0,556
  - mAP@0.5:0,95 ≈ 0,417
  - precision ≈ 0,765
  - recall ≈ 0,482

- **Classificador EfficientNet-B4 binário (dataset 50/50)**:
  - acurácia de teste ≈ 0,9721
  - matriz de confusão equilibrada entre as classes "limpo" e "sujo".

- **Validação cruzada 5-fold (EfficientNet-B4)**:
  - acurácia média ≈ 0,964
  - desvio padrão ≈ 0,0027 entre os folds.

Os tempos de treinamento por época e totais para YOLO11n, EfficientNet-B4 (1 fold) e EfficientNet-B4 (5 folds) são apresentados na Tabela 4.6 do TCC e foram estimados a partir dos logs de treino (`results.csv`, `relatorio_treinamento.json`, `resultados_completos.json`).

---

## Trabalhos futuros

Conforme discutido no TCC, os próximos passos naturais para esta linha de pesquisa incluem:

- concluir e integrar uma aplicação web completa para anotação de módulos, acompanhamento de treinamentos e visualização de métricas;
- incorporar geração de ortomosaicos e mapas de homogeneidade de sujidade a partir de campanhas reais com drone;
- explorar arquiteturas adicionais para detecção e classificação, incluindo modelos específicos para diferentes tipos de defeitos;
- avaliar o desempenho dos modelos em novos cenários e usinas, com variação de clima, sujidade e condições de iluminação.

---

## Licença

Este projeto foi desenvolvido como parte de um Trabalho de Conclusão de Curso em Engenharia Mecatrônica e está licenciado sob a MIT License.

