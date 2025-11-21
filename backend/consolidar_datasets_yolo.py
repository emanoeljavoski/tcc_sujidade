"""Ferramentas para consolidar múltiplos datasets YOLO em um único dataset.

Este script:
- Varre diretórios em F:/datasets_publicos_rgb e F:/dataset_yolo_detector_modulos
- Localiza arquivos data.yaml/dataset.yaml de datasets no formato YOLO
- Coleta todas as imagens + labels válidos
- Reparte em splits estratificados por origem (train/val/test 70/20/10)
- Copia tudo para F:/dataset_yolo_completo
- Gera dataset.yaml unificado e um relatório JSON com estatísticas

Uso recomendado (no PowerShell, com o venv ativado):

    cd F:\tccemanoel\sistema-paineis-solares\backend
    ..\.venv\Scripts\python.exe consolidar_datasets_yolo.py

O script é idempotente no sentido de que se F:/dataset_yolo_completo já
existir e não estiver vazio, ele aborta para não misturar dados antigos
com novos. Apague manualmente a pasta se quiser recriar tudo.
"""
from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

# Raízes onde procurar datasets YOLO
# Observação: alguns datasets baixados anteriormente foram parar em
# F:/datasets_publicos (por exemplo, Aerial-Solar-Panels-13), enquanto
# outros ficam em F:/datasets_publicos_rgb. Incluímos todas essas
# raízes para que a consolidação realmente use "todos" os datasets
# disponíveis no disco.
DATASET_ROOTS: List[Path] = [
    Path("F:/datasets_publicos_rgb/aereos_drone"),
    Path("F:/datasets_publicos_rgb/solo_binario"),
    Path("F:/dataset_yolo_detector_modulos"),  # se existir
    Path("F:/datasets_publicos"),
]

# Destino consolidado
DEST_ROOT = Path("F:/dataset_yolo_completo")

# Semente para reprodutibilidade
RNG_SEED = 42

# Proporções dos splits
SPLIT_RATIOS: Dict[str, float] = {"train": 0.7, "val": 0.2, "test": 0.1}


@dataclass
class Sample:
    """Representa um par imagem/label e sua origem."""

    image: Path
    label: Path
    source: str  # nome do dataset de origem


@dataclass
class SourceSummary:
    """Resumo de um dataset de origem após consolidação."""

    name: str
    total: int = 0
    per_split: Dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})


def configurar_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def localizar_yaml_datasets(roots: Sequence[Path]) -> List[Path]:
    """Encontra arquivos YAML candidatos a dataset YOLO.

    Procura por data.yaml ou dataset.yaml em cada raiz fornecida.
    """

    yaml_paths: List[Path] = []
    for root in roots:
        if not root.exists():
            logging.info("Raiz %s não existe; ignorando.", root)
            continue
        for path in root.rglob("*.yaml"):
            if path.name.lower() in {"data.yaml", "dataset.yaml"}:
                yaml_paths.append(path)
    return yaml_paths


def carregar_yaml(path: Path) -> Optional[dict]:
    """Carrega um YAML de forma segura.

    Retorna None em caso de erro.
    """

    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Falha ao ler YAML %s: %s", path, exc)
        return None


def _resolver_subpasta(base: Path, value: str) -> Path:
    """Resolve um caminho de subpasta de imagens/labels baseado no YAML.

    Muitos data.yaml usam caminhos relativos como '../train/images'.
    """

    return (base / value).resolve()


def coletar_amostras_de_yaml(yaml_path: Path) -> List[Sample]:
    """Coleta todas as amostras válidas (imagem+label) de um dataset YOLO.

    Considera splits train/val/test (ou valid). Não assume que os splits
    já estão no formato 70/20/10 – eles são apenas usados como fonte de
    imagens, pois a redistribuição será feita depois.
    """

    dados = carregar_yaml(yaml_path)
    if not dados:
        return []

    base = yaml_path.parent
    source_name = base.name

    split_keys = {
        "train": dados.get("train"),
        "val": dados.get("val") or dados.get("valid"),
        "test": dados.get("test"),
    }

    amostras: List[Sample] = []

    for split, rel_path in split_keys.items():
        if not rel_path:
            continue
        images_dir = _resolver_subpasta(base, rel_path)

        # Fallback: alguns datasets (como os baixados via Roboflow) usam
        # caminhos relativos do tipo '../train/images' mesmo quando o
        # conteúdo está em 'BASE/train/images'. Se o caminho resolvido
        # não existir, tentamos localizar essas pastas padrão sob a
        # própria pasta base do dataset.
        if not images_dir.exists():
            alt_dir = None
            if split == "train":
                cand = base / "train" / "images"
                if cand.exists():
                    alt_dir = cand
            elif split == "val":
                for nome in ("val", "valid"):
                    cand = base / nome / "images"
                    if cand.exists():
                        alt_dir = cand
                        break
            elif split == "test":
                cand = base / "test" / "images"
                if cand.exists():
                    alt_dir = cand

            if alt_dir is not None:
                logging.info(
                    "Usando diretório alternativo de imagens %s (dataset %s, split %s)",
                    alt_dir,
                    source_name,
                    split,
                )
                images_dir = alt_dir
            else:
                logging.warning(
                    "Diretório de imagens %s não existe (dataset %s, split %s)",
                    images_dir,
                    source_name,
                    split,
                )
                continue

        # Assumimos convenção YOLO: labels/ é paralelo a images/
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.exists():
            logging.warning("Diretório de labels %s não existe (dataset %s)", labels_dir, source_name)
            continue

        # Coletar todos os arquivos de imagem
        imagens = list(images_dir.glob("*.*"))
        if not imagens:
            logging.info("Nenhuma imagem encontrada em %s", images_dir)
            continue

        logging.info("%s: encontrado %d arquivos de imagem em %s (%s)", source_name, len(imagens), images_dir, split)

        for img in imagens:
            label = labels_dir / (img.stem + ".txt")
            if not label.exists():
                # Ignorar imagens sem label correspondente
                continue
            amostras.append(Sample(image=img, label=label, source=source_name))

    return amostras


def repartir_por_origem(amostras: Sequence[Sample]) -> Tuple[Dict[str, List[Sample]], Dict[str, SourceSummary]]:
    """Reparte amostras em train/val/test estratificado por origem.

    Retorna um dicionário split->lista de Samples e um resumo por origem.
    """

    rng = random.Random(RNG_SEED)

    # Agrupar por origem
    por_origem: Dict[str, List[Sample]] = {}
    for s in amostras:
        por_origem.setdefault(s.source, []).append(s)

    por_split: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}
    summaries: Dict[str, SourceSummary] = {}

    for source, lista in por_origem.items():
        if not lista:
            continue
        rng.shuffle(lista)
        n = len(lista)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))
        # Garante que a soma não ultrapasse
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        train_samples = lista[:n_train]
        val_samples = lista[n_train:n_train + n_val]
        test_samples = lista[n_train + n_val:]

        por_split["train"].extend(train_samples)
        por_split["val"].extend(val_samples)
        por_split["test"].extend(test_samples)

        summaries[source] = SourceSummary(
            name=source,
            total=n,
            per_split={
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
        )

    return por_split, summaries


def preparar_destino(dest_root: Path) -> None:
    """Cria a estrutura de diretórios do dataset consolidado.

    Se o diretório já existir e não estiver vazio, aborta para evitar
    misturar dados antigos.
    """

    if dest_root.exists():
        # Verificar se está vazio
        if any(dest_root.iterdir()):
            raise RuntimeError(
                f"O diretório {dest_root} já existe e não está vazio. "
                "Apague-o manualmente se realmente quiser recriar o dataset consolidado."
            )
    dest_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (dest_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dest_root / split / "labels").mkdir(parents=True, exist_ok=True)


def copiar_amostras(dest_root: Path, por_split: Dict[str, List[Sample]]) -> Dict[str, int]:
    """Copia imagens e labels para o destino consolidado.

    Retorna contagem de imagens por split.
    """

    contagens: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split, samples in por_split.items():
        if not samples:
            continue
        img_dest_dir = dest_root / split / "images"
        lbl_dest_dir = dest_root / split / "labels"

        for s in samples:
            # Prefixar nome do arquivo com a origem para minimizar colisões
            img_name = f"{s.source}__{s.image.name}"
            lbl_name = f"{s.source}__{s.label.name}"

            dest_img = img_dest_dir / img_name
            dest_lbl = lbl_dest_dir / lbl_name

            try:
                shutil.copy2(s.image, dest_img)
                shutil.copy2(s.label, dest_lbl)
                contagens[split] += 1
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("Falha ao copiar %s ou %s: %s", s.image, s.label, exc)

    return contagens


def escrever_dataset_yaml(dest_root: Path) -> None:
    """Escreve o dataset.yaml unificado no destino."""

    yaml_path = dest_root / "dataset.yaml"
    conteudo = {
        "path": str(dest_root).replace("\\", "/"),  # uso genérico
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["solar_module"],
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(conteudo, f, sort_keys=False, allow_unicode=True)
    logging.info("dataset.yaml criado em %s", yaml_path)


def escrever_relatorio(dest_root: Path, summaries: Dict[str, SourceSummary], contagens: Dict[str, int]) -> None:
    """Gera arquivo JSON com estatísticas de consolidação."""

    total_datasets = len(summaries)
    total_imagens = sum(s.total for s in summaries.values())

    relatorio = {
        "destino": str(dest_root),
        "total_datasets": total_datasets,
        "total_imagens": total_imagens,
        "splits": contagens,
        "datasets": [
            {
                "nome": s.name,
                "total_imagens": s.total,
                "train": s.per_split["train"],
                "val": s.per_split["val"],
                "test": s.per_split["test"],
            }
            for s in sorted(summaries.values(), key=lambda x: x.name)
        ],
    }

    rel_path = dest_root / "relatorio_consolidacao.json"
    with rel_path.open("w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    logging.info("Relatório de consolidação salvo em %s", rel_path)


def main() -> None:
    """Ponto de entrada principal."""

    configurar_logging()

    logging.info("Iniciando consolidação de datasets YOLO...")

    yaml_paths = localizar_yaml_datasets(DATASET_ROOTS)
    if not yaml_paths:
        logging.error("Nenhum data.yaml ou dataset.yaml encontrado nas raízes configuradas.")
        raise SystemExit(1)

    logging.info("Foram encontrados %d arquivos YAML de datasets.", len(yaml_paths))

    todas_amostras: List[Sample] = []
    for yml in yaml_paths:
        amostras = coletar_amostras_de_yaml(yml)
        if not amostras:
            continue
        todas_amostras.extend(amostras)

    if not todas_amostras:
        logging.error("Nenhuma amostra válida (imagem+label) encontrada em nenhum dataset.")
        raise SystemExit(1)

    logging.info("Total de amostras válidas encontradas: %d", len(todas_amostras))

    por_split, summaries = repartir_por_origem(todas_amostras)

    preparar_destino(DEST_ROOT)

    contagens = copiar_amostras(DEST_ROOT, por_split)

    escrever_dataset_yaml(DEST_ROOT)
    escrever_relatorio(DEST_ROOT, summaries, contagens)

    logging.info("Consolidação concluída com sucesso.")
    logging.info("Resumo final de imagens por split: %s", contagens)


if __name__ == "__main__":
    main()
