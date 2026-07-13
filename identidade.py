"""
Identidade visual (marca) do app e do relatorio IDF.

O idf-generator nasceu como app do LAPLA (FECFAU/Unicamp) e passou a ser usado
tambem pela Azevedo Consultoria Ambiental e Energetica. Em vez de fixar a marca
no codigo, a identidade e um dict sobrescrivivel — mesmo padrao do chuva_vazao
(`chuva_vazao/identidade.py`) e do Hidroenergetico.

Marca ativa:
    - default: Azevedo.
    - override: variavel de ambiente ``IDF_MARCA`` = "lapla" | "azevedo"
      (ou st.secrets["marca"] no Streamlit Cloud, se presente).

Os logos ficam na RAIZ do projeto (layout flat do idf-generator).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent


IDENTIDADE_AZEVEDO: dict[str, str] = {
    "marca": "azevedo",
    "logo_filename": "logo_azevedo.png",
    # Rodape do PDF (linha central)
    "rodape_pdf": "Gerador de Curvas IDF  -  Azevedo Consultoria Ambiental e Energética",
    # Creditos na sidebar do app (markdown do st.caption)
    "creditos_sidebar": (
        "**Azevedo** - Consultoria Ambiental e Energética\n\n"
        "[Repositório](https://github.com/viniciusazeved/idf-generator)"
    ),
}

IDENTIDADE_LAPLA: dict[str, str] = {
    "marca": "lapla",
    "logo_filename": "logo_lapla.png",
    "rodape_pdf": "Gerado por Gerador de Curvas IDF  -  LAPLA - FECFAU/Unicamp",
    "creditos_sidebar": (
        "**LAPLA** - Laboratorio de Planejamento Ambiental\n\n"
        "FECFAU / Unicamp\n\n"
        "[Repositorio](https://github.com/viniciusazeved/idf-generator)"
    ),
}

_MARCAS: dict[str, dict[str, str]] = {
    "azevedo": IDENTIDADE_AZEVEDO,
    "lapla": IDENTIDADE_LAPLA,
}


def _marca_ativa_key() -> str:
    """Le a marca ativa de env var (ou st.secrets), default 'azevedo'."""
    marca = os.environ.get("IDF_MARCA", "").strip().lower()
    if not marca:
        try:
            import streamlit as st  # noqa: PLC0415

            marca = str(st.secrets.get("marca", "")).strip().lower()
        except Exception:
            marca = ""
    return marca if marca in _MARCAS else "azevedo"


def identidade_ativa(override: dict[str, str] | None = None) -> dict[str, str]:
    """
    Dict da marca ativa (default Azevedo), com ``override`` parcial mesclado por
    cima. Sempre inclui ``logo_path`` resolvido (absoluto), quando o arquivo
    existe na raiz do projeto.
    """
    base = dict(_MARCAS[_marca_ativa_key()])
    if override:
        base = {**base, **override}
    logo = ROOT_DIR / base.get("logo_filename", "")
    base["logo_path"] = str(logo) if logo.is_file() else ""
    return base
