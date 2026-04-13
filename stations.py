"""
Catalogo de estacoes pluviometricas da ANA (fonte ANAF).

O ANAF e um catalogo filtrado contendo apenas estacoes com dados registrados.
Fonte: https://doi.org/10.5281/zenodo.3755065
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

_ANAF_URL = (
    "https://raw.githubusercontent.com/wallissoncarvalho/hydrobr/"
    "master/hydrobr/resources/ANAF_prec_stations.csv"
)


@st.cache_data(ttl=3600, show_spinner="Carregando catalogo de estacoes...")
def load_catalog() -> pd.DataFrame:
    """
    Carrega o catalogo ANAF de estacoes pluviometricas.

    Returns
    -------
    pd.DataFrame
        Colunas: Code, Name, Type, SubBasin, City, State, Responsible,
        Latitude, Longitude, StartDate, EndDate, NYD, MD, N_YWOMD, YWMD.
    """
    df = pd.read_csv(_ANAF_URL)
    df["Code"] = df["Code"].apply(lambda x: f"{int(x):08}")
    return df


def get_states(catalog: pd.DataFrame) -> list[str]:
    """Retorna lista ordenada de estados brasileiros no catalogo."""
    # Filtrar paises vizinhos que aparecem no ANAF
    exclude = {
        "ARGENTINA", "BOLIVIA", "PARAGUAI", "PERU",
        "GUIANA FRANCESA", "SURINAME", "COLOMBIA",
        "GUIANA", "VENEZUELA", "URUGUAI",
    }
    states = catalog["State"].dropna().unique()
    return sorted(s for s in states if s.upper() not in exclude)


def get_cities(catalog: pd.DataFrame, state: str) -> list[str]:
    """Retorna lista ordenada de cidades para um estado."""
    mask = catalog["State"] == state
    return sorted(catalog.loc[mask, "City"].dropna().unique())


def get_stations(catalog: pd.DataFrame, state: str, city: str) -> pd.DataFrame:
    """Retorna estacoes filtradas por estado e cidade."""
    mask = (catalog["State"] == state) & (catalog["City"] == city)
    return catalog[mask].reset_index(drop=True)
