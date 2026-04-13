"""
Cliente para a API de dados historicos da ANA (Agencia Nacional de Aguas).

Baseado no get_data.py do projeto TensorHydro, extraindo e limpando
apenas a parte de precipitacao com retry e tratamento de erros.

API: http://telemetriaws1.ana.gov.br/ServiceANA.asmx
"""
from __future__ import annotations

import calendar
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ANAConnectionError(Exception):
    """Erro de conexao com a API da ANA."""


_ANA_BASE = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx"


def _build_session() -> requests.Session:
    """Cria sessao HTTP com retry automatico."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_daily_precipitation(
    station_code: str,
    only_consisted: bool = False,
    on_progress: callable | None = None,
) -> pd.Series:
    """
    Baixa serie diaria de precipitacao de uma estacao da ANA.

    Parameters
    ----------
    station_code : str
        Codigo da estacao (ex: '02244133').
    only_consisted : bool
        Se True, retorna apenas dados consistidos (nivel 2).
    on_progress : callable, optional
        Callback chamado com (etapa, total) para progresso.

    Returns
    -------
    pd.Series
        Serie diaria indexada por DatetimeIndex, NaN para falhas.

    Raises
    ------
    ANAConnectionError
        Se a API estiver inacessivel apos retries.
    ValueError
        Se nao houver dados para a estacao.
    """
    session = _build_session()
    params = {
        "codEstacao": str(station_code),
        "dataInicio": "",
        "dataFim": "",
        "tipoDados": "2",  # precipitacao
        "nivelConsistencia": "",
    }

    try:
        response = session.get(
            f"{_ANA_BASE}/HidroSerieHistorica",
            params=params,
            timeout=(10, 120),
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise ANAConnectionError(
            f"Falha ao conectar com a API da ANA para estacao {station_code}: {e}"
        ) from e

    tree = ET.ElementTree(ET.fromstring(response.content))
    root = tree.getroot()

    frames: list[pd.DataFrame] = []
    code = f"{int(station_code):08}"

    for month_elem in root.iter("SerieHistorica"):
        consist = int(month_elem.find("NivelConsistencia").text)
        date_str = month_elem.find("DataHora").text
        date = pd.to_datetime(date_str, dayfirst=False)
        date = pd.Timestamp(date.year, date.month, 1)
        last_day = calendar.monthrange(date.year, date.month)[1]
        month_dates = pd.date_range(date, periods=last_day, freq="D")

        values: list[float | None] = []
        consist_levels: list[int] = []

        for i in range(last_day):
            tag = f"Chuva{i + 1:02d}"
            elem = month_elem.find(tag)
            if elem is not None and elem.text is not None:
                try:
                    values.append(float(elem.text))
                except (ValueError, TypeError):
                    values.append(None)
            else:
                values.append(None)
            consist_levels.append(consist)

        idx = pd.MultiIndex.from_arrays(
            [month_dates, consist_levels],
            names=["Date", "Consistence"],
        )
        frames.append(pd.DataFrame({code: values}, index=idx))

    if not frames:
        raise ValueError(f"Nenhum dado encontrado para a estacao {station_code}.")

    df = pd.concat(frames).sort_index()

    if only_consisted:
        df = df[df.index.get_level_values("Consistence") == 2]
        df = df.droplevel("Consistence")
        if df.empty:
            raise ValueError(
                f"Nenhum dado consistido encontrado para a estacao {station_code}."
            )
    else:
        # Prioriza consistidos: em datas duplicadas, keep='last' apos sort
        # (consistidos vem depois dos brutos no MultiIndex ordenado)
        dup_mask = df.droplevel("Consistence").index.duplicated(keep="last")
        df = df[~dup_mask].droplevel("Consistence")

    series = df[code].astype(float)

    # Reindexar para cobrir todo o periodo sem lacunas no indice
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index)
    series.index.name = "Data"

    return series
