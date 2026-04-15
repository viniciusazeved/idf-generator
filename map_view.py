"""
Mapa interativo de estacoes pluviometricas para selecao espacial.

Usa folium para renderizar estacoes do catalogo ANAF coloridas por
qualidade de dados (anos, falhas, recencia).
"""
from __future__ import annotations

from datetime import datetime

import folium
import numpy as np
import pandas as pd
import streamlit as st

# Cores alinhadas com a paleta de plots.py
_QUALITY_COLORS = {
    "Excelente": "#2ca02c",
    "Moderada": "#ff7f0e",
    "Limitada": "#d62728",
}


# ---------------------------------------------------------------------------
# Scoring de qualidade
# ---------------------------------------------------------------------------

@st.cache_data
def compute_quality_score(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas quality_score (0-1) e quality_label ao catalogo.

    Score composto:
        50% NYD (anos de dados, saturando em 30)
        30% completude (100 - MD%, saturando em 50%)
        20% recencia (anos desde EndDate, saturando em 20)
    """
    df = catalog.copy()

    # Sub-score NYD
    nyd = df["NYD"].fillna(0).astype(float)
    s_nyd = (nyd / 30).clip(upper=1.0)

    # Sub-score falhas (MD = missing data %)
    md = df["MD"].fillna(100).astype(float)
    s_md = (1.0 - md / 50).clip(lower=0.0)

    # Sub-score recencia
    end_dates = pd.to_datetime(df["EndDate"], errors="coerce")
    current_year = datetime.now().year
    years_since = current_year - end_dates.dt.year.fillna(1950)
    s_rec = (1.0 - years_since / 20).clip(lower=0.0, upper=1.0)

    df["quality_score"] = 0.5 * s_nyd + 0.3 * s_md + 0.2 * s_rec

    df["quality_label"] = pd.cut(
        df["quality_score"],
        bins=[-np.inf, 0.6, 0.9, np.inf],
        labels=["Limitada", "Moderada", "Excelente"],
    )

    return df


# ---------------------------------------------------------------------------
# Criacao do mapa
# ---------------------------------------------------------------------------

_LEGEND_HTML = """
<div style="
    position: fixed; bottom: 30px; right: 30px; z-index: 1000;
    background: white; padding: 10px 14px; border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px;
    line-height: 1.6;
">
<b>Qualidade dos dados</b><br>
<svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#2ca02c"/></svg>
 Excelente<br>
<svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#ff7f0e"/></svg>
 Moderada<br>
<svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#d62728"/></svg>
 Limitada
</div>
"""


@st.cache_resource
def create_station_map(
    _scored_catalog: pd.DataFrame,
    min_years: int,
    _catalog_len: int,
) -> folium.Map:
    """
    Cria mapa folium com estacoes via GeoJSON + JS (leve).

    Parameters
    ----------
    _scored_catalog : pd.DataFrame
        Catalogo com colunas quality_score e quality_label.
    min_years : int
        Filtro minimo de anos de dados (NYD).
    _catalog_len : int
        Comprimento do catalogo (para invalidacao de cache).
    """
    m = folium.Map(
        location=[-14.2, -51.9],
        zoom_start=4,
        tiles="CartoDB positron",
    )

    df = _scored_catalog.copy()
    df = df.dropna(subset=["Latitude", "Longitude"])
    if min_years > 0:
        df = df[df["NYD"].fillna(0) >= min_years]

    # Construir GeoJSON (muito mais leve que CircleMarkers individuais)
    features = []
    for _, row in df.iterrows():
        label = str(row.get("quality_label", "Limitada"))
        color = _QUALITY_COLORS.get(label, "#d62728")
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["Longitude"]), float(row["Latitude"])],
            },
            "properties": {
                "code": str(row["Code"]),
                "name": str(row["Name"]),
                "city": str(row.get("City", "?")),
                "state": str(row.get("State", "?")),
                "nyd": int(row.get("NYD", 0)),
                "md": float(row.get("MD", 0)),
                "start": str(row.get("StartDate", "?")),
                "end": str(row.get("EndDate", "?")),
                "quality": label,
                "color": color,
            },
        })

    geojson_data = {"type": "FeatureCollection", "features": features}

    # Usar GeoJson com pointToLayer JS para renderizar CircleMarkers no browser
    folium.GeoJson(
        geojson_data,
        name="Estacoes",
        marker=folium.CircleMarker(radius=6, fill=True, fill_opacity=0.8, weight=1),
        style_function=lambda feat: {
            "color": feat["properties"]["color"],
            "fillColor": feat["properties"]["color"],
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["code", "name"],
            aliases=["", ""],
            localize=False,
            labels=False,
            style="font-size: 12px;",
        ),
        popup=folium.GeoJsonPopup(
            fields=["code", "name", "city", "state", "nyd", "md", "start", "end", "quality"],
            aliases=["Codigo", "Nome", "Cidade", "Estado", "Anos", "Falhas (%)", "Inicio", "Fim", "Qualidade"],
            localize=False,
            max_width=300,
        ),
    ).add_to(m)

    # Legenda
    m.get_root().html.add_child(folium.Element(_LEGEND_HTML))

    return m


# ---------------------------------------------------------------------------
# Resolucao de cliques
# ---------------------------------------------------------------------------

def resolve_clicked_station(
    scored_catalog: pd.DataFrame,
    last_clicked: dict,
) -> pd.Series | None:
    """
    Encontra a estacao mais proxima do ponto clicado no mapa.

    Returns None se nenhuma estacao estiver a menos de 0.01 graus (~1 km).
    """
    lat = last_clicked.get("lat")
    lng = last_clicked.get("lng")
    if lat is None or lng is None:
        return None

    df = scored_catalog.dropna(subset=["Latitude", "Longitude"])
    dist = (df["Latitude"] - lat) ** 2 + (df["Longitude"] - lng) ** 2
    idx_min = dist.idxmin()
    if dist[idx_min] > 0.01**2:
        return None
    return df.loc[idx_min]
