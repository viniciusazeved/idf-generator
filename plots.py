"""
Figuras Plotly para o app IDF.

Todas as funcoes retornam go.Figure, sem dependencia do Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from idf import DistributionFitResult, IDFEquationResult

# Paleta consistente
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_daily_timeseries(series: pd.Series, station_name: str) -> go.Figure:
    """Serie temporal diaria de precipitacao."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        line=dict(width=0.5, color=_COLORS[0]),
        name="Precipitacao",
    ))
    fig.update_layout(
        title=f"Serie Temporal Diaria - {station_name}",
        xaxis_title="Data",
        yaxis_title="Precipitacao (mm)",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_annual_totals(series: pd.Series, station_name: str) -> go.Figure:
    """Totais anuais de precipitacao com linha da media."""
    annual = series.dropna().groupby(series.dropna().index.year).sum()
    mean_val = annual.mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=annual.index,
        y=annual.values,
        marker_color=_COLORS[0],
        name="Total Anual",
        opacity=0.7,
    ))
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Media: {mean_val:.0f} mm",
    )
    fig.update_layout(
        title=f"Precipitacao Anual - {station_name}",
        xaxis_title="Ano",
        yaxis_title="Precipitacao (mm)",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_availability(series: pd.Series) -> go.Figure:
    """Heatmap de disponibilidade por ano e mes."""
    df = series.to_frame("prec")
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["valid"] = df["prec"].notna().astype(int)

    pivot = df.pivot_table(values="valid", index="year", columns="month", aggfunc="mean")
    pivot.columns = [
        "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
        "Jul", "Ago", "Set", "Out", "Nov", "Dez",
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, "#d62728"], [0.5, "#ffdd57"], [1, "#2ca02c"]],
        colorbar=dict(title="Disponibilidade", tickformat=".0%"),
        zmin=0,
        zmax=1,
    ))
    fig.update_layout(
        title="Disponibilidade de Dados por Ano e Mes",
        xaxis_title="Mes",
        yaxis_title="Ano",
        template="plotly_white",
        height=max(400, len(pivot) * 8),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_annual_maxima(maxima: pd.Series) -> go.Figure:
    """Barras dos maximos anuais."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=maxima.index,
        y=maxima.values,
        marker_color=_COLORS[1],
        name="Maximo Anual",
    ))
    fig.add_hline(
        y=maxima.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Media: {maxima.mean():.1f} mm",
    )
    fig.update_layout(
        title="Precipitacao Maxima Diaria Anual",
        xaxis_title="Ano",
        yaxis_title="Precipitacao (mm)",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_distribution_fit(
    maxima: pd.Series,
    fit: DistributionFitResult,
) -> go.Figure:
    """Histograma dos maximos + PDF ajustada."""
    x = np.linspace(maxima.min() * 0.8, maxima.max() * 1.3, 200)
    y = fit.frozen.pdf(x)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=maxima.values,
        nbinsx=15,
        histnorm="probability density",
        name="Dados",
        marker_color=_COLORS[0],
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"PDF {fit.name}",
        line=dict(color=_COLORS[3], width=2),
    ))
    fig.update_layout(
        title=f"Ajuste da Distribuicao {fit.name}",
        xaxis_title="Precipitacao (mm)",
        yaxis_title="Densidade de Probabilidade",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_qq(
    theoretical: np.ndarray,
    sample: np.ndarray,
) -> go.Figure:
    """QQ-plot com linha 1:1."""
    min_val = min(theoretical.min(), sample.min())
    max_val = max(theoretical.max(), sample.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical,
        y=sample,
        mode="markers",
        marker=dict(color=_COLORS[0], size=8),
        name="Quantis",
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="1:1",
    ))
    fig.update_layout(
        title="QQ-Plot",
        xaxis_title="Quantis Teoricos (mm)",
        yaxis_title="Quantis Amostrais (mm)",
        template="plotly_white",
        height=450,
        width=500,
    )
    return fig


def plot_idf_curves(idf_table: pd.DataFrame) -> go.Figure:
    """Curvas IDF interativas: intensidade vs duracao para cada TR."""
    fig = go.Figure()

    for i, tr in enumerate(idf_table.columns):
        fig.add_trace(go.Scatter(
            x=idf_table.index,
            y=idf_table[tr],
            mode="lines+markers",
            name=f"TR = {tr} anos",
            line=dict(color=_COLORS[i % len(_COLORS)], shape="spline"),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="Curvas IDF (Intensidade-Duracao-Frequencia)",
        xaxis_title="Duracao (min)",
        yaxis_title="Intensidade (mm/h)",
        template="plotly_white",
        height=500,
        xaxis=dict(
            tickmode="array",
            tickvals=idf_table.index.tolist(),
            ticktext=[str(d) for d in idf_table.index],
        ),
        legend=dict(title="Tempo de Retorno"),
    )
    return fig


def plot_idf_comparison(
    idf_table: pd.DataFrame,
    eq: IDFEquationResult,
) -> go.Figure:
    """Comparacao entre IDF desagregada (real) e IDF da equacao (ajustada)."""
    fig = go.Figure()

    for i, tr in enumerate(idf_table.columns):
        color = _COLORS[i % len(_COLORS)]
        tr_num = float(tr)

        # Real (solido)
        fig.add_trace(go.Scatter(
            x=idf_table.index,
            y=idf_table[tr],
            mode="lines+markers",
            name=f"Desagregada TR={tr}",
            line=dict(color=color, shape="spline"),
            marker=dict(size=5),
        ))

        # Equacao (tracejado)
        durations = idf_table.index.to_numpy(dtype=float)
        i_pred = eq.K * (tr_num ** eq.a) / ((durations + eq.b) ** eq.c)
        fig.add_trace(go.Scatter(
            x=idf_table.index,
            y=i_pred,
            mode="lines",
            name=f"Equacao TR={tr}",
            line=dict(color=color, dash="dash", shape="spline"),
        ))

    fig.update_layout(
        title=f"Comparacao IDF: Desagregada vs Equacao (R²={eq.r_squared:.4f})",
        xaxis_title="Duracao (min)",
        yaxis_title="Intensidade (mm/h)",
        template="plotly_white",
        height=500,
        legend=dict(title="Legenda"),
    )
    return fig
