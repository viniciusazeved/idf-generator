"""
Modulo de calculo IDF (Intensidade-Duracao-Frequencia).

Contem: extracao de maximos anuais, ajuste de distribuicoes (Gumbel, GEV),
desagregacao de chuva, construcao da tabela IDF e ajuste da equacao IDF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import genextreme, goodness_of_fit, gumbel_r

from disaggregation import DURATIONS_MIN, disaggregate_24h, to_intensity


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class DistributionFitResult:
    """Resultado do ajuste de distribuicao."""
    name: str
    params: dict[str, float]
    frozen: object  # scipy frozen distribution

    @property
    def loc(self) -> float:
        return self.params["loc"]

    @property
    def scale(self) -> float:
        return self.params["scale"]


@dataclass
class GoFTestResult:
    """Resultado do teste de aderencia (Anderson-Darling, Monte Carlo)."""
    statistic: float
    p_value: float


@dataclass
class IDFEquationResult:
    """Resultado do ajuste da equacao IDF."""
    K: float
    a: float
    b: float
    c: float
    r_squared: float
    rmse: float
    mae: float


# ---------------------------------------------------------------------------
# Maximos anuais
# ---------------------------------------------------------------------------

def compute_annual_maxima(
    series: pd.Series,
    start_year: int,
    end_year: int,
    hydrological_year: bool = False,
) -> pd.Series:
    """
    Extrai maximos anuais de precipitacao diaria.

    Parameters
    ----------
    series : pd.Series
        Serie diaria de precipitacao (mm).
    start_year : int
        Ano inicial do periodo de analise.
    end_year : int
        Ano final do periodo de analise.
    hydrological_year : bool
        Se True, usa ano hidrologico (Out-Set), rotulado pelo ano final.

    Returns
    -------
    pd.Series
        Maximos anuais indexados pelo ano.
    """
    s = series.dropna().copy()
    s = s[s > 0]  # remover zeros exatos (dias sem chuva)

    if hydrological_year:
        # Ano hidrologico: Out do ano N-1 a Set do ano N -> rotulo N
        s = s.copy()
        hydro_year = s.index.year + (s.index.month >= 10).astype(int)
        s.index = hydro_year
        s.index.name = "Ano"
        maxima = s.groupby(level=0).max()
    else:
        maxima = s.groupby(s.index.year).max()
        maxima.index.name = "Ano"

    maxima = maxima[(maxima.index >= start_year) & (maxima.index <= end_year)]
    return maxima


# ---------------------------------------------------------------------------
# Ajuste de distribuicoes
# ---------------------------------------------------------------------------

def fit_gumbel(annual_maxima: pd.Series) -> DistributionFitResult:
    """Ajusta distribuicao Gumbel (maximos) aos dados."""
    loc, scale = gumbel_r.fit(annual_maxima.values)
    frozen = gumbel_r(loc=loc, scale=scale)
    return DistributionFitResult(
        name="Gumbel",
        params={"loc": loc, "scale": scale},
        frozen=frozen,
    )


def fit_gev(annual_maxima: pd.Series) -> DistributionFitResult:
    """Ajusta distribuicao GEV (Generalized Extreme Value) aos dados.

    Nota: params["xi (shape)"] usa convencao hidrologica (xi = -c do scipy).
    O atributo ``frozen`` usa o sinal interno do scipy (c = -xi).
    """
    c, loc, scale = genextreme.fit(annual_maxima.values)
    frozen = genextreme(c=c, loc=loc, scale=scale)
    # scipy usa c com sinal oposto a convencao hidrologica (xi = -c)
    return DistributionFitResult(
        name="GEV",
        params={"\u03be (shape)": -c, "loc": loc, "scale": scale},
        frozen=frozen,
    )


# ---------------------------------------------------------------------------
# Teste de aderencia
# ---------------------------------------------------------------------------

def gof_test(annual_maxima: pd.Series, fit: DistributionFitResult) -> GoFTestResult:
    """
    Executa teste de aderencia Anderson-Darling com bootstrap parametrico.

    Usa Monte Carlo para calcular o p-value corretamente mesmo quando os
    parametros da distribuicao sao estimados a partir dos proprios dados.
    """
    dist = gumbel_r if fit.name == "Gumbel" else genextreme
    result = goodness_of_fit(
        dist, annual_maxima.values, statistic="ad", n_mc_samples=500, random_state=42,
    )
    return GoFTestResult(statistic=result.statistic, p_value=result.pvalue)


def qq_data(
    annual_maxima: pd.Series, fit: DistributionFitResult
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula quantis teoricos e amostrais para QQ-plot.

    Returns
    -------
    (theoretical, sample)
    """
    sorted_data = np.sort(annual_maxima.values)
    n = len(sorted_data)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = fit.frozen.ppf(probabilities)
    return theoretical, sorted_data


# ---------------------------------------------------------------------------
# Precipitacao por tempo de retorno
# ---------------------------------------------------------------------------

def return_period_precipitation(
    fit: DistributionFitResult,
    tr_values: list[int],
) -> dict[int, float]:
    """
    Calcula precipitacao maxima diaria para cada tempo de retorno.

    Parameters
    ----------
    fit : DistributionFitResult
        Distribuicao ajustada.
    tr_values : list[int]
        Tempos de retorno em anos (ex: [2, 5, 10, 25, 50, 100]).

    Returns
    -------
    dict[int, float]
        {TR_anos: precipitacao_mm}.
    """
    return {
        tr: float(fit.frozen.ppf(1 - 1 / tr))
        for tr in tr_values
    }


# ---------------------------------------------------------------------------
# Tabela IDF
# ---------------------------------------------------------------------------

def compute_idf_table(precip_by_tr: dict[int, float]) -> pd.DataFrame:
    """
    Constroi tabela IDF completa: intensidade (mm/h) por duracao e TR.

    Aplica desagregacao DNAEE a cada precipitacao de TR.

    Parameters
    ----------
    precip_by_tr : dict[int, float]
        {TR_anos: precipitacao_maxima_diaria_mm}.

    Returns
    -------
    pd.DataFrame
        Index = duracao (min), columns = TR (anos), values = intensidade (mm/h).
    """
    result: dict[int, dict[int, float]] = {}

    for tr, p24h in precip_by_tr.items():
        depths = disaggregate_24h(p24h)
        intensities = to_intensity(depths)
        result[tr] = intensities

    df = pd.DataFrame(result)
    df.index.name = "Duracao (min)"
    df.columns.name = "TR (anos)"
    df = df.loc[sorted(df.index)]
    return df


# ---------------------------------------------------------------------------
# Equacao IDF
# ---------------------------------------------------------------------------

def _idf_equation(data: tuple, K: float, a: float, b: float, c: float) -> np.ndarray:
    """Modelo IDF: i = K * TR^a / (t + b)^c"""
    TR, t = data
    return K * (TR ** a) / ((t + b) ** c)


def fit_idf_equation(idf_table: pd.DataFrame) -> IDFEquationResult:
    """
    Ajusta a equacao IDF: i = K * TR^a / (t + b)^c.

    Parameters
    ----------
    idf_table : pd.DataFrame
        Tabela IDF (index=duracao min, columns=TR anos, values=intensidade mm/h).

    Returns
    -------
    IDFEquationResult
        Coeficientes ajustados e metricas de erro.

    Raises
    ------
    RuntimeError
        Se o ajuste nao convergir.
    """
    tr_values = idf_table.columns.to_numpy(dtype=float)
    durations = idf_table.index.to_numpy(dtype=float)

    # Montar vetores para curve_fit
    tr_list, t_list, intensity_list = [], [], []
    for t in durations:
        for tr in tr_values:
            tr_list.append(tr)
            t_list.append(t)
            intensity_list.append(idf_table.loc[t, idf_table.columns[idf_table.columns == tr][0]])

    TR_arr = np.array(tr_list, dtype=float)
    t_arr = np.array(t_list, dtype=float)
    i_arr = np.array(intensity_list, dtype=float)

    # Bounds fisicamente coerentes:
    # K > 0, 0 < a < 1, 0 < b < 100, 0 < c < 2
    popt, _ = curve_fit(
        _idf_equation,
        (TR_arr, t_arr),
        i_arr,
        p0=[1000.0, 0.2, 10.0, 0.8],
        bounds=([0, 0, 0, 0], [np.inf, 1.0, 100.0, 2.0]),
        maxfev=20000,
    )

    K, a, b, c = popt

    # Metricas de erro
    i_pred = _idf_equation((TR_arr, t_arr), K, a, b, c)
    residuals = i_arr - i_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((i_arr - np.mean(i_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    return IDFEquationResult(
        K=float(K),
        a=float(a),
        b=float(b),
        c=float(c),
        r_squared=float(r_squared),
        rmse=rmse,
        mae=mae,
    )


def idf_equation_predict(
    eq: IDFEquationResult,
    tr_values: list[int],
    durations: list[int] | None = None,
) -> pd.DataFrame:
    """
    Gera tabela de intensidades a partir da equacao IDF ajustada.

    Parameters
    ----------
    eq : IDFEquationResult
        Equacao ajustada.
    tr_values : list[int]
        Tempos de retorno.
    durations : list[int], optional
        Duracoes em minutos. Se None, usa DURATIONS_MIN.

    Returns
    -------
    pd.DataFrame
        Index = duracao (min), columns = TR (anos).
    """
    if durations is None:
        durations = DURATIONS_MIN

    result: dict[int, list[float]] = {}
    for tr in tr_values:
        intensities = [
            eq.K * (tr ** eq.a) / ((t + eq.b) ** eq.c)
            for t in durations
        ]
        result[tr] = intensities

    df = pd.DataFrame(result, index=durations)
    df.index.name = "Duracao (min)"
    df.columns.name = "TR (anos)"
    return df
