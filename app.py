"""
Gerador de Curvas IDF - Streamlit App

Gera curvas Intensidade-Duracao-Frequencia a partir de dados
pluviometricos da ANA (Agencia Nacional de Aguas).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ana_client import ANAConnectionError, fetch_daily_precipitation
from idf import (
    IDFEquationResult,
    compute_annual_maxima,
    compute_idf_table,
    fit_gev,
    fit_gumbel,
    fit_idf_equation,
    idf_equation_predict,
    ks_test,
    qq_data,
    return_period_precipitation,
)
from plots import (
    plot_annual_maxima,
    plot_annual_totals,
    plot_availability,
    plot_daily_timeseries,
    plot_distribution_fit,
    plot_idf_comparison,
    plot_idf_curves,
    plot_qq,
)
from stations import get_cities, get_states, get_stations, load_catalog

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gerador de Curvas IDF",
    page_icon=":rain_cloud:",
    layout="wide",
)

DEFAULT_TRS = [2, 5, 10, 25, 50, 100]


# ---------------------------------------------------------------------------
# Metodologia
# ---------------------------------------------------------------------------
def _render_methodology():
    """Renderiza a aba de metodologia com fundamentacao tecnica completa."""

    st.header("Metodologia")

    st.markdown(
        "Esta secao descreve, em detalhe, cada etapa do procedimento adotado "
        "para a construcao das curvas IDF a partir de dados pluviometricos da ANA."
    )

    # --- 1. Fonte de dados ---
    st.subheader("1. Fonte de Dados")
    st.markdown("""
Os dados de precipitacao diaria sao obtidos do sistema **HidroWeb** da
Agencia Nacional de Aguas e Saneamento Basico (ANA), por meio da API
SOAP disponivel em `telemetriaws1.ana.gov.br`.

O catalogo de estacoes utilizado e o **ANAF** (ANA Filtered), que contem
apenas estacoes com dados efetivamente registrados. O ANAF foi compilado
por Carvalho & Braga (2020) e esta disponivel em
[doi:10.5281/zenodo.3755065](https://doi.org/10.5281/zenodo.3755065).

Quando uma estacao possui dados em dois niveis de consistencia (bruto e
consistido), o app prioriza os **dados consistidos** (nivel 2). Em datas
onde ambos os niveis existem, o dado consistido substitui o bruto.
    """)

    # --- 2. Maximos anuais ---
    st.subheader("2. Extracao de Maximos Anuais")
    st.markdown("""
A serie de precipitacoes maximas anuais e extraida selecionando o maior
valor diario registrado em cada ano do periodo de analise.

O app oferece duas opcoes de agrupamento anual:

- **Ano civil (Jan-Dez):** cada ano compreende de 1 de janeiro a 31 de
  dezembro.
- **Ano hidrologico (Out-Set):** o ano inicia em 1 de outubro e termina
  em 30 de setembro do ano seguinte. Rotulado pelo ano final (ex.:
  Out/2000 a Set/2001 = ano 2001). Este agrupamento e recomendado no
  Brasil pois mantem a estacao chuvosa intacta dentro de um mesmo
  periodo.
    """)

    # --- 3. Distribuicao de Gumbel ---
    st.subheader("3. Distribuicao de Gumbel (Tipo I - Maximos)")
    st.markdown("""
A distribuicao de Gumbel para maximos (tambem chamada de distribuicao de
valores extremos Tipo I) e a mais utilizada em hidrologia para modelar
precipitacoes maximas anuais. Sua funcao de densidade de probabilidade
(PDF) e:
    """)
    st.latex(
        r"f(x) = \frac{1}{\beta} \exp\!\left[-\frac{x - \mu}{\beta}"
        r"- \exp\!\left(-\frac{x - \mu}{\beta}\right)\right]"
    )
    st.markdown("E sua funcao de distribuicao acumulada (CDF):")
    st.latex(
        r"F(x) = \exp\!\left[-\exp\!\left(-\frac{x - \mu}{\beta}\right)\right]"
    )
    st.markdown(r"""
Onde:
- $\mu$ = parametro de locacao (loc) - relacionado a moda da distribuicao
- $\beta$ = parametro de escala (scale) - controla a dispersao

Os parametros sao estimados pelo **Metodo da Maxima Verossimilhanca**
(MLE), implementado via `scipy.stats.gumbel_r.fit()`.

A precipitacao associada a um tempo de retorno $TR$ e obtida pela funcao
quantil (inversa da CDF):
    """)
    st.latex(
        r"P_{TR} = \mu - \beta \cdot \ln\!\left[-\ln\!\left(1 - \frac{1}{TR}\right)\right]"
    )

    # --- 4. Distribuicao GEV ---
    st.subheader("4. Distribuicao GEV (Generalized Extreme Value)")
    st.markdown("""
A distribuicao GEV generaliza a Gumbel ao incluir um parametro de forma
$\\xi$ (shape) que controla o comportamento da cauda superior:
    """)
    st.latex(
        r"F(x) = \exp\!\left\{-\left[1 + \xi\!\left(\frac{x-\mu}{\beta}\right)"
        r"\right]^{-1/\xi}\right\}"
    )
    st.markdown(r"""
Casos especiais:
- $\xi = 0$: **Gumbel** (Tipo I) - cauda exponencial
- $\xi > 0$: **Frechet** (Tipo II) - cauda pesada (eventos extremos mais provaveis)
- $\xi < 0$: **Weibull** (Tipo III) - cauda limitada superiormente

A GEV e mais flexivel que a Gumbel e pode se ajustar melhor quando a
amostra possui eventos extremos outliers. Porem, requer amostras maiores
para uma estimacao confiavel do parametro de forma.
    """)

    # --- 5. Teste KS ---
    st.subheader("5. Teste de Aderencia de Kolmogorov-Smirnov")
    st.markdown("""
O teste de Kolmogorov-Smirnov (KS) avalia se a amostra provem da
distribuicao teorica ajustada. A estatistica de teste e:
    """)
    st.latex(r"D_n = \sup_x |F_n(x) - F(x)|")
    st.markdown(r"""
Onde $F_n(x)$ e a funcao de distribuicao empirica e $F(x)$ e a CDF
teorica. A hipotese nula $H_0$ e que os dados seguem a distribuicao.

Interpretacao do **p-value**:
- **p > 0.05:** nao ha evidencia para rejeitar $H_0$ — o ajuste e aceitavel
- **p < 0.05:** o ajuste pode ser inadequado — considere outra distribuicao

O **QQ-Plot** complementa o teste KS de forma visual: pontos proximos da
reta 1:1 indicam bom ajuste; desvios sistematicos nas caudas indicam
problemas na modelagem de eventos extremos.
    """)

    # --- 6. Fator 1.14 ---
    st.subheader("6. Correcao Dia Fixo para Dia Movel (Fator 1.14)")
    st.markdown("""
As leituras de pluviometros convencionais sao feitas em horarios fixos
(tipicamente as 7h). Assim, a "precipitacao maxima de 1 dia" pode nao
coincidir com o maximo real em uma janela movel de 24 horas.

Estudos estatisticos demonstraram que a precipitacao maxima em 24 horas
(janela movel) e, em media, **14% superior** a precipitacao maxima de
1 dia (janela fixa). Portanto:
    """)
    st.latex(r"P_{24h} = 1{,}14 \times P_{1\,dia}")
    st.markdown("""
Este fator e aplicado automaticamente na etapa de desagregacao, antes
da aplicacao dos demais coeficientes. E uma pratica padrao na hidrologia
brasileira, adotada pelo DAEE, CETESB e diversos manuais de drenagem.
    """)

    # --- 7. Desagregacao ---
    st.subheader("7. Desagregacao de Chuvas - Tabela DAEE/CETESB (1980)")
    st.markdown("""
A desagregacao permite estimar precipitacoes de curta duracao (5 a
1440 minutos) a partir da precipitacao maxima diaria, utilizando
coeficientes empiricos.

A tabela de referencia e a do **DAEE/CETESB (1980)**, publicada no
*Manual de Drenagem Urbana* da CETESB. A desagregacao segue uma
**cadeia sequencial** (nao sao fatores independentes):
    """)

    st.markdown("**Etapa 1:** Dia fixo para 24h")
    st.latex(r"P_{1440\min} = 1{,}14 \times P_{1\,dia}")

    st.markdown("**Etapa 2:** De 24h (1440 min) para duracoes intermediarias")

    daee_1440 = pd.DataFrame({
        "Duracao origem": ["24h (1440 min)"] * 5,
        "Duracao destino": ["12h (720 min)", "10h (600 min)", "8h (480 min)",
                            "6h (360 min)", "1h (60 min)"],
        "Coeficiente": [0.85, 0.82, 0.78, 0.72, 0.51],
    })
    st.dataframe(daee_1440, use_container_width=True, hide_index=True)

    st.markdown("**Etapa 3:** De 1h (60 min) para 2h e 30 min")

    daee_60 = pd.DataFrame({
        "Duracao origem": ["1h (60 min)"] * 2,
        "Duracao destino": ["2h (120 min)", "30 min"],
        "Coeficiente": [1.27, 0.74],
    })
    st.dataframe(daee_60, use_container_width=True, hide_index=True)

    st.markdown("**Etapa 4:** De 30 min para duracoes menores")

    daee_30 = pd.DataFrame({
        "Duracao origem": ["30 min"] * 5,
        "Duracao destino": ["25 min", "20 min", "15 min", "10 min", "5 min"],
        "Coeficiente": [0.91, 0.81, 0.70, 0.54, 0.34],
    })
    st.dataframe(daee_30, use_container_width=True, hide_index=True)

    st.markdown("""
Ao final, as alturas de precipitacao (mm) sao convertidas em
**intensidades** (mm/h) dividindo pela duracao em horas:
    """)
    st.latex(r"i = \frac{P_d}{d/60}")
    st.markdown(r"Onde $P_d$ e a precipitacao (mm) para a duracao $d$ (minutos).")

    # --- 8. Equacao IDF ---
    st.subheader("8. Equacao IDF")
    st.markdown("""
A equacao IDF parametrica permite calcular a intensidade para qualquer
combinacao de duracao e tempo de retorno. A forma adotada e a classica
brasileira (tambem conhecida como equacao de chuvas intensas):
    """)
    st.latex(r"i = \frac{K \cdot TR^{a}}{(t + b)^{c}}")
    st.markdown(r"""
Onde:
- $i$ = intensidade da chuva (mm/h)
- $TR$ = tempo de retorno (anos)
- $t$ = duracao da chuva (minutos)
- $K$, $a$, $b$, $c$ = parametros a ajustar

Os parametros sao estimados pelo **metodo dos minimos quadrados nao
lineares** (`scipy.optimize.curve_fit`), minimizando a soma dos quadrados
dos residuos entre as intensidades da tabela IDF e os valores preditos
pela equacao.

**Restricoes fisicas (bounds) aplicadas ao ajuste:**

| Parametro | Minimo | Maximo | Significado fisico |
|-----------|--------|--------|--------------------|
| $K$ | 0 | $+\infty$ | Fator de escala (sempre positivo) |
| $a$ | 0 | 1,0 | Expoente do TR (retornos decrescentes) |
| $b$ | 0 | 100 | Offset temporal em minutos |
| $c$ | 0 | 2,0 | Expoente de decaimento com a duracao |

Essas restricoes evitam parametros fisicamente impossiveis (como
expoentes negativos ou valores de TR elevados a potencias maiores que 1)
e garantem a convergencia numerica do ajuste.
    """)

    # --- 9. Metricas ---
    st.subheader("9. Metricas de Qualidade do Ajuste")
    st.markdown("**Coeficiente de determinacao (R²):**")
    st.latex(r"R^2 = 1 - \frac{\sum(i_{obs} - i_{pred})^2}{\sum(i_{obs} - \bar{i}_{obs})^2}")
    st.markdown("**Raiz do Erro Quadratico Medio (RMSE):**")
    st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{k=1}^{n}(i_{obs,k} - i_{pred,k})^2}")
    st.markdown("**Erro Medio Absoluto (MAE):**")
    st.latex(r"MAE = \frac{1}{n}\sum_{k=1}^{n}|i_{obs,k} - i_{pred,k}|")
    st.markdown("""
R² proximo de 1 indica que a equacao reproduz bem a tabela IDF.
Valores tipicos para chuvas brasileiras ficam entre 0,995 e 0,9999.
RMSE e MAE tem unidade de mm/h e permitem avaliar a magnitude
absoluta dos erros.
    """)

    # --- 10. Referencias ---
    st.subheader("10. Referencias")
    st.markdown("""
- CETESB. (1980). *Drenagem Urbana - Manual de Projeto*. Sao Paulo: CETESB/ASCETESB.
- DAEE/CETESB. (1980). *Tabela de coeficientes de desagregacao de chuvas*. Sao Paulo.
- Koutsoyiannis, D., Kozonis, D., & Manetas, A. (1998). A mathematical framework
  for studying rainfall intensity-duration-frequency relationships.
  *Journal of Hydrology*, 206(1-2), 118-135.
- Naghettini, M., & Pinto, E. J. A. (2007). *Hidrologia Estatistica*. Belo Horizonte: CPRM.
- Back, A. J. (2002). Relacoes entre precipitacoes intensas de diferentes duracoes
  para desagregacao da chuva diaria em Santa Catarina. *Revista Brasileira de
  Engenharia Agricola e Ambiental*, 6(2), 258-265.
- Carvalho, W. A., & Braga, A. S. (2020). HydroBR: A Python package for hydrometeorological
  data acquisition from Brazilian databases. Zenodo. https://doi.org/10.5281/zenodo.3755065
    """)


# ---------------------------------------------------------------------------
# Cache de download
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Baixando dados da ANA... Isso pode levar 30-60 segundos.")
def _download_precipitation(station_code: str) -> pd.Series:
    return fetch_daily_precipitation(station_code)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Gerador de Curvas IDF")

# 1. Selecao de estacao
st.sidebar.header("1. Selecao de Estacao")

catalog = load_catalog()
states = get_states(catalog)

selected_state = st.sidebar.selectbox("Estado", states, index=None, placeholder="Selecione...")

selected_city = None
selected_station_row = None

if selected_state:
    cities = get_cities(catalog, selected_state)
    selected_city = st.sidebar.selectbox("Cidade", cities, index=None, placeholder="Selecione...")

if selected_city:
    stations_df = get_stations(catalog, selected_state, selected_city)
    station_options = {
        f"{row['Code']} - {row['Name']} ({row.get('NYD', '?')} anos)": row["Code"]
        for _, row in stations_df.iterrows()
    }
    selected_label = st.sidebar.selectbox(
        "Estacao",
        list(station_options.keys()),
        index=None,
        placeholder="Selecione...",
    )
    if selected_label:
        selected_code = station_options[selected_label]
        selected_station_row = stations_df[stations_df["Code"] == selected_code].iloc[0]

load_btn = st.sidebar.button("Carregar Dados", type="primary", disabled=selected_station_row is None)

# 2. Configuracao
st.sidebar.header("2. Configuracao")

col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_year = st.number_input("Ano Inicial", value=1970, min_value=1900, max_value=2030)
with col_end:
    end_year = st.number_input("Ano Final", value=2023, min_value=1900, max_value=2030)

hydro_year = st.sidebar.checkbox("Usar Ano Hidrologico (Out-Set)", value=False)
dist_choice = st.sidebar.radio("Distribuicao", ["Gumbel", "GEV"], index=0)
tr_values = st.sidebar.multiselect(
    "Tempos de Retorno (anos)",
    options=[2, 5, 10, 25, 50, 100, 200, 500, 1000],
    default=DEFAULT_TRS,
)

if not tr_values:
    tr_values = DEFAULT_TRS

# Creditos
st.sidebar.divider()
st.sidebar.markdown(
    "**Desenvolvido por**\n\n"
    "Vinicius de Azevedo Silva\n\n"
    "Engenheiro Ambiental\n"
    "Doutorando em Recursos Hidricos,\n"
    "Energeticos e Ambientais\n"
    "FECFAU / Unicamp\n\n"
    "[GitHub](https://github.com/viniciusazeved) · "
    "[Repositorio](https://github.com/viniciusazeved/idf-generator)",
    unsafe_allow_html=False,
)


# ---------------------------------------------------------------------------
# Logica de carregamento
# ---------------------------------------------------------------------------
if load_btn and selected_station_row is not None:
    st.session_state["station_code"] = selected_station_row["Code"]
    st.session_state["station_name"] = selected_station_row["Name"]
    st.session_state["station_row"] = selected_station_row
    try:
        data = _download_precipitation(selected_station_row["Code"])
        st.session_state["precipitation_data"] = data
    except ANAConnectionError as e:
        st.error(f"Erro de conexao com a ANA: {e}")
        st.stop()
    except ValueError as e:
        st.warning(str(e))
        st.stop()


# ---------------------------------------------------------------------------
# Area principal
# ---------------------------------------------------------------------------
if "precipitation_data" not in st.session_state:
    st.title("Gerador de Curvas IDF")
    tab_home, tab_metodo = st.tabs(["Inicio", "Metodologia"])
    with tab_home:
        st.markdown(
            "Selecione uma estacao pluviometrica no menu lateral e clique em "
            "**Carregar Dados** para comecar."
        )
        st.info(
            "Este app gera curvas IDF (Intensidade-Duracao-Frequencia) a partir de dados "
            "historicos de precipitacao da ANA, usando desagregacao DAEE/CETESB (1980) e "
            "ajuste estatistico (Gumbel ou GEV)."
        )
    with tab_metodo:
        _render_methodology()
    st.stop()


# Dados carregados — prosseguir
series = st.session_state["precipitation_data"]
station_name = st.session_state["station_name"]
station_code = st.session_state["station_code"]
station_row = st.session_state["station_row"]

st.title(f"Curvas IDF - {station_name}")

tab_analise, tab_metodo = st.tabs(["Analise", "Metodologia"])

with tab_metodo:
    _render_methodology()

with tab_analise:

    # Info da estacao
    info_cols = st.columns(4)
    info_cols[0].metric("Codigo", station_code)
    info_cols[1].metric("Latitude", f"{station_row.get('Latitude', '?')}")
    info_cols[2].metric("Longitude", f"{station_row.get('Longitude', '?')}")
    info_cols[3].metric("Responsavel", station_row.get("Responsible", "?"))

    # ---- Secao 1: Dados e Disponibilidade ----
    st.header("1. Dados e Disponibilidade")

    valid_data = series.dropna()
    total_days = len(series)
    valid_days = len(valid_data)
    missing_pct = 100 * (1 - valid_days / total_days) if total_days > 0 else 0
    years_span = (series.index.max() - series.index.min()).days / 365.25

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Periodo", f"{series.index.min().year} - {series.index.max().year}")
    m2.metric("Anos de dados", f"{years_span:.1f}")
    m3.metric("Dias validos", f"{valid_days:,}")
    m4.metric("Falhas", f"{missing_pct:.1f}%")

    st.plotly_chart(plot_daily_timeseries(series, station_name), use_container_width=True)

    with st.expander("Totais Anuais"):
        st.plotly_chart(plot_annual_totals(series, station_name), use_container_width=True)

    with st.expander("Mapa de Disponibilidade (Ano x Mes)"):
        st.plotly_chart(plot_availability(series), use_container_width=True)

    # ---- Secao 2: Maximos Anuais ----
    st.header("2. Maximos Anuais")

    maxima = compute_annual_maxima(series, start_year, end_year, hydro_year)

    if len(maxima) < 5:
        st.error(
            f"Apenas {len(maxima)} anos com dados no periodo {start_year}-{end_year}. "
            "Sao necessarios pelo menos 5 anos para o ajuste estatistico. "
            "Ajuste o periodo de analise."
        )
        st.stop()

    st.plotly_chart(plot_annual_maxima(maxima), use_container_width=True)

    with st.expander("Tabela de Maximos Anuais"):
        st.dataframe(
            maxima.to_frame("P max (mm)").style.format("{:.1f}"),
            use_container_width=True,
        )

    # ---- Secao 3: Ajuste de Distribuicao ----
    st.header("3. Ajuste de Distribuicao")

    if dist_choice == "Gumbel":
        fit_result = fit_gumbel(maxima)
    else:
        fit_result = fit_gev(maxima)

    ks_result = ks_test(maxima, fit_result)
    theoretical, sample = qq_data(maxima, fit_result)

    col_fit1, col_fit2 = st.columns(2)
    with col_fit1:
        st.plotly_chart(plot_distribution_fit(maxima, fit_result), use_container_width=True)
    with col_fit2:
        st.plotly_chart(plot_qq(theoretical, sample), use_container_width=True)

    param_cols = st.columns(len(fit_result.params) + 2)
    for i, (name, val) in enumerate(fit_result.params.items()):
        param_cols[i].metric(name, f"{val:.4f}")
    param_cols[-2].metric("KS statistic", f"{ks_result.statistic:.4f}")
    param_cols[-1].metric("KS p-value", f"{ks_result.p_value:.4f}")

    if ks_result.p_value < 0.05:
        st.warning(
            f"p-value do teste KS = {ks_result.p_value:.4f} (< 0.05). "
            "A distribuicao pode nao ser adequada para estes dados. "
            "Considere testar outra distribuicao ou revisar o periodo."
        )

    # ---- Secao 4: Precipitacao por Tempo de Retorno ----
    st.header("4. Precipitacao por Tempo de Retorno")

    precip_by_tr = return_period_precipitation(fit_result, sorted(tr_values))

    df_tr = pd.DataFrame(
        {"TR (anos)": list(precip_by_tr.keys()), "P 24h (mm)": list(precip_by_tr.values())}
    )
    st.dataframe(
        df_tr.style.format({"P 24h (mm)": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Nota: P 24h inclui fator 1.14 (correcao dia fixo -> movel) aplicado na desagregacao.")

    # ---- Secao 5: Curvas IDF ----
    st.header("5. Curvas IDF")

    idf_table = compute_idf_table(precip_by_tr)

    st.plotly_chart(plot_idf_curves(idf_table), use_container_width=True)

    with st.expander("Tabela IDF completa (intensidade em mm/h)"):
        st.dataframe(
            idf_table.style.format("{:.2f}"),
            use_container_width=True,
        )

    # ---- Secao 6: Equacao IDF ----
    st.header("6. Equacao IDF")

    st.latex(r"i = \frac{K \cdot TR^{a}}{(t + b)^{c}}")
    st.caption("Onde: i = intensidade (mm/h), TR = tempo de retorno (anos), t = duracao (min)")

    try:
        eq_result = fit_idf_equation(idf_table)

        eq_cols = st.columns(7)
        eq_cols[0].metric("K", f"{eq_result.K:.2f}")
        eq_cols[1].metric("a", f"{eq_result.a:.4f}")
        eq_cols[2].metric("b", f"{eq_result.b:.4f}")
        eq_cols[3].metric("c", f"{eq_result.c:.4f}")
        eq_cols[4].metric("R\u00b2", f"{eq_result.r_squared:.6f}")
        eq_cols[5].metric("RMSE", f"{eq_result.rmse:.2f} mm/h")
        eq_cols[6].metric("MAE", f"{eq_result.mae:.2f} mm/h")

        st.plotly_chart(plot_idf_comparison(idf_table, eq_result), use_container_width=True)

        st.code(
            f"i = {eq_result.K:.4f} * TR^{eq_result.a:.4f} / (t + {eq_result.b:.4f})^{eq_result.c:.4f}",
            language=None,
        )

    except RuntimeError:
        st.error(
            "O ajuste da equacao IDF nao convergiu. "
            "Verifique a qualidade dos dados ou tente um periodo diferente."
        )
        eq_result = None

    # ---- Secao 7: Exportar ----
    st.header("7. Exportar")

    export_cols = st.columns(3)

    csv_idf = idf_table.to_csv()
    export_cols[0].download_button(
        "Tabela IDF (CSV)",
        csv_idf,
        file_name=f"idf_{station_code}.csv",
        mime="text/csv",
    )

    if eq_result is not None:
        eq_text = (
            f"Estacao: {station_code} - {station_name}\n"
            f"Periodo: {start_year}-{end_year}\n"
            f"Distribuicao: {dist_choice}\n"
            f"Ano Hidrologico: {'Sim' if hydro_year else 'Nao'}\n\n"
            f"Equacao IDF: i = K * TR^a / (t + b)^c\n\n"
            f"K = {eq_result.K:.6f}\n"
            f"a = {eq_result.a:.6f}\n"
            f"b = {eq_result.b:.6f}\n"
            f"c = {eq_result.c:.6f}\n\n"
            f"R\u00b2 = {eq_result.r_squared:.6f}\n"
            f"RMSE = {eq_result.rmse:.4f} mm/h\n"
            f"MAE = {eq_result.mae:.4f} mm/h\n"
        )
        export_cols[1].download_button(
            "Coeficientes (TXT)",
            eq_text,
            file_name=f"idf_equacao_{station_code}.txt",
            mime="text/plain",
        )

    csv_tr = df_tr.to_csv(index=False)
    export_cols[2].download_button(
        "Precipitacao por TR (CSV)",
        csv_tr,
        file_name=f"precipitacao_tr_{station_code}.csv",
        mime="text/csv",
    )
