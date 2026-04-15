"""
Gerador de relatorio PDF tecnico para curvas IDF.

Relatorio autoexplicativo com graficos, tabelas e metodologia detalhada.
Usa fpdf2 para layout e kaleido (via plotly) para exportar graficos.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

from idf import DistributionFitResult, GoFTestResult, IDFEquationResult


class IDFReport(FPDF):
    """PDF tecnico para relatorio IDF."""

    def __init__(self, station_code: str, station_name: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.station_code = station_code
        self.station_name = station_name
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f"Relatorio IDF - Estacao {self.station_code}", align="L")
        self.cell(0, 5, f"Pagina {self.page_no()}/{{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "Gerado por Gerador de Curvas IDF (github.com/viniciusazeved/idf-generator)", align="C")
        self.set_text_color(0, 0, 0)

    def add_title(self, text: str):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 12, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def add_section(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 70, 130)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def add_subsection(self, text: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def add_text(self, text: str, size: int = 9):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_param(self, name: str, value: str, description: str = ""):
        self.set_font("Helvetica", "B", 9)
        self.cell(40, 5, name)
        self.set_font("Helvetica", "", 9)
        self.cell(30, 5, value)
        if description:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, description)
            self.set_text_color(0, 0, 0)
        self.ln(5)

    def add_figure(self, fig: go.Figure, width_mm: int = 180, height_mm: int = 100):
        """Exporta plotly Figure para imagem e insere no PDF."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.write_image(tmp.name, width=900, height=500, scale=2)
            x = (210 - width_mm) / 2
            self.image(tmp.name, x=x, w=width_mm, h=height_mm)
            self.ln(4)

    def add_dataframe(self, df: pd.DataFrame, col_widths: list[int] | None = None):
        """Renderiza DataFrame como tabela no PDF."""
        cols = list(df.columns)
        n_cols = len(cols)
        if col_widths is None:
            available = 190
            col_widths = [available // n_cols] * n_cols

        # Cabecalho
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(230, 235, 245)
        for i, col in enumerate(cols):
            self.cell(col_widths[i], 6, str(col), border=1, fill=True, align="C")
        self.ln()

        # Dados
        self.set_font("Helvetica", "", 7)
        for _, row in df.iterrows():
            for i, col in enumerate(cols):
                val = row[col]
                if isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = str(val)
                self.cell(col_widths[i], 5, text, border=1, align="C")
            self.ln()
        self.ln(3)


def generate_pdf(
    station_code: str,
    station_name: str,
    station_row: object,
    start_year: int,
    end_year: int,
    hydro_year: bool,
    hydro_start_month: int,
    dist_choice: str,
    tr_values: list[int],
    fit_result: DistributionFitResult,
    gof_result: GoFTestResult,
    eq_result: IDFEquationResult | None,
    maxima: pd.Series,
    idf_table: pd.DataFrame,
    precip_by_tr: dict[int, float],
    fig_maxima: go.Figure,
    fig_dist: go.Figure,
    fig_qq: go.Figure,
    fig_idf: go.Figure,
    fig_comparison: go.Figure | None = None,
) -> bytes:
    """Gera relatorio PDF tecnico completo."""
    pdf = IDFReport(station_code, station_name)
    pdf.alias_nb_pages()

    # ===== CAPA / TITULO =====
    pdf.add_page()
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "Relatorio Tecnico", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Curvas Intensidade-Duracao-Frequencia (IDF)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Estacao: {station_code} - {station_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"{station_row.get('City', '?')}, {station_row.get('State', '?')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(30, 70, 130)
    pdf.set_line_width(0.5)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Periodo de analise: {start_year} - {end_year}", align="C", new_x="LMARGIN", new_y="NEXT")

    ano_tipo = "Ano civil (Janeiro a Dezembro)"
    if hydro_year:
        _meses = {1:"Janeiro",2:"Fevereiro",3:"Marco",4:"Abril",5:"Maio",6:"Junho",
                  7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"}
        end_m = (hydro_start_month - 1) if hydro_start_month > 1 else 12
        ano_tipo = f"Ano hidrologico ({_meses[hydro_start_month]} a {_meses[end_m]})"
    pdf.cell(0, 7, ano_tipo, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Distribuicao: {dist_choice}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Tempos de retorno: {', '.join(str(t) for t in tr_values)} anos", align="C", new_x="LMARGIN", new_y="NEXT")

    # ===== 1. DADOS DA ESTACAO =====
    pdf.add_page()
    pdf.add_section("1. Dados da Estacao Pluviometrica")

    pdf.add_text(
        "Os dados de precipitacao diaria foram obtidos do sistema HidroWeb da Agencia Nacional "
        "de Aguas e Saneamento Basico (ANA), por meio da API SOAP disponivel em "
        "telemetriaws1.ana.gov.br. Quando existem dados em dois niveis de consistencia "
        "(bruto e consistido), os dados consistidos (nivel 2) sao priorizados."
    )

    pdf.add_param("Codigo:", station_code)
    pdf.add_param("Nome:", station_name)
    pdf.add_param("Municipio:", str(station_row.get("City", "?")))
    pdf.add_param("Estado:", str(station_row.get("State", "?")))
    pdf.add_param("Latitude:", f"{station_row.get('Latitude', '?')}")
    pdf.add_param("Longitude:", f"{station_row.get('Longitude', '?')}")
    pdf.add_param("Responsavel:", str(station_row.get("Responsible", "?")))
    pdf.add_param("Periodo analisado:", f"{start_year} - {end_year}")
    pdf.add_param("Tipo de ano:", ano_tipo)
    pdf.add_param("Anos com dados:", f"{len(maxima)}")

    # ===== 2. MAXIMOS ANUAIS =====
    pdf.add_section("2. Serie de Maximos Anuais")

    pdf.add_text(
        "A serie de precipitacoes maximas anuais e construida selecionando o maior valor "
        "diario registrado em cada ano do periodo de analise. Dias sem precipitacao (zero) "
        "e dias com falha (NaN) sao excluidos antes da selecao."
    )
    if hydro_year:
        pdf.add_text(
            f"Foi utilizado o ano hidrologico com inicio em {_meses[hydro_start_month]}, "
            "que mantem a estacao chuvosa intacta dentro de um mesmo periodo. "
            "O rotulo do ano corresponde ao ano de termino."
        )

    pdf.add_figure(fig_maxima)

    # Tabela de maximos
    pdf.add_subsection("Tabela de Maximos Anuais")
    max_df = maxima.to_frame("P max (mm)").reset_index()
    max_df.columns = ["Ano", "P max (mm)"]
    # Dividir em 2 colunas se muitos anos
    if len(max_df) > 20:
        half = (len(max_df) + 1) // 2
        left = max_df.iloc[:half].reset_index(drop=True)
        right = max_df.iloc[half:].reset_index(drop=True)
        # Renderizar lado a lado
        pdf.set_font("Helvetica", "", 7)
        for i in range(half):
            pdf.cell(20, 5, str(int(left.iloc[i]["Ano"])), border=1, align="C")
            pdf.cell(25, 5, f"{left.iloc[i]['P max (mm)']:.1f}", border=1, align="C")
            pdf.cell(20, 5, "", border=0)  # espaco
            if i < len(right):
                pdf.cell(20, 5, str(int(right.iloc[i]["Ano"])), border=1, align="C")
                pdf.cell(25, 5, f"{right.iloc[i]['P max (mm)']:.1f}", border=1, align="C")
            pdf.ln()
        pdf.ln(3)
    else:
        pdf.add_dataframe(max_df, col_widths=[30, 40])

    # Estatisticas descritivas
    pdf.add_subsection("Estatisticas Descritivas")
    pdf.add_param("Numero de anos:", f"{len(maxima)}")
    pdf.add_param("Media:", f"{maxima.mean():.1f} mm")
    pdf.add_param("Desvio padrao:", f"{maxima.std():.1f} mm")
    pdf.add_param("Coef. variacao:", f"{(maxima.std()/maxima.mean())*100:.1f}%")
    pdf.add_param("Minimo:", f"{maxima.min():.1f} mm ({int(maxima.idxmin())})")
    pdf.add_param("Maximo:", f"{maxima.max():.1f} mm ({int(maxima.idxmax())})")

    # ===== 3. AJUSTE ESTATISTICO =====
    pdf.add_page()
    pdf.add_section("3. Ajuste de Distribuicao de Probabilidade")

    if dist_choice == "Gumbel":
        pdf.add_text(
            "A distribuicao de Gumbel para maximos (Tipo I de valores extremos) e a mais "
            "utilizada em hidrologia para modelar precipitacoes maximas anuais. Sua funcao "
            "de distribuicao acumulada (CDF) e:\n\n"
            "    F(x) = exp[-exp(-(x - mu) / beta)]\n\n"
            "Os parametros foram estimados pelo Metodo da Maxima Verossimilhanca (MLE):"
        )
    else:
        pdf.add_text(
            "A distribuicao GEV (Generalized Extreme Value) generaliza a Gumbel ao incluir "
            "um parametro de forma (xi) que controla o comportamento da cauda superior. "
            "Sua CDF e:\n\n"
            "    F(x) = exp{-[1 + xi*(x-mu)/beta]^(-1/xi)}\n\n"
            "Casos: xi ~ 0 = Gumbel; xi > 0 = Frechet (cauda pesada); xi < 0 = Weibull (limitada).\n"
            "Os parametros foram estimados pelo Metodo da Maxima Verossimilhanca (MLE):"
        )

    pdf.ln(2)
    for name, val in fit_result.params.items():
        clean_name = name.replace("\u03be", "xi")
        if "loc" in name.lower():
            desc = "Valor central dos maximos anuais (precipitacao tipica, mm)"
        elif "scale" in name.lower():
            desc = "Variabilidade interanual (dispersao dos maximos, mm)"
        elif "shape" in name.lower():
            desc = "Peso da cauda: >0 = eventos extremos mais provaveis"
        else:
            desc = ""
        pdf.add_param(f"  {clean_name}:", f"{val:.4f}", desc)

    # Graficos
    pdf.add_subsection("Histograma e Distribuicao Ajustada")
    pdf.add_figure(fig_dist, height_mm=85)

    pdf.add_subsection("Grafico Quantil-Quantil (QQ-Plot)")
    pdf.add_text(
        "O QQ-Plot compara os quantis teoricos (eixo X) com os quantis amostrais (eixo Y). "
        "Se os pontos seguem a reta 1:1, o modelo esta representando bem os dados. "
        "Desvios nas extremidades indicam que o modelo pode subestimar ou superestimar "
        "os eventos mais raros."
    )
    pdf.add_figure(fig_qq, height_mm=85)

    # Teste de aderencia
    pdf.add_page()
    pdf.add_subsection("Teste de Aderencia de Anderson-Darling")
    pdf.add_text(
        "O teste de Anderson-Darling (AD) avalia se os dados observados sao compativeis "
        "com a distribuicao ajustada. E mais sensivel que o teste de Kolmogorov-Smirnov "
        "para detectar desvios nas caudas, que e onde a modelagem de extremos e mais critica.\n\n"
        "O p-value e calculado por bootstrap parametrico (Monte Carlo, 500 simulacoes), "
        "o que corrige o vies que ocorre quando os parametros da distribuicao sao estimados "
        "a partir dos proprios dados."
    )
    pdf.add_param("  AD statistic:", f"{gof_result.statistic:.4f}",
                  "Distancia modelo-dados (menor = melhor)")
    pdf.add_param("  p-value (MC):", f"{gof_result.p_value:.4f}",
                  "p > 0.05 = ajuste aceitavel")

    if gof_result.p_value >= 0.05:
        pdf.add_text(
            f"Resultado: p-value = {gof_result.p_value:.4f} (> 0.05). "
            "Nao ha evidencia para rejeitar a hipotese de que os dados seguem "
            f"a distribuicao {dist_choice}. O ajuste e considerado aceitavel."
        )
    else:
        pdf.add_text(
            f"Resultado: p-value = {gof_result.p_value:.4f} (< 0.05). "
            "Ha evidencia de que a distribuicao pode nao ser adequada para estes dados. "
            "Recomenda-se testar outra distribuicao ou revisar o periodo de analise."
        )

    # ===== 4. PRECIPITACAO POR TEMPO DE RETORNO =====
    pdf.add_section("4. Precipitacao por Tempo de Retorno")

    pdf.add_text(
        "A precipitacao maxima diaria associada a cada tempo de retorno (TR) e obtida "
        "pela funcao quantil (inversa da CDF) da distribuicao ajustada. O TR representa "
        "o intervalo medio de recorrencia: uma chuva com TR = 100 anos tem probabilidade "
        "de 1% de ser igualada ou superada em qualquer ano."
    )

    tr_df = pd.DataFrame({
        "TR (anos)": list(precip_by_tr.keys()),
        "Prob. anual (%)": [100 / tr for tr in precip_by_tr.keys()],
        "P max diaria (mm)": list(precip_by_tr.values()),
    })
    pdf.add_dataframe(tr_df, col_widths=[35, 45, 45])

    # ===== 5. DESAGREGACAO =====
    pdf.add_section("5. Desagregacao de Chuvas")

    pdf.add_text(
        "A desagregacao permite estimar precipitacoes de curta duracao (5 a 1440 minutos) "
        "a partir da precipitacao maxima diaria, usando coeficientes empiricos da tabela "
        "nacional do antigo DNAEE.\n\n"
        "Etapas da cadeia de desagregacao:\n"
        "1. Correcao dia fixo para dia movel: P(24h) = 1.14 x P(1dia). O fator 1.14 "
        "corrige o fato de que leituras de pluviometros convencionais sao feitas em "
        "horarios fixos (tipicamente 7h), enquanto o maximo real ocorre em uma janela "
        "movel de 24h (Weiss, 1964).\n"
        "2. De 24h para duracoes intermediarias (12h, 10h, 8h, 6h, 1h) usando coeficientes "
        "aplicados a P(24h).\n"
        "3. De 1h para 2h e 30min.\n"
        "4. De 30min para duracoes menores (25, 20, 15, 10, 5 min).\n\n"
        "As alturas de precipitacao (mm) sao convertidas em intensidades (mm/h) "
        "dividindo pela duracao em horas: i = P / (d/60)."
    )

    # ===== 6. CURVAS IDF =====
    pdf.add_page()
    pdf.add_section("6. Curvas IDF")

    pdf.add_text(
        "As curvas IDF (Intensidade-Duracao-Frequencia) representam a relacao entre "
        "a intensidade da chuva, sua duracao e a frequencia de ocorrencia (expressa "
        "pelo tempo de retorno). Sao fundamentais para o dimensionamento de obras "
        "de drenagem, vertedores, bueiros e sistemas de esgotamento pluvial."
    )

    pdf.add_figure(fig_idf)

    # Tabela IDF completa
    pdf.add_subsection("Tabela IDF - Intensidades (mm/h)")
    idf_display = idf_table.copy()
    idf_display.index.name = "Duracao (min)"
    idf_display = idf_display.reset_index()
    # Calcular larguras proporcionais
    n = len(idf_display.columns)
    w = [25] + [int(165 / (n - 1))] * (n - 1)
    pdf.add_dataframe(idf_display, col_widths=w)

    # ===== 7. EQUACAO IDF =====
    if eq_result is not None:
        pdf.add_page()
        pdf.add_section("7. Equacao IDF")

        pdf.add_text(
            "A equacao IDF parametrica permite calcular a intensidade para qualquer "
            "combinacao de duracao e tempo de retorno. A forma adotada e:\n\n"
            "    i = K * TR^a / (t + b)^c\n\n"
            "Onde:\n"
            "  i = intensidade da chuva (mm/h)\n"
            "  TR = tempo de retorno (anos)\n"
            "  t = duracao da chuva (minutos)\n"
            "  K, a, b, c = coeficientes ajustados\n\n"
            "Os coeficientes foram estimados por minimos quadrados nao lineares, "
            "minimizando a diferenca entre as intensidades da tabela IDF e os valores "
            "previstos pela equacao."
        )

        pdf.add_subsection("Coeficientes Ajustados")
        pdf.add_param("  K:", f"{eq_result.K:.4f}",
                      "Fator de escala geral da intensidade")
        pdf.add_param("  a:", f"{eq_result.a:.4f}",
                      "Expoente do TR (sensibilidade ao periodo de retorno)")
        pdf.add_param("  b:", f"{eq_result.b:.4f}",
                      "Offset temporal em minutos (deslocamento da curva)")
        pdf.add_param("  c:", f"{eq_result.c:.4f}",
                      "Expoente de decaimento (quao rapido i cai com a duracao)")

        pdf.add_subsection("Metricas de Qualidade do Ajuste")
        pdf.add_text(
            "R2 (coeficiente de determinacao): mede a proporcao da variancia dos dados "
            "explicada pela equacao. Valores tipicos para chuvas brasileiras ficam entre "
            "0.995 e 0.9999.\n"
            "RMSE (raiz do erro quadratico medio): magnitude tipica do erro em mm/h.\n"
            "MAE (erro medio absoluto): similar ao RMSE, mas menos sensivel a outliers."
        )
        pdf.add_param("  R2:", f"{eq_result.r_squared:.6f}")
        pdf.add_param("  RMSE:", f"{eq_result.rmse:.2f} mm/h")
        pdf.add_param("  MAE:", f"{eq_result.mae:.2f} mm/h")

        if fig_comparison is not None:
            pdf.add_subsection("Comparacao: Tabela IDF vs Equacao Ajustada")
            pdf.add_figure(fig_comparison)

    # ===== 8. METODOLOGIA =====
    pdf.add_page()
    pdf.add_section("8. Metodologia")

    pdf.add_subsection("8.1. Fonte de Dados")
    pdf.add_text(
        "Precipitacao diaria da rede de monitoramento da ANA, obtida via API SOAP "
        "(telemetriaws1.ana.gov.br). O catalogo de estacoes utilizado e o ANAF "
        "(ANA Filtered), compilado por Carvalho & Braga (2020). Dados consistidos "
        "(nivel 2) sao priorizados sobre dados brutos quando ambos existem para a "
        "mesma data."
    )

    pdf.add_subsection("8.2. Extracao de Maximos Anuais")
    pdf.add_text(
        "Para cada ano do periodo selecionado, extrai-se o maior valor diario de "
        "precipitacao registrado. Dias sem chuva (P = 0) e dias com falha sao excluidos. "
        "A serie resultante alimenta o ajuste da distribuicao de probabilidade."
    )

    pdf.add_subsection("8.3. Ajuste de Distribuicao")
    pdf.add_text(
        "A distribuicao de probabilidade e ajustada aos maximos anuais pelo Metodo da "
        "Maxima Verossimilhanca (MLE). O MLE encontra os parametros que maximizam a "
        "probabilidade de observar os dados coletados, dado o modelo teorico. E o metodo "
        "recomendado pela WMO e por Naghettini & Pinto (2007) para analise de extremos.\n\n"
        "Distribuicao de Gumbel: possui dois parametros (locacao mu e escala beta). "
        "E o modelo mais simples e mais utilizado para chuvas maximas anuais no Brasil.\n\n"
        "Distribuicao GEV: generaliza a Gumbel com um terceiro parametro (forma xi) que "
        "controla o peso da cauda superior. Requer amostras maiores (minimo 20 anos)."
    )

    pdf.add_subsection("8.4. Teste de Aderencia")
    pdf.add_text(
        "O teste de Anderson-Darling (AD) verifica se os dados sao compativeis com a "
        "distribuicao ajustada. A estatistica AD mede a distancia ponderada entre a "
        "distribuicao empirica e a teorica, com maior peso nas caudas. O p-value e "
        "calculado por bootstrap parametrico Monte Carlo (500 simulacoes), que corrige "
        "o vies do teste classico quando os parametros sao estimados dos mesmos dados."
    )

    pdf.add_subsection("8.5. Desagregacao de Chuvas")
    pdf.add_text(
        "Utiliza a tabela nacional de desagregacao do antigo DNAEE, reproduzida em "
        "CETESB (1980) e Tucci (2009). O fator 1.14 de correcao dia fixo para dia movel "
        "baseia-se em Weiss (1964). A cadeia de desagregacao e sequencial: cada duracao "
        "e obtida a partir de uma duracao de referencia, nao diretamente da precipitacao diaria."
    )

    pdf.add_subsection("8.6. Equacao IDF")
    pdf.add_text(
        "A equacao i = K * TR^a / (t + b)^c e ajustada por minimos quadrados nao lineares "
        "(Levenberg-Marquardt) com restricoes fisicas nos parametros: K > 0, 0 < a < 1, "
        "0 < b < 100, 0 < c < 2. Essas restricoes garantem comportamento fisicamente "
        "coerente (intensidade positiva, decrescente com a duracao, crescente com o TR)."
    )

    # ===== 9. REFERENCIAS =====
    pdf.add_section("9. Referencias")
    refs = [
        "CETESB. (1980). Drenagem Urbana - Manual de Projeto. Sao Paulo: CETESB/ASCETESB.",
        "DNAEE. Tabela nacional de desagregacao de chuvas. Reproduzida em CETESB (1980) e Tucci (2009).",
        "Weiss, L. (1964). Ratio of true to fixed-interval maximum rainfall. Journal of the Hydraulic Division, 90(1), 77-82.",
        "Koutsoyiannis, D., Kozonis, D., & Manetas, A. (1998). A mathematical framework for studying "
        "rainfall intensity-duration-frequency relationships. Journal of Hydrology, 206(1-2), 118-135.",
        "Naghettini, M., & Pinto, E. J. A. (2007). Hidrologia Estatistica. Belo Horizonte: CPRM.",
        "Back, A. J. (2002). Relacoes entre precipitacoes intensas de diferentes duracoes para "
        "desagregacao da chuva diaria em Santa Catarina. Rev. Bras. Eng. Agric. Ambiental, 6(2), 258-265.",
        "Carvalho, W. A., & Braga, A. S. (2020). HydroBR: A Python package for hydrometeorological "
        "data acquisition from Brazilian databases. Zenodo. doi:10.5281/zenodo.3755065",
    ]
    for ref in refs:
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 4, f"- {ref}")
        pdf.ln(1)

    return bytes(pdf.output())
