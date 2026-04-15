"""
Gerador de relatorio PDF com estatisticas, graficos e metodologia.

Usa fpdf2 para layout e kaleido (via plotly) para exportar graficos.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import plotly.graph_objects as go
from fpdf import FPDF

from idf import GoFTestResult, IDFEquationResult, DistributionFitResult

_LOGO_PATH = Path(__file__).parent / "logo_lapla.png"


class IDFReport(FPDF):
    """PDF customizado com cabecalho e rodape LAPLA."""

    def __init__(self, station_code: str, station_name: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.station_code = station_code
        self.station_name = station_name
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if _LOGO_PATH.exists():
            self.image(str(_LOGO_PATH), x=10, y=8, w=18)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 5, "LAPLA - Laboratorio de Planejamento Ambiental", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.cell(0, 4, "FECFAU / Unicamp", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.cell(0, 10, f"Estacao {self.station_code} - {self.station_name} | Pagina {self.page_no()}/{{nb}}", align="C")

    def _add_title(self, text: str):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def _add_section(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 80, 140)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def _add_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def _add_param(self, name: str, value: str, description: str = ""):
        self.set_font("Helvetica", "B", 9)
        self.cell(35, 5, name)
        self.set_font("Helvetica", "", 9)
        self.cell(30, 5, value)
        if description:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, description)
            self.set_text_color(0, 0, 0)
        self.ln(5)

    def _add_figure(self, fig: go.Figure, width_mm: int = 180, height_mm: int = 100):
        """Exporta plotly Figure para imagem e insere no PDF."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.write_image(tmp.name, width=900, height=500, scale=2)
            x = (210 - width_mm) / 2  # centralizar em A4
            self.image(tmp.name, x=x, w=width_mm, h=height_mm)
            self.ln(4)


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
    fig_maxima: go.Figure,
    fig_dist: go.Figure,
    fig_qq: go.Figure,
    fig_idf: go.Figure,
) -> bytes:
    """
    Gera relatorio PDF completo e retorna como bytes.
    """
    pdf = IDFReport(station_code, station_name)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Titulo
    pdf._add_title(f"Relatorio IDF - {station_name}")
    pdf._add_text(f"Estacao: {station_code} - {station_name}")

    # Info da estacao
    pdf._add_section("1. Dados da Estacao")
    pdf._add_param("Codigo:", station_code)
    pdf._add_param("Nome:", station_name)
    pdf._add_param("Municipio:", str(station_row.get("City", "?")))
    pdf._add_param("Estado:", str(station_row.get("State", "?")))
    pdf._add_param("Latitude:", str(station_row.get("Latitude", "?")))
    pdf._add_param("Longitude:", str(station_row.get("Longitude", "?")))
    pdf._add_param("Responsavel:", str(station_row.get("Responsible", "?")))
    pdf._add_param("Periodo analisado:", f"{start_year} - {end_year}")

    ano_tipo = "Civil (Jan-Dez)"
    if hydro_year:
        _meses = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
                  7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
        end_m = (hydro_start_month - 1) if hydro_start_month > 1 else 12
        ano_tipo = f"Hidrologico ({_meses[hydro_start_month]}-{_meses[end_m]})"
    pdf._add_param("Ano:", ano_tipo)
    pdf._add_param("Distribuicao:", dist_choice)
    pdf._add_param("Tempos de retorno:", ", ".join(str(t) for t in tr_values) + " anos")

    # Maximos anuais
    pdf._add_section("2. Maximos Anuais")
    pdf._add_figure(fig_maxima)

    # Ajuste estatistico
    pdf.add_page()
    pdf._add_section("3. Ajuste Estatistico")
    pdf._add_text(f"Distribuicao ajustada: {fit_result.name}")

    for name, val in fit_result.params.items():
        clean_name = name.replace("\u03be", "xi")
        if "loc" in name.lower():
            desc = "Precipitacao maxima tipica (mm)"
        elif "scale" in name.lower():
            desc = "Variabilidade interanual dos maximos (mm)"
        elif "shape" in name.lower():
            desc = "Peso da cauda superior (eventos extremos)"
        else:
            desc = ""
        pdf._add_param(f"{clean_name}:", f"{val:.4f}", desc)

    pdf._add_param("AD statistic:", f"{gof_result.statistic:.4f}",
                    "Distancia entre modelo e dados (menor = melhor)")
    pdf._add_param("p-value (MC):", f"{gof_result.p_value:.4f}",
                    "p > 0.05 = ajuste aceitavel")

    pdf._add_figure(fig_dist, height_mm=80)
    pdf._add_figure(fig_qq, height_mm=80)

    # Curvas IDF
    pdf.add_page()
    pdf._add_section("4. Curvas IDF")
    pdf._add_figure(fig_idf)

    # Equacao IDF
    if eq_result is not None:
        pdf._add_section("5. Equacao IDF")
        pdf._add_text("i = K * TR^a / (t + b)^c")
        pdf._add_text("Onde: i = intensidade (mm/h), TR = tempo de retorno (anos), t = duracao (min)")
        pdf._add_param("K:", f"{eq_result.K:.4f}", "Fator de escala")
        pdf._add_param("a:", f"{eq_result.a:.4f}", "Expoente do tempo de retorno")
        pdf._add_param("b:", f"{eq_result.b:.4f}", "Offset temporal (min)")
        pdf._add_param("c:", f"{eq_result.c:.4f}", "Expoente de decaimento com duracao")
        pdf._add_param("R2:", f"{eq_result.r_squared:.6f}", "Coeficiente de determinacao")
        pdf._add_param("RMSE:", f"{eq_result.rmse:.2f} mm/h", "Erro quadratico medio")
        pdf._add_param("MAE:", f"{eq_result.mae:.2f} mm/h", "Erro absoluto medio")

    # Metodologia resumida
    pdf.add_page()
    pdf._add_section("6. Metodologia Resumida")
    pdf._add_text(
        "1. Dados de precipitacao diaria obtidos do HidroWeb/ANA via API SOAP. "
        "Dados consistidos (nivel 2) priorizados sobre dados brutos.\n\n"
        "2. Maximos anuais extraidos do periodo selecionado.\n\n"
        f"3. Distribuicao {dist_choice} ajustada por Maxima Verossimilhanca (MLE). "
        "Teste de aderencia Anderson-Darling com p-value por bootstrap parametrico Monte Carlo.\n\n"
        "4. Desagregacao de chuvas pela tabela nacional DNAEE. "
        "Fator 1.14 para correcao dia fixo -> dia movel (Weiss, 1964).\n\n"
        "5. Equacao IDF (i = K*TR^a/(t+b)^c) ajustada por minimos quadrados nao lineares.\n\n"
        "Referencias: DNAEE/CETESB (1980); Naghettini & Pinto (2007); "
        "Koutsoyiannis et al. (1998); Weiss (1964)."
    )

    return bytes(pdf.output())
