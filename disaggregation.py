"""
Coeficientes de desagregacao de chuva DAEE/CETESB (1980).

Referencia: CETESB (1980). Drenagem Urbana - Manual de Projeto.
Tabela de relacoes entre chuvas de diferentes duracoes.
"""

# Coeficientes de desagregacao: (duracao_origem_min, duracao_destino_min) -> coeficiente
# A cadeia e: dia -> 1440min (*1.14) -> duracoes via 1440 -> 60min -> 120/30 -> subdivisoes de 30
DAEE_CETESB: dict[tuple[int, int], float] = {
    # Dia para 24h (fator de correcao dia fixo -> dia movel)
    (1440, 1440): 1.14,
    # De 24h para duracoes menores
    (1440, 720): 0.85,
    (1440, 600): 0.82,
    (1440, 480): 0.78,
    (1440, 360): 0.72,
    (1440, 60): 0.51,
    # De 1h para 2h e 30min
    (60, 120): 1.27,
    (60, 30): 0.74,
    # De 30min para duracoes menores
    (30, 25): 0.91,
    (30, 20): 0.81,
    (30, 15): 0.70,
    (30, 10): 0.54,
    (30, 5): 0.34,
}

# Duracoes-alvo ordenadas (minutos)
DURATIONS_MIN: list[int] = [5, 10, 15, 20, 25, 30, 60, 120, 360, 480, 600, 720, 1440]


def disaggregate_24h(rainfall_1day_mm: float) -> dict[int, float]:
    """
    Desagrega precipitacao maxima diaria em alturas (mm) para 13 duracoes.

    A cadeia de desagregacao segue a tabela DAEE/CETESB (1980):
    1. Multiplica P_1dia por 1.14 para obter P_1440min (correcao dia fixo -> movel)
    2. De 1440min: aplica coeficientes para 720, 600, 480, 360 e 60min
    3. De 60min: aplica coeficientes para 120min e 30min
    4. De 30min: aplica coeficientes para 25, 20, 15, 10 e 5min

    Parameters
    ----------
    rainfall_1day_mm : float
        Precipitacao maxima diaria (mm) — leitura do pluviometro.

    Returns
    -------
    dict[int, float]
        Dicionario {duracao_minutos: altura_mm}.
    """
    depths: dict[int, float] = {}

    # Passo 1: dia -> 1440min
    p1440 = rainfall_1day_mm * DAEE_CETESB[(1440, 1440)]
    depths[1440] = p1440

    # Passo 2: de 1440min para duracoes intermediarias
    for (src, dst), coef in DAEE_CETESB.items():
        if src == 1440 and dst != 1440:
            depths[dst] = p1440 * coef

    # Passo 3: de 60min para 120min e 30min
    p60 = depths[60]
    depths[120] = p60 * DAEE_CETESB[(60, 120)]
    depths[30] = p60 * DAEE_CETESB[(60, 30)]

    # Passo 4: de 30min para subdivisoes
    p30 = depths[30]
    for (src, dst), coef in DAEE_CETESB.items():
        if src == 30:
            depths[dst] = p30 * coef

    return depths


def to_intensity(depths: dict[int, float]) -> dict[int, float]:
    """
    Converte alturas de chuva (mm) em intensidades (mm/h).

    Parameters
    ----------
    depths : dict[int, float]
        Dicionario {duracao_minutos: altura_mm}.

    Returns
    -------
    dict[int, float]
        Dicionario {duracao_minutos: intensidade_mm_h}.
    """
    return {dur: (depth / dur) * 60 for dur, depth in depths.items()}
