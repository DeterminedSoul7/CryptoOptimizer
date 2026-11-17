import heapq
import os
from math import inf
from collections import deque
import graphviz as gv
import networkx as nx
import pandas as pd
import streamlit as st

# =============================================================================
# CONFIGURACIÓN (Optimizado para Streamlit y Parquet con fallback a CSV)
# =============================================================================
PARQUET_NODOS = "nodos.parquet"
PARQUET_ARISTAS = "aristas.parquet"
PARQUET_CCXT = "dataset_ccxt_completo.parquet"

CSV_NODOS = PARQUET_NODOS if os.path.exists(PARQUET_NODOS) else "nodos.csv"
CSV_ARISTAS = PARQUET_ARISTAS if os.path.exists(PARQUET_ARISTAS) else "aristas.csv"
CSV_CCXT = PARQUET_CCXT if os.path.exists(PARQUET_CCXT) else "dataset_ccxt_completo.csv"

ALPHA_TIEMPO = 0.000005                 # USD por ms (pasa el tiempo a dólares en f(n))
FALLBACK_TRANS_MS = 15.0 * 60000.0   # exchange que no está en el CSV CCXT: 15 min
FALLBACK_TRADE_MS = 150.0            # exchange que no está en el CSV CCXT: 150 ms
TAKER_FEE_DEFECTO = 0.001            # exchange sin taker fee en aristas: 0.1 %
H_EXCHANGE = 0.10                    # heurística: otro exchange
H_TOKEN = 0.00                       # heurística: otra moneda
VIGENCIA_DATOS_SEG = 28800           # pasado este tiempo se avisa que los datos son viejos


# =============================================================================
# 1) FORMATO DE TIEMPOS Y VALORES
# =============================================================================
def formatear_min(ms):
    """Duracion de una ruta en minutos."""
    if ms <= 0:
        return "0.00 min"
    minutos = ms / 60_000
    return "<0.01 min" if minutos < 0.01 else f"{minutos:,.2f} min"


def formatear_antiguedad(seg):
    """Antiguedad de los datos en la unidad mas legible (min, h o dias)."""
    if seg < 7_200:
        return f"{seg / 60:,.0f} min"
    if seg < 172_800:
        return f"{seg / 3_600:,.1f} h"
    return f"{seg / 86_400:,.1f} dias"


def _cant(x):
    if x >= 1000:
        return f"{x:,.2f}"
    return f"{x:.4f}" if x >= 1 else f"{x:.6f}"


def _usd(x):
    return f"${x:,.2f} USD" if (x >= 0.01 or x == 0) else f"${x:.4f} USD"


# =============================================================================
# 2) GRAFO (Cacheado para Streamlit con soporte Parquet/CSV)
# =============================================================================
@st.cache_data(show_spinner=False)
def cargar_tiempos_ccxt(csv=CSV_CCXT):
    """{id_exchange: {"trans_ms", "trade_ms"}} del dataset CCXT ({} si no esta o no se lee)."""
    if not os.path.exists(csv):
        return {}
    try:
        df = pd.read_parquet(csv) if csv.endswith('.parquet') else pd.read_csv(csv)
        return {
            str(f["id"]).strip().lower(): {
                "trans_ms": float(f["transferencia_peso_total_min"]) * 60000.0,
                "trade_ms": float(f["trade_peso_total_ms"])
            }
            for _, f in df.iterrows()
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=True)
def cargar_grafo(nodos_csv=CSV_NODOS, aristas_csv=CSV_ARISTAS, ccxt_csv=CSV_CCXT):
    nodos = pd.read_parquet(nodos_csv) if nodos_csv.endswith('.parquet') else pd.read_csv(nodos_csv)
    ar = pd.read_parquet(aristas_csv) if aristas_csv.endswith('.parquet') else pd.read_csv(aristas_csv)
    lookup = cargar_tiempos_ccxt(ccxt_csv)

    def t_trans(ex):
        return lookup.get(str(ex).strip().lower(), {}).get("trans_ms", FALLBACK_TRANS_MS)

    def t_trade(ex):
        return lookup.get(str(ex).strip().lower(), {}).get("trade_ms", FALLBACK_TRADE_MS)

    G = nx.DiGraph(archivo=aristas_csv, tiempos_csv=ccxt_csv if lookup else None, orden=[])
    orden = G.graph["orden"]
    precios = ar.groupby("nodo_id")["Precio Last"].first()
    for r in nodos.itertuples(index=False):
        nid = int(r.node_id)
        G.add_node(nid, exchange=r.Exchange, cripto=r.Criptomoneda, red=r.Red,
                   precio=float(precios.loc[nid]) if nid in precios.index else 1.0)

    def agregar(u, v, **datos):
        if u == v or u not in G or v not in G:
            return
        if G.has_edge(u, v):
            if G[u][v]["fijo_usd"] <= datos["fijo_usd"]:
                return
        else:
            orden.append((u, v))
        G.add_edge(u, v, **datos)

    # A. TRANSFERENCIAS entre exchanges (misma moneda y red)
    for (token, red), grupo in ar.groupby(["Criptomoneda", "Red"]):
        filas = grupo[["nodo_id", "Exchange", "Withdrawal Fee USD"]].values
        for i in range(len(filas)):
            for j in range(len(filas)):
                if i != j:
                    u, ex_u = int(filas[i][0]), str(filas[i][1]).lower()
                    v, fee = int(filas[j][0]), float(filas[j][2])
                    if fee != fee:
                        fee = 0.0
                    agregar(u, v, tipo="inter", fee_pct=0.0, fijo_usd=fee,
                            tiempo_ms=t_trans(ex_u), red=red, token=token)

    # B. TRADES dentro de cada exchange (cambio de moneda o de red)
    taker = ar.groupby("Exchange")["Taker Fee"].mean().to_dict()
    for exchange, grupo in nodos.groupby("Exchange"):
        t_ms = t_trade(str(exchange).lower())
        pct = taker.get(exchange, TAKER_FEE_DEFECTO)
        if not pct or pct <= 0.0 or pct != pct:
            pct = TAKER_FEE_DEFECTO
        ids = grupo["node_id"].tolist()
        for u in ids:
            for v in ids:
                if u != v and (G.nodes[u]["cripto"] != G.nodes[v]["cripto"]
                                or G.nodes[u]["red"] != G.nodes[v]["red"]):
                    agregar(u, v, tipo="intra", fee_pct=pct, fijo_usd=0.0, tiempo_ms=t_ms)

    exchanges = {str(e).strip().lower() for e in nodos["Exchange"]}
    G.graph["sin_tiempo"] = sorted(e for e in exchanges if e not in lookup)
    G.graph["n_exchanges"] = len(exchanges)
    return G


def buscar_nodos(G, exchange=None, cripto=None):
    return [n for n, d in G.nodes(data=True)
            if (not exchange or d["exchange"].lower() == exchange.lower())
            and (not cripto or d["cripto"].upper() == cripto.upper())]


def seleccionar_destinos(G, ex_o, cr_d):
    todos = buscar_nodos(G, None, cr_d)
    externos = [n for n in todos if G.nodes[n]["exchange"].lower() != ex_o.lower()]
    return externos if externos else todos


# =============================================================================
# 3) ALGORITMOS (Kruskal y A*)
# =============================================================================
def costo_arista(d, monto):
    return monto * d["fee_pct"] + d["fijo_usd"]


def peso_arista(d, monto):
    return costo_arista(d, monto) + ALPHA_TIEMPO * d["tiempo_ms"]


def _reconstruir(padre, fin):
    ruta = []
    while fin is not None:
        ruta.append(fin)
        fin = padre[fin]
    return ruta[::-1]


def _resultado(G, ruta, monto, expandidos, peso):
    costo = sum(costo_arista(G[u][v], monto) for u, v in zip(ruta, ruta[1:]))
    tiempo = sum(G[u][v]["tiempo_ms"] for u, v in zip(ruta, ruta[1:]))
    return {"ruta": ruta, "costo": costo, "tiempo_ms": tiempo, "peso": peso,
            "expandidos": expandidos}


# Union-Find para Kruskal
def _find(parent, i):
    while i != parent[i]:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent, u, v):
    ru, rv = _find(parent, u), _find(parent, v)
    if ru != rv:
        parent[ru] = rv
        return True
    return False


def kruskal_mst(nodos, aristas, criterio="precio"):
    parent = {n: n for n in nodos}
    ordenadas = sorted(aristas, key=lambda x: x[2] if criterio == "precio" else x[3])
    mst, total_fee, total_ms = [], 0.0, 0.0
    for u, v, fee, t_ms in ordenadas:
        if _union(parent, u, v):
            mst.append((u, v, fee, t_ms))
            total_fee += fee
            total_ms += t_ms
    return parent, mst, total_fee, total_ms


def aristas_kruskal(G, monto):
    return [(u, v, costo_arista(G[u][v], monto), G[u][v]["tiempo_ms"])
            for u, v in G.graph["orden"] if G.has_edge(u, v)]


def hay_conexion(parent, origenes, destinos):
    return any(_find(parent, o) == _find(parent, d) for o in origenes for d in destinos)


def astar_grupo(G, origenes, destinos, monto, bloqueadas=frozenset()):
    origen = origenes[0]
    destinos = set(destinos)
    pares_destino = {(G.nodes[d]["exchange"], G.nodes[d]["cripto"]) for d in destinos}
    cache_h = {}

    def h(v):
        clave = (G.nodes[v]["exchange"], G.nodes[v]["cripto"])
        if clave not in cache_h:
            mejor = inf
            for ex_d, tok_d in pares_destino:
                x = 0.0
                if clave[0] != ex_d:
                    x += H_EXCHANGE
                if clave[1] != tok_d:
                    x += H_TOKEN
                mejor = min(mejor, x)
            cache_h[clave] = mejor
        return cache_h[clave]

    pq = [(0.0, 0.0, 0.0, origen, [origen])]
    distancias, expandidos = {}, 0
    while pq:
        f, g_costo, g_tiempo, u, camino = heapq.heappop(pq)
        if u in destinos:
            return _resultado(G, camino, monto, expandidos, g_costo + ALPHA_TIEMPO * g_tiempo)
        if u in distancias and distancias[u] <= f:
            continue
        distancias[u] = f
        expandidos += 1
        for v, d in G[u].items():
            if (u, v) in bloqueadas:
                continue
            nuevo_g_costo = g_costo + costo_arista(d, monto)
            nuevo_g_tiempo = g_tiempo + d["tiempo_ms"]
            f_score = nuevo_g_costo + (ALPHA_TIEMPO * nuevo_g_tiempo) + h(v)
            heapq.heappush(pq, (f_score, nuevo_g_costo, nuevo_g_tiempo, v, camino + [v]))
    return None


def ruta_directa(G, origenes, destinos):
    destinos = set(destinos)
    padre = {o: None for o in origenes}
    cola = deque(origenes)
    while cola:
        u = cola.popleft()
        if u in destinos:
            return _reconstruir(padre, u)
        for v in G[u]:
            if v not in padre:
                padre[v] = u
                cola.append(v)
    return None


def buscar_ruta(G, origenes, destinos, monto):
    aristas = aristas_kruskal(G, monto)
    nodos = list(G.nodes)
    parent, _, fee_p, t_p = kruskal_mst(nodos, aristas, "precio")
    _, _, fee_t, t_t = kruskal_mst(nodos, aristas, "tiempo")
    r = {"conectado": hay_conexion(parent, origenes[:1], destinos), "ruta": None,
         "alternativa": None, "topologia": (fee_p, t_p, fee_t, t_t)}
    if r["conectado"]:
        r["ruta"] = astar_grupo(G, origenes, destinos, monto)
        if r["ruta"] is not None:
            ruta = r["ruta"]["ruta"]
            bloqueadas = {(ruta[i], ruta[i + 1]) for i in range(len(ruta) - 1)}
            r["alternativa"] = astar_grupo(G, origenes, destinos, monto, bloqueadas)
    return r


# =============================================================================
# 4) CONSULTA: VALIDACIÓN Y EVALUACIÓN
# =============================================================================
def obtener_ts_cotizacion(archivo_datos, archivo_fecha="fecha_extraccion.txt"):
    if os.path.exists(archivo_fecha):
        try:
            with open(archivo_fecha, encoding="utf-8") as f:
                txt = f.read().strip()
            return float(txt)
        except ValueError:
            try:
                from datetime import datetime
                return datetime.fromisoformat(txt).timestamp()
            except ValueError:
                pass
    return os.path.getmtime(archivo_datos)


def evaluar_consulta(G, p):
    import time
    res = {"estado": None, "origenes": [], "destinos": [], "r": None, "t_ms": 0.0,
           "vencidos": False, "edad_seg": 0.0, "p": p}
    try:
        monto = float(p.get("monto"))
    except (TypeError, ValueError):
        monto = None
    ex_o = str(p.get("ex_o") or "").strip().lower()
    cr_o = str(p.get("cr_o") or "").strip().upper()
    cr_d = str(p.get("cr_d") or "").strip().upper()
    if monto is None or not (0 < monto < inf) or not ex_o or not cr_o or not cr_d:
        res["estado"] = "ENTRADA_INVALIDA"
        return res
    res["p"] = p = {"monto": monto, "ex_o": ex_o, "cr_o": cr_o, "cr_d": cr_d}
    if ex_o not in {d["exchange"].lower() for _, d in G.nodes(data=True)}:
        res["estado"] = "ORIGEN_INEXISTENTE"
        return res
    inicio = buscar_nodos(G, ex_o, cr_o)
    if not inicio:
        res["estado"] = "MONEDA_ORIGEN_INEXISTENTE"
        return res
    destinos = seleccionar_destinos(G, ex_o, cr_d)
    res["origenes"], res["destinos"] = inicio[:1], destinos
    if not destinos:
        res["estado"] = "DESTINO_INEXISTENTE"
        return res
    if inicio[0] in destinos:
        res["estado"] = "YA_EN_DESTINO"
        return res
    try:
        archivo_ref = G.graph.get("archivo", CSV_NODOS)
        if os.path.exists(archivo_ref):
            edad = time.time() - obtener_ts_cotizacion(archivo_ref)
            res["vencidos"], res["edad_seg"] = edad > VIGENCIA_DATOS_SEG, edad
    except OSError:
        pass

    t0 = time.perf_counter()
    r = buscar_ruta(G, res["origenes"], destinos, monto)
    res["t_ms"] = (time.perf_counter() - t0) * 1000
    res["r"] = r
    if not r["conectado"]:
        res["estado"] = "SIN_CONEXION"
    elif r["ruta"] is None:
        res["estado"] = "SIN_RUTA"
    else:
        res["estado"] = "OK"
    return res


# =============================================================================
# 5) DETALLES DE RUTA Y GENERACIÓN DE GRAFO
# =============================================================================
def _pasos_ruta(G, res, monto):
    ruta = res["ruta"]
    pasos, restante = [], monto
    for u, v in zip(ruta, ruta[1:]):
        d = G[u][v]
        nu, nv = G.nodes[u], G.nodes[v]
        costo = costo_arista(d, monto)
        cant = restante / (nu.get("precio") or 1.0)
        ex_u, ex_v = nu["exchange"].capitalize(), nv["exchange"].capitalize()
        if d["tipo"] == "intra":
            tipo, medio, red = "Trade", "Plataforma", "Local"
            if nu["cripto"] == nv["cripto"]:
                detalle = f"{ex_u}: Cambiar red de {_cant(cant)} {nu['cripto']} ({nu['red']} -> {nv['red']})"
            else:
                detalle = f"{ex_u}: Intercambiar {_cant(cant)} {nu['cripto']} -> {nv['cripto']}"
        else:
            tipo, medio, red = "Transferencia", "Red de transferencia", str(d["red"])
            detalle = f"Transferir de {ex_u} -> {ex_v} ({_cant(cant)} {nu['cripto']})"
        pasos.append({"tipo": tipo, "detalle": detalle, "medio": medio, "red": red,
                      "costo": costo, "tiempo_ms": d["tiempo_ms"]})
        restante -= costo
    return pasos


def _resumen_ruta(G, res, monto):
    pasos = _pasos_ruta(G, res, monto)
    trading = sum(x["costo"] for x in pasos if x["tipo"] == "Trade")
    retiro = sum(x["costo"] for x in pasos if x["tipo"] == "Transferencia")
    total = trading + retiro
    fin = G.nodes[res["ruta"][-1]]
    final_txt = f"{_cant((monto - total) / (fin.get('precio') or 1.0))} {fin['cripto']}"
    return {"pasos": pasos, "trading": trading, "retiro": retiro, "total": total,
            "redes": list(dict.fromkeys(x["red"] for x in pasos if x["red"] and x["red"] != "Local")),
            "tiempo_ms": sum(x["tiempo_ms"] for x in pasos), "final_txt": final_txt}


def generar_grafo_visual(G, r, monto, p):
    dot = gv.Digraph(
        comment="Ruta_Optima_A_Star", format="png",
        graph_attr={"rankdir": "LR", "bgcolor": "#FFFFFF", "splines": "true", "nodesep": "0.6", "ranksep": "1.0"},
        node_attr={"shape": "rect", "style": "rounded,filled", "fontname": "Helvetica", "fontsize": "9"},
        edge_attr={"fontname": "Helvetica", "fontsize": "8", "arrowsize": "0.7"}
    )

    rutas_a_dibujar = []
    if r.get("ruta") and r["ruta"].get("ruta"):
        rutas_a_dibujar.append(("óptima", r["ruta"]["ruta"]))
    if r.get("alternativa") and r["alternativa"].get("ruta"):
        rutas_a_dibujar.append(("alternativa", r["alternativa"]["ruta"]))

    todos_nodos = set()
    for _, camino in rutas_a_dibujar:
        todos_nodos.update(camino)

    if not todos_nodos:
        return None

    df_nodos = pd.read_parquet(CSV_NODOS) if CSV_NODOS.endswith('.parquet') else pd.read_csv(CSV_NODOS)
    df_ruta = df_nodos[df_nodos["node_id"].isin(todos_nodos)]
    moneda_origen_actual = p["cr_o"].upper()

    for ex in df_ruta["Exchange"].unique():
        with dot.subgraph(name=f"cluster_{ex}") as c:
            c.attr(label=f"EXCHANGE: {ex.upper()}", style="dashed", color="#546E7A", bgcolor="#F8F9FA", fontcolor="#263238", fontsize="10")
            for _, fila in df_ruta[df_ruta["Exchange"] == ex].iterrows():
                nid = str(fila["node_id"])
                label_nodo = f"{fila['Criptomoneda']}\n({fila['Red']})"
                color = "#FFF2CC" if str(fila['Criptomoneda']).upper() == moneda_origen_actual else "#C8E6C9"
                c.node(nid, label=label_nodo, fillcolor=color, color="#455A64")

    if r.get("ruta") and r["ruta"].get("ruta"):
        camino_0 = r["ruta"]["ruta"]
        for i in range(len(camino_0) - 1):
            u, v = camino_0[i], camino_0[i + 1]
            d = G[u][v]
            fee = costo_arista(d, monto)
            t_min = d["tiempo_ms"] / 60000.0
            formato_t = f"{t_min:.2f} min" if t_min >= 0.01 else f"{t_min:.4f} min"
            if d["tipo"] == "inter":
                etiqueta = f"Transferencia: {d.get('red', 'Principal')}\n${fee:.2f} | {formato_t}"
                color_l = "#E65100"
            else:
                etiqueta = f"Trade\n${fee:.2f} | {formato_t}"
                color_l = "#2E7D32"
            dot.edge(str(u), str(v), label=etiqueta, color=color_l, penwidth="2.5", fontcolor="#263238")

    if r.get("alternativa") and r["alternativa"].get("ruta"):
        camino_1 = r["alternativa"]["ruta"]
        for i in range(len(camino_1) - 1):
            u, v = camino_1[i], camino_1[i + 1]
            d = G[u][v]
            fee = costo_arista(d, monto)
            t_min = d["tiempo_ms"] / 60000.0
            formato_t = f"{t_min:.2f} min" if t_min >= 0.01 else f"{t_min:.4f} min"
            if d["tipo"] == "inter":
                etiqueta = f"[Alt] Transferencia: {d.get('red', 'Principal')}\n${fee:.2f} | {formato_t}"
            else:
                etiqueta = f"[Alt] Trade\n${fee:.2f} | {formato_t}"
            dot.edge(str(u), str(v), label=etiqueta, color="#1565C0", style="dashed", penwidth="1.8", fontcolor="#1565C0")

    return dot


# =============================================================================
# 6) INTERFAZ GRÁFICA STREAMLIT (Soporte Modo Lite y Modo Pro)
# =============================================================================
st.set_page_config(
    page_title="CryptoOptimizer - Enrutamiento Multiexchange",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
    .main-title { font-size: 2rem; font-weight: 800; color: #1E293B; margin-bottom: 0px; }
    .sub-title { font-size: 0.95rem; color: #64748B; margin-bottom: 1.5rem; }
    .badge-lite { background: #E0F2FE; color: #0369A1; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.8rem; }
    .badge-pro { background: #FEF3C7; color: #B45309; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# Carga de grafo
with st.spinner("Cargando grafo y datasets optimizados (Parquet/CSV)..."):
    G = cargar_grafo()

if G is None:
    st.error("No se pudo cargar el grafo. Verifique la existencia de los archivos Parquet o CSV.")
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("Configuración de Operación")

modo_ui = st.sidebar.radio(
    "Modalidad de Vista:",
    ["Modo Lite (Usuario)", "Modo Pro (Avanzado)"],
    index=0,
    help="El modo Lite muestra únicamente la ruta de transferencia sin detalles técnicos de grafos."
)
es_pro = (modo_ui == "Modo Pro (Avanzado)")

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de Entrada")

monto_input = st.sidebar.number_input("Monto Inicial (USD)", min_value=1.0, value=1000.0, step=100.0)

exchanges_disponibles = sorted(list({d["exchange"] for _, d in G.nodes(data=True)}))
def_ex = "binance" if "binance" in exchanges_disponibles else exchanges_disponibles[0]
idx_ex = exchanges_disponibles.index(def_ex)
ex_o_input = st.sidebar.selectbox("Exchange de Origen", exchanges_disponibles, index=idx_ex)

monedas_origen = sorted(list({d["cripto"] for _, d in G.nodes(data=True) if d["exchange"].lower() == ex_o_input.lower()}))
def_cr_o = "BTC" if "BTC" in monedas_origen else (monedas_origen[0] if monedas_origen else "BTC")
idx_cr_o = monedas_origen.index(def_cr_o) if def_cr_o in monedas_origen else 0
cr_o_input = st.sidebar.selectbox("Moneda que Posee", monedas_origen, index=idx_cr_o)

todas_monedas = sorted(list({d["cripto"] for _, d in G.nodes(data=True)}))
def_cr_d = "ADA" if "ADA" in todas_monedas else todas_monedas[0]
idx_cr_d = todas_monedas.index(def_cr_d)
cr_d_input = st.sidebar.selectbox("Moneda de Destino", todas_monedas, index=idx_cr_d)

ejecutar = st.sidebar.button("Optimizar Transferencia", type="primary", use_container_width=True)

# Encabezado
badge = '<span class="badge-pro">MODO PRO</span>' if es_pro else '<span class="badge-lite">MODO LITE</span>'
st.markdown(f"<div class='main-title'>⚡ CryptoOptimizer {badge}</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Sistema de optimización de rutas entre exchanges con A* y Kruskal (MST)</div>", unsafe_allow_html=True)

# =============================================================================
# EJECUCIÓN Y RENDERIZADO
# =============================================================================
if ejecutar or "ultima_evaluacion" in st.session_state:
    if ejecutar:
        params = {"monto": monto_input, "ex_o": ex_o_input, "cr_o": cr_o_input, "cr_d": cr_d_input}
        with st.spinner("Procesando enrutamiento y análisis topológico..."):
            ev = evaluar_consulta(G, params)
        st.session_state["ultima_evaluacion"] = ev
    else:
        ev = st.session_state["ultima_evaluacion"]

    estado = ev["estado"]
    p = ev["p"]

    if estado == "ENTRADA_INVALIDA":
        st.error("Monto o datos inválidos: ingrese un monto mayor que 0.")
    elif estado == "ORIGEN_INEXISTENTE":
        st.error(f"El exchange '{p['ex_o']}' no existe en el dataset.")
    elif estado == "MONEDA_ORIGEN_INEXISTENTE":
        st.error(f"El exchange '{p['ex_o']}' no tiene la moneda '{p['cr_o']}'.")
    elif estado == "DESTINO_INEXISTENTE":
        st.error(f"La moneda '{p['cr_d']}' no existe en el dataset.")
    elif estado == "YA_EN_DESTINO":
        st.warning(f"Posesión actual detectada: ya posee {p['cr_d']} en {p['ex_o']}.")
    elif estado == "SIN_CONEXION":
        st.error("Validación Kruskal: No existe conectividad estructural hacia el destino.")
    elif estado == "SIN_RUTA":
        st.error("Validación de red: Existe conexión estructural pero no se encontró ruta dirigida válida.")
    elif estado == "OK":
        r = ev["r"]
        res_o = _resumen_ruta(G, r["ruta"], p["monto"])
        camino_dir = ruta_directa(G, ev["origenes"], ev["destinos"])
        res_d = _resumen_ruta(G, _resultado(G, camino_dir, p["monto"], 0, 0.0), p["monto"]) if camino_dir else None
        res_a = _resumen_ruta(G, r["alternativa"], p["monto"]) if r.get("alternativa") else None

        if ev["vencidos"]:
            st.warning(f"Aviso de mercado: Los datos presentan {formatear_antiguedad(ev['edad_seg'])} de antigüedad; los costos pueden haber variado.")

        # =====================================================================
        # VISTA: MODO LITE
        # =====================================================================
        if not es_pro:
            st.subheader("Resumen de tu Operación")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Monto a Enviar", f"${p['monto']:,.2f} USD")
            c2.metric("Comisión Total (Fees)", _usd(res_o["total"]))
            c3.metric("Tiempo Total Estimado", formatear_min(res_o["tiempo_ms"]))
            c4.metric("Recibirás en Destino", res_o["final_txt"])

            if res_d and res_o:
                ahorro = res_d["total"] - res_o["total"]
                if ahorro > 1e-9 and res_d["total"] > 0:
                    pct = (ahorro / res_d["total"]) * 100
                    st.success(f"💡 **Ruta recomendada:** Al optimizar la ruta te ahorras **{_usd(ahorro)}** ({pct:.1f}% menos en comisiones que la transferencia directa tradicional).")

            st.markdown("### 📋 Pasos a Seguir para la Transferencia")
            df_simple = pd.DataFrame(res_o["pasos"])
            df_simple["Costo (USD)"] = df_simple["costo"].map(_usd)
            df_simple["Tiempo Estimado"] = df_simple["tiempo_ms"].map(formatear_min)
            df_simple["Paso"] = range(1, len(df_simple) + 1)

            st.dataframe(
                df_simple[["Paso", "tipo", "detalle", "red", "Costo (USD)", "Tiempo Estimado"]],
                use_container_width=True,
                hide_index=True
            )
            st.caption("ℹ️ Para ver la topología completa del grafo, árboles de expansión mínima (Kruskal) y alternativas, activa el **Modo Pro** en la barra lateral.")

        # =====================================================================
        # VISTA: MODO PRO
        # =====================================================================
        else:
            tab_res, tab_graph, tab_data = st.tabs([
                "🚀 Resultados y Optimización",
                "🗺️ Topología de Grafo",
                "📊 Explorador de Datos"
            ])

            with tab_res:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Nodos en Grafo |V|", f"{G.number_of_nodes():,}")
                col_m2.metric("Aristas |E|", f"{G.number_of_edges():,}")
                n_ex = G.graph.get("n_exchanges", len(exchanges_disponibles))
                sin_t = len(G.graph.get("sin_tiempo", []))
                col_m3.metric("Exchanges CCXT", f"{n_ex - sin_t} / {n_ex}")
                col_m4.metric("Conexión Kruskal", "Conectado", delta="True (Union-Find)")

                fee_p, t_p, fee_t, t_t = r["topologia"]
                with st.expander("🌐 Validación Estructural con Kruskal (MST)", expanded=True):
                    st.info(
                        f"**1. MST por Precio:** Tarifa: ${fee_p:,.2f} USD | Tiempo: {formatear_min(t_p)}  \n"
                        f"**2. MST por Tiempo:** Tarifa: ${fee_t:,.2f} USD | Tiempo: {formatear_min(t_t)}  \n"
                        f"*Balance:* Optimizar por tiempo ahorra {formatear_min(t_p - t_t)} con un sobrecosto global de ${fee_t - fee_p:,.2f} USD."
                    )

                st.markdown("#### 📊 Comparativa de Enrutamiento")
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.markdown("##### ❌ Ruta Directa (Sin A*)")
                    if res_d:
                        st.write(f"• Pasos: {len(res_d['pasos'])}")
                        st.write(f"• Total Fees: **{_usd(res_d['total'])}**")
                        st.write(f"• Tiempo estimado: {formatear_min(res_d['tiempo_ms'])}")
                        st.write(f"• Monto Final: {res_d['final_txt']}")
                    else:
                        st.write("No disponible")

                with col_r2:
                    st.markdown("##### ⭐ Ruta Óptima (A*)")
                    if res_o:
                        st.write(f"• Pasos: {len(res_o['pasos'])}")
                        st.write(f"• Total Fees: **{_usd(res_o['total'])}**")
                        st.write(f"• Tiempo estimado: {formatear_min(res_o['tiempo_ms'])}")
                        st.write(f"• Monto Final: {res_o['final_txt']}")
                    else:
                        st.write("No disponible")

                with col_r3:
                    st.markdown("##### 🔄 Ruta Alternativa (A*)")
                    if res_a:
                        st.write(f"• Pasos: {len(res_a['pasos'])}")
                        st.write(f"• Total Fees: **{_usd(res_a['total'])}**")
                        st.write(f"• Tiempo estimado: {formatear_min(res_a['tiempo_ms'])}")
                        st.write(f"• Monto Final: {res_a['final_txt']}")
                    else:
                        st.caption("No existe ruta disjunta secundaria")

                if res_d and res_o:
                    ahorro = res_d["total"] - res_o["total"]
                    if ahorro > 1e-9 and res_d["total"] > 0:
                        pct = (ahorro / res_d["total"]) * 100
                        st.success(f"🎉 **¡Ahorro total de {_usd(ahorro)} ({pct:.1f}% de reducción en comisiones con A*)!**")

                st.markdown("#### 📋 Detalle de Pasos (Ruta Óptima)")
                df_p_opt = pd.DataFrame(res_o["pasos"])
                df_p_opt["Costo_USD"] = df_p_opt["costo"].map(_usd)
                df_p_opt["Tiempo"] = df_p_opt["tiempo_ms"].map(formatear_min)
                df_p_opt["Paso"] = range(1, len(df_p_opt) + 1)
                st.dataframe(
                    df_p_opt[["Paso", "tipo", "detalle", "medio", "red", "Costo_USD", "Tiempo"]],
                    use_container_width=True,
                    hide_index=True
                )

                if res_a:
                    st.markdown("#### 📋 Detalle de Pasos (Ruta Alternativa)")
                    df_p_alt = pd.DataFrame(res_a["pasos"])
                    df_p_alt["Costo_USD"] = df_p_alt["costo"].map(_usd)
                    df_p_alt["Tiempo"] = df_p_alt["tiempo_ms"].map(formatear_min)
                    df_p_alt["Paso"] = range(1, len(df_p_alt) + 1)
                    st.dataframe(
                        df_p_alt[["Paso", "tipo", "detalle", "medio", "red", "Costo_USD", "Tiempo"]],
                        use_container_width=True,
                        hide_index=True
                    )

            with tab_graph:
                st.markdown("#### 🗺️ Visualización Topológica en Graphviz")
                st.caption("Línea naranja continua: Ruta Óptima (A*) | Línea azul discontinua: Ruta Alternativa (A*)")
                dot_obj = generar_grafo_visual(G, r, p["monto"], p)
                if dot_obj:
                    st.graphviz_chart(dot_obj.source, use_container_width=True)
                else:
                    st.info("No se pudo generar el diagrama visual del grafo.")

            with tab_data:
                st.markdown("#### 📂 Datasets del Mercado")
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    st.write(f"**Nodos ({CSV_NODOS}):**")
                    df_nodos_view = pd.read_parquet(CSV_NODOS) if CSV_NODOS.endswith('.parquet') else pd.read_csv(CSV_NODOS)
                    st.dataframe(df_nodos_view.head(100), height=350)
                with c_d2:
                    st.write(f"**Aristas ({CSV_ARISTAS}):**")
                    df_aristas_view = pd.read_parquet(CSV_ARISTAS) if CSV_ARISTAS.endswith('.parquet') else pd.read_csv(CSV_ARISTAS)
                    st.dataframe(df_aristas_view.head(100), height=350)
else:
    st.info("Configure los parámetros en la barra lateral y haga clic en **Optimizar Transferencia** para iniciar el análisis.")