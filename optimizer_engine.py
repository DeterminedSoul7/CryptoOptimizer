"""
optimizer_engine.py - Motor de Optimización de Rutas Cripto (Core)
Compatible con CryptoOptimizer y Streamlit.
"""

import heapq
import os
import time
from math import inf
import graphviz as gv
import networkx as nx
import pandas as pd

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================
CSV_NODOS = "nodos.csv"
CSV_ARISTAS = "aristas.csv"
CSV_CCXT = "dataset_ccxt_completo.csv"

ALPHA_TIEMPO = 0.000005              # USD por ms (factor de ponderación de tiempo)
FALLBACK_TRANS_MS = 15.0 * 60000.0   # Respaldo: 15 min
FALLBACK_TRADE_MS = 150.0            # Respaldo: 150 ms
TAKER_FEE_DEFECTO = 0.001            # Respaldo: 0.1%
H_EXCHANGE = 0.10                    # Heurística: costo de cambio de exchange
H_TOKEN = 0.00                       # Heurística: costo de cambio de token
VIGENCIA_DATOS_SEG = 300             # Tiempo en seg para advertencia de frescura


# =============================================================================
# FUNCIONES DE TIEMPO Y FORMATO
# =============================================================================
def formatear_min(ms):
    if ms <= 0:
        return "0.00 min"
    minutos = ms / 60_000
    return "<0.01 min" if minutos < 0.01 else f"{minutos:,.2f} min"


def formatear_antiguedad(seg):
    if seg < 7_200:
        return f"{seg / 60:,.0f} min"
    if seg < 172_800:
        return f"{seg / 3_600:,.1f} h"
    return f"{seg / 86_400:,.1f} dias"


def formatear_cant(x):
    if x >= 1000:
        return f"{x:,.2f}"
    return f"{x:.4f}" if x >= 1 else f"{x:.6f}"


def formatear_usd(x):
    return f"${x:,.2f} USD" if (x >= 0.01 or x == 0) else f"${x:.4f} USD"


# =============================================================================
# CONSTRUCCIÓN DEL GRAFO
# =============================================================================
def cargar_tiempos_ccxt(csv=CSV_CCXT):
    if not os.path.exists(csv):
        return {}
    try:
        df = pd.read_csv(csv)
        return {str(f["id"]).strip().lower():
                {"trans_ms": float(f["transferencia_peso_total_min"]) * 60000.0,
                 "trade_ms": float(f["trade_peso_total_ms"])}
                for _, f in df.iterrows()}
    except Exception:
        return {}


def cargar_grafo(nodos_csv=CSV_NODOS, aristas_csv=CSV_ARISTAS, ccxt_csv=CSV_CCXT):
    nodos = pd.read_csv(nodos_csv)
    ar = pd.read_csv(aristas_csv)
    lookup = cargar_tiempos_ccxt(ccxt_csv)

    def t_trans(ex):
        return lookup.get(str(ex).strip().lower(), {}).get("trans_ms", FALLBACK_TRANS_MS)

    def t_trade(ex):
        return lookup.get(str(ex).strip().lower(), {}).get("trade_ms", FALLBACK_TRADE_MS)

    G = nx.DiGraph(archivo=aristas_csv, tiempos_csv=ccxt_csv if lookup else None, orden=[], archivo_nodos=nodos_csv)
    orden = G.graph["orden"]
    precios = ar.groupby("nodo_id")["Precio Last"].first()

    for r in nodos.itertuples(index=False):
        nid = int(r.node_id)
        G.add_node(nid, exchange=str(r.Exchange).strip(), cripto=str(r.Criptomoneda).strip().upper(),
                   red=str(r.Red).strip(), precio=float(precios.loc[nid]) if nid in precios.index else 1.0)

    def agregar(u, v, **datos):
        if u == v or u not in G or v not in G:
            return
        if G.has_edge(u, v):
            if G[u][v]["fijo_usd"] <= datos["fijo_usd"]:
                return
        else:
            orden.append((u, v))
        G.add_edge(u, v, **datos)

    # Transferencias inter-exchange
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

    # Trades intra-exchange
    taker = ar.groupby("Exchange")["Taker Fee"].mean().to_dict()
    for exchange, grupo in nodos.groupby("Exchange"):
        t_ms = t_trade(str(exchange).lower())
        pct = taker.get(exchange, TAKER_FEE_DEFECTO)
        if not pct or pct <= 0.0 or pct != pct:
            pct = TAKER_FEE_DEFECTO
        ids = grupo["node_id"].tolist()
        for u in ids:
            for v in ids:
                if u != v and (G.nodes[u]["cripto"] != G.nodes[v]["cripto"] or G.nodes[u]["red"] != G.nodes[v]["red"]):
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
# ALGORITMOS: KRUSKAL Y A*
# =============================================================================
def costo_arista(d, monto):
    return monto * d["fee_pct"] + d["fijo_usd"]


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
    cola = [origenes[0]]
    idx = 0
    while idx < len(cola):
        u = cola[idx]
        idx += 1
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
# EVALUACIÓN Y DESGLOSE DE MÉTRICAS
# =============================================================================
def obtener_ts_cotizacion(archivo_datos, archivo_fecha="fecha_extraccion.txt"):
    if os.path.exists(archivo_fecha):
        try:
            txt = open(archivo_fecha, encoding="utf-8").read().strip()
            return float(txt)
        except Exception:
            pass
    return os.path.getmtime(archivo_datos)


def evaluar_consulta(G, p):
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
    exchanges_disponibles = {d["exchange"].lower() for _, d in G.nodes(data=True)}
    if ex_o not in exchanges_disponibles:
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
        edad = time.time() - obtener_ts_cotizacion(G.graph["archivo"])
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


def obtener_detalles_ruta(G, res_camino, monto):
    if not res_camino or not res_camino.get("ruta"):
        return None
    ruta = res_camino["ruta"]
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
                detalle = f"{ex_u}: Cambiar red de {formatear_cant(cant)} {nu['cripto']} ({nu['red']} -> {nv['red']})"
            else:
                detalle = f"{ex_u}: Intercambiar {formatear_cant(cant)} {nu['cripto']} -> {nv['cripto']}"
        else:
            tipo, medio, red = "Transferencia", "Red de transferencia", str(d.get("red", "Principal"))
            detalle = f"Transferir de {ex_u} -> {ex_v} ({formatear_cant(cant)} {nu['cripto']})"
        pasos.append({"Paso": len(pasos) + 1, "Tipo": tipo, "Detalle": detalle, "Medio": medio, "Red": red,
                      "Costo_USD": costo, "Tiempo_ms": d["tiempo_ms"]})
        restante -= costo

    trading = sum(x["Costo_USD"] for x in pasos if x["Tipo"] == "Trade")
    retiro = sum(x["Costo_USD"] for x in pasos if x["Tipo"] == "Transferencia")
    total = trading + retiro
    fin = G.nodes[ruta[-1]]
    monto_recibido = (monto - total) / (fin.get("precio") or 1.0)

    return {
        "pasos": pasos,
        "trading": trading,
        "retiro": retiro,
        "total": total,
        "redes": list(dict.fromkeys(x["Red"] for x in pasos if x["Red"] and x["Red"] != "Local")),
        "tiempo_ms": sum(x["Tiempo_ms"] for x in pasos),
        "monto_recibido": monto_recibido,
        "moneda_recibida": fin["cripto"],
        "final_txt": f"{formatear_cant(monto_recibido)} {fin['cripto']}"
    }


# =============================================================================
# VISUALIZACIÓN CON GRAPHVIZ
# =============================================================================
def generar_dot_graphviz(G, ruta_optima=None, ruta_alternativa=None):
    dot = gv.Digraph(
        comment="Rutas_CryptoOptimizer",
        format="png",
        graph_attr={"rankdir": "LR", "bgcolor": "#FFFFFF", "splines": "true", "nodesep": "0.5", "ranksep": "1.0"},
        node_attr={"shape": "rect", "style": "rounded,filled", "fontname": "Helvetica", "fontsize": "9"},
        edge_attr={"fontname": "Helvetica", "fontsize": "8", "arrowsize": "0.7"}
    )

    nodos_totales = set()
    if ruta_optima and ruta_optima.get("ruta"):
        nodos_totales.update(ruta_optima["ruta"])
    if ruta_alternativa and ruta_alternativa.get("ruta"):
        nodos_totales.update(ruta_alternativa["ruta"])

    if not nodos_totales:
        return dot

    exchanges_involucrados = {G.nodes[n]["exchange"] for n in nodos_totales}
    for ex in exchanges_involucrados:
        with dot.subgraph(name=f"cluster_{ex.lower()}") as c:
            c.attr(label=f"EXCHANGE: {ex.upper()}", style="dashed", color="#546E7A",
                   bgcolor="#F8F9FA", fontcolor="#263238", fontsize="10")
            nodos_del_ex = [n for n in nodos_totales if G.nodes[n]["exchange"] == ex]
            for n in nodos_del_ex:
                d = G.nodes[n]
                color_fondo = "#FFF2CC" if d["cripto"] == "BTC" else "#D5E8D4"
                label_nodo = f"{d['cripto']}\n({d['red']})"
                c.node(str(n), label=label_nodo, fillcolor=color_fondo, color="#455A64")

    if ruta_optima and ruta_optima.get("ruta"):
        camino = ruta_optima["ruta"]
        for i in range(len(camino) - 1):
            u, v = camino[i], camino[i + 1]
            d = G[u][v]
            costo = d["fijo_usd"]
            t_txt = formatear_min(d["tiempo_ms"])
            tipo_txt = "Retiro" if d["tipo"] == "inter" else "Trade"
            label_e = f"[Óptima] {tipo_txt}\n${costo:.2f} | {t_txt}"
            dot.edge(str(u), str(v), label=label_e, color="#E65100", penwidth="2.5")

    if ruta_alternativa and ruta_alternativa.get("ruta"):
        camino = ruta_alternativa["ruta"]
        for i in range(len(camino) - 1):
            u, v = camino[i], camino[i + 1]
            d = G[u][v]
            costo = d["fijo_usd"]
            t_txt = formatear_min(d["tiempo_ms"])
            tipo_txt = "Retiro" if d["tipo"] == "inter" else "Trade"
            label_e = f"[Alt] {tipo_txt}\n${costo:.2f} | {t_txt}"
            dot.edge(str(u), str(v), label=label_e, color="#1565C0", style="dashed", penwidth="1.8")

    return dot