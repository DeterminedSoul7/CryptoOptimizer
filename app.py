"""
app.py - Interfaz Web en Streamlit para CryptoOptimizer
Soporta Modo Lite (Usuario) y Modo Pro (Técnico/Analista).
"""

import os
import optimizer_engine as engine
import pandas as pd
import streamlit as st

# =============================================================================
# CONFIGURACIÓN GENERAL Y ESTILOS
# =============================================================================
st.set_page_config(
    page_title="CryptoOptimizer - Enrutamiento Multiexchange",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2rem; font-weight: 800; color: #1E293B; margin-bottom: 0px; }
    .sub-title { font-size: 0.95rem; color: #64748B; margin-bottom: 1.5rem; }
    .lite-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .badge-lite {
        background: #E0F2FE;
        color: #0369A1;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    .badge-pro {
        background: #FEF3C7;
        color: #B45309;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando dataset de mercado y construyendo el grafo...")
def obtener_grafo():
  return engine.cargar_grafo()


# Validar archivos requeridos
faltan = [
    f
    for f in (engine.CSV_NODOS, engine.CSV_ARISTAS)
    if not os.path.exists(f)
]
if faltan:
  st.error(
      f"Faltan archivos esenciales en el directorio: {', '.join(faltan)}. Por"
      " favor súbelos para continuar."
  )
  st.stop()

G = obtener_grafo()

exchanges_disponibles = sorted(
    list({d["exchange"].lower() for _, d in G.nodes(data=True)})
)
monedas_disponibles = sorted(
    list({d["cripto"].upper() for _, d in G.nodes(data=True)})
)

# =============================================================================
# BARRA LATERAL: MODO Y PARÁMETROS
# =============================================================================
st.sidebar.header("Configuración de Operación")

# Selector de Modo
modo_ui = st.sidebar.radio(
    "Modalidad de Vista:",
    ["Modo Lite (Usuario)", "Modo Pro (Avanzado)"],
    index=0,
    help="El modo Lite muestra únicamente la ruta de transferencia sin detalles técnicos de grafos.",
)
es_pro = modo_ui == "Modo Pro (Avanzado)"

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de Entrada")

monto = st.sidebar.number_input(
    "Monto inicial (USD):", min_value=1.0, value=1000.0, step=50.0
)

def_ex = "binance" if "binance" in exchanges_disponibles else exchanges_disponibles[0]
idx_ex = exchanges_disponibles.index(def_ex)
ex_o = st.sidebar.selectbox("Exchange Origen:", exchanges_disponibles, index=idx_ex)

monedas_en_ex = sorted(
    list(
        {
            d["cripto"].upper()
            for _, d in G.nodes(data=True)
            if d["exchange"].lower() == ex_o.lower()
        }
    )
)
def_cr_o = "BTC" if "BTC" in monedas_en_ex else (monedas_en_ex[0] if monedas_en_ex else monedas_disponibles[0])
idx_cr_o = monedas_en_ex.index(def_cr_o) if def_cr_o in monedas_en_ex else 0
cr_o = st.sidebar.selectbox("Moneda que Posee:", monedas_en_ex, index=idx_cr_o)

def_cr_d = "ADA" if "ADA" in monedas_disponibles else monedas_disponibles[0]
idx_cr_d = monedas_disponibles.index(def_cr_d)
cr_d = st.sidebar.selectbox("Moneda Deseada:", monedas_disponibles, index=idx_cr_d)

btn_calcular = st.sidebar.button(
    "Optimizar Transferencia", type="primary", use_container_width=True
)

# =============================================================================
# ENCABEZADO PRINCIPAL
# =============================================================================
badge = (
    '<span class="badge-pro">MODO PRO</span>'
    if es_pro
    else '<span class="badge-lite">MODO LITE</span>'
)
st.markdown(
    f"<div class='main-title'>CryptoOptimizer {badge}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-title'>Enrutamiento inteligente entre exchanges multiactivo con costo y tiempo optimizados</div>",
    unsafe_allow_html=True,
)

# =============================================================================
# LÓGICA DE CÁLCULO
# =============================================================================
if btn_calcular or "ultima_evaluacion" in st.session_state:
  if btn_calcular:
    params = {"monto": monto, "ex_o": ex_o, "cr_o": cr_o, "cr_d": cr_d}
    ev = engine.evaluar_consulta(G, params)
    st.session_state["ultima_evaluacion"] = ev
  else:
    ev = st.session_state["ultima_evaluacion"]

  estado = ev["estado"]
  p = ev["p"]

  if estado == "ENTRADA_INVALIDA":
    st.error("Monto o parámetros inválidos. Ingrese un valor mayor a cero.")
  elif estado == "ORIGEN_INEXISTENTE":
    st.error(f"El exchange '{p['ex_o']}' no existe en el sistema.")
  elif estado == "MONEDA_ORIGEN_INEXISTENTE":
    st.error(f"El exchange '{p['ex_o']}' no tiene disponible la moneda '{p['cr_o']}'.")
  elif estado == "DESTINO_INEXISTENTE":
    st.error(f"La moneda de destino '{p['cr_d']}' no está registrada en el mercado.")
  elif estado == "YA_EN_DESTINO":
    st.warning(f"Ya posee la moneda {p['cr_d']} en {p['ex_o']}. Elija otro destino.")
  elif estado == "SIN_CONEXION":
    st.error("Validación Kruskal: No existe conectividad estructural hacia el destino.")
  elif estado == "SIN_RUTA":
    st.error("Existe conectividad, pero no se encontró un camino válido con saldo remanente suficiente.")
  elif estado == "OK":
    r = ev["r"]
    det_opt = engine.obtener_detalles_ruta(G, r["ruta"], p["monto"])
    det_alt = engine.obtener_detalles_ruta(G, r["alternativa"], p["monto"])

    camino_dir = engine.ruta_directa(G, ev["origenes"][:1], ev["destinos"])
    res_dir = engine._resultado(G, camino_dir, p["monto"], 0, 0.0) if camino_dir else None
    det_dir = engine.obtener_detalles_ruta(G, res_dir, p["monto"])

    if ev["vencidos"]:
      st.warning(f"Aviso: Datos de mercado con {engine.formatear_antiguedad(ev['edad_seg'])} de antigüedad.")

    # =========================================================================
    # VISTA 1: MODO LITE (Solo lo esencial para transferir)
    # =========================================================================
    if not es_pro:
      st.subheader("Resumen de tu Operación")
      
      c1, c2, c3, c4 = st.columns(4)
      c1.metric("Monto a Enviar", f"${p['monto']:,.2f} USD")
      c2.metric("Comisión Total (Fees)", engine.formatear_usd(det_opt["total"]))
      c3.metric("Tiempo Total Estimado", engine.formatear_min(det_opt["tiempo_ms"]))
      c4.metric("Recibirás en Destino", det_opt["final_txt"])

      # Tarjeta de ahorro limpia
      if det_dir and det_opt:
        ahorro = det_dir["total"] - det_opt["total"]
        if ahorro > 1e-9 and det_dir["total"] > 0:
          pct = (ahorro / det_dir["total"]) * 100
          st.success(
              f"**Ruta recomendada:** Al optimizar la ruta te ahorras **{engine.formatear_usd(ahorro)}** "
              f"({pct:.1f}% menos en comisiones que haciendo una transferencia directa tradicional)."
          )

      st.markdown("### Pasos a Seguir para la Transferencia")
      df_simple = pd.DataFrame(det_opt["pasos"])
      df_simple["Costo (USD)"] = df_simple["Costo_USD"].map(engine.formatear_usd)
      df_simple["Tiempo Estimado"] = df_simple["Tiempo_ms"].map(engine.formatear_min)
      
      st.dataframe(
          df_simple[["Paso", "Tipo", "Detalle", "Red", "Costo (USD)", "Tiempo Estimado"]],
          use_container_width=True,
          hide_index=True,
      )

      st.caption("Para ver la topología completa del grafo, árboles de expansión mínima (Kruskal) y alternativas, activa el **Modo Pro** en la barra lateral.")

    # =========================================================================
    # VISTA 2: MODO PRO (Métricas avanzadas, Kruskal, A*, Grafo y Datasets)
    # =========================================================================
    else:
      tab_res, tab_graph, tab_data = st.tabs([
          "Resultados y Optimización",
          "Topología de Grafo",
          "Explorador de Datos",
      ])

      with tab_res:
        # Métricas de infraestructura
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Nodos en Grafo |V|", f"{G.number_of_nodes():,}")
        col_m2.metric("Aristas |E|", f"{G.number_of_edges():,}")
        n_ex = G.graph.get("n_exchanges", len(exchanges_disponibles))
        sin_t = len(G.graph.get("sin_tiempo", []))
        col_m3.metric("Exchanges CCXT", f"{n_ex - sin_t} / {n_ex}")
        col_m4.metric("Conexión Kruskal", "Conectado", delta="True (Union-Find)")

        # Kruskal MST
        fee_p, t_p, fee_t, t_t = r["topologia"]
        with st.expander("Validación Estructural con Kruskal (MST)", expanded=True):
          st.info(
              f"**1. MST por Precio:** Tarifa: ${fee_p:,.2f} USD | Tiempo: {engine.formatear_min(t_p)}  \n"
              f"**2. MST por Tiempo:** Tarifa: ${fee_t:,.2f} USD | Tiempo: {engine.formatear_min(t_t)}  \n"
              f"*Balance:* Optimizar por tiempo ahorra {engine.formatear_min(t_p - t_t)} con un sobrecosto global de ${fee_t - fee_p:,.2f} USD."
          )

        # Comparativa 3 Rutas
        st.markdown("#### Comparativa de Enrutamiento")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
          st.markdown("##### Ruta Directa (Sin A*)")
          if det_dir:
            st.write(f"• Pasos: {len(det_dir['pasos'])}")
            st.write(f"• Total Fees: **{engine.formatear_usd(det_dir['total'])}**")
            st.write(f"• Tiempo estimado: {engine.formatear_min(det_dir['tiempo_ms'])}")
            st.write(f"• Monto Final: {det_dir['final_txt']}")
          else:
            st.write("No disponible")

        with col_r2:
          st.markdown("##### Ruta Óptima (A*)")
          if det_opt:
            st.write(f"• Pasos: {len(det_opt['pasos'])}")
            st.write(f"• Total Fees: **{engine.formatear_usd(det_opt['total'])}**")
            st.write(f"• Tiempo estimado: {engine.formatear_min(det_opt['tiempo_ms'])}")
            st.write(f"• Monto Final: {det_opt['final_txt']}")
          else:
            st.write("No disponible")

        with col_r3:
          st.markdown("##### Ruta Alternativa (A*)")
          if det_alt:
            st.write(f"• Pasos: {len(det_alt['pasos'])}")
            st.write(f"• Total Fees: **{engine.formatear_usd(det_alt['total'])}**")
            st.write(f"• Tiempo estimado: {engine.formatear_min(det_alt['tiempo_ms'])}")
            st.write(f"• Monto Final: {det_alt['final_txt']}")
          else:
            st.caption("No existe ruta disjunta secundaria")

        if det_dir and det_opt:
          ahorro = det_dir["total"] - det_opt["total"]
          if ahorro > 1e-9 and det_dir["total"] > 0:
            pct = (ahorro / det_dir["total"]) * 100
            st.success(f"**¡Ahorro total de {engine.formatear_usd(ahorro)} ({pct:.1f}% de reducción en comisiones con A*)!**")

        st.markdown("#### Detalle de Pasos (Ruta Óptima)")
        df_p_opt = pd.DataFrame(det_opt["pasos"])
        df_p_opt["Costo_USD"] = df_p_opt["Costo_USD"].map(engine.formatear_usd)
        df_p_opt["Tiempo"] = df_p_opt["Tiempo_ms"].map(engine.formatear_min)
        st.dataframe(
            df_p_opt[["Paso", "Tipo", "Detalle", "Medio", "Red", "Costo_USD", "Tiempo"]],
            use_container_width=True,
            hide_index=True,
        )

        if det_alt:
          st.markdown("#### Detalle de Pasos (Ruta Alternativa)")
          df_p_alt = pd.DataFrame(det_alt["pasos"])
          df_p_alt["Costo_USD"] = df_p_alt["Costo_USD"].map(engine.formatear_usd)
          df_p_alt["Tiempo"] = df_p_alt["Tiempo_ms"].map(engine.formatear_min)
          st.dataframe(
              df_p_alt[["Paso", "Tipo", "Detalle", "Medio", "Red", "Costo_USD", "Tiempo"]],
              use_container_width=True,
              hide_index=True,
          )

      with tab_graph:
        st.markdown("#### Visualización Topológica en Graphviz")
        st.caption("Línea naranja continua: Ruta Óptima (A*) | Línea azul discontinua: Ruta Alternativa (A*)")
        dot = engine.generar_dot_graphviz(G, r["ruta"], r.get("alternativa"))
        st.graphviz_chart(dot, use_container_width=True)

      with tab_data:
        st.markdown("#### Datasets del Mercado")
        c_d1, c_d2 = st.columns(2)
        with c_d1:
          st.write(f"**nodos.csv** ({G.number_of_nodes():,} filas):")
          st.dataframe(pd.read_csv(engine.CSV_NODOS).head(100), height=350)
        with c_d2:
          st.write("**aristas.csv** (Precios y Tarifas de Retiro):")
          st.dataframe(pd.read_csv(engine.CSV_ARISTAS).head(100), height=350)

else:
  st.info("Configura los parámetros en el panel lateral y presiona **Optimizar Transferencia**.")