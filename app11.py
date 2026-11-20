import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import shap

from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# RUTA DEL DATASET: se busca en la misma carpeta que este script,
# sin importar desde qué carpeta se ejecute "streamlit run".
# ------------------------------------------------------------
NOMBRES_CSV = ["Trabajo modelamiento DataSetFRANCIA.csv", "DataSetFRANCIA.csv"]
CARPETA_APP = os.path.dirname(os.path.abspath(__file__))

def leer_dataset():
    candidatos = []
    for carpeta in [CARPETA_APP, os.getcwd()]:
        candidatos += [os.path.join(carpeta, n) for n in NOMBRES_CSV]
        # cualquier CSV cuyo nombre contenga "FRANCIA"
        candidatos += sorted(os.path.join(carpeta, f) for f in os.listdir(carpeta)
                             if f.lower().endswith(".csv") and "francia" in f.lower())
    for ruta in candidatos:
        if os.path.exists(ruta):
            return pd.read_csv(ruta, low_memory=False)
    st.error(f"No se encontró el dataset de Francia (.csv). Copia el archivo en la misma carpeta que la app:\n{CARPETA_APP}")
    st.stop()


# ============================================================
# 0. CONFIGURACIÓN GENERAL DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Análisis de Data Set de un Retail de Francia",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .best-model-badge {
        display: inline-block; background-color: #16a34a; color: white;
        padding: 4px 12px; border-radius: 999px; font-size: 0.88rem; font-weight: 600;
    }
    .inventory-badge {
        display: inline-block; background-color: #2563eb; color: white;
        padding: 4px 12px; border-radius: 999px; font-size: 0.88rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTES Y FUNCIONES BASE (PRONÓSTICO E INVENTARIO)
# ============================================================
TARGET_CATS = ["Soda", "Chocolate", "Energy drink", "Water", "Toothpaste"]
N_TEST = 6
N_FORECAST = 6
FECHA_CORTE_MODELO = pd.Timestamp("2023-06-30")   # usado por inventario (datos diarios)
FECHA_CORTE_MENSUAL = pd.Timestamp("2023-06-01")  # usado por pronóstico (datos mensuales)

def calcular_mae(y_real, y_pred): return float(np.mean(np.abs(np.asarray(y_real) - np.asarray(y_pred))))
def calcular_mse(y_real, y_pred): return float(np.mean((np.asarray(y_real) - np.asarray(y_pred)) ** 2))
def calcular_mape_operativo(y_real, y_pred):
    y_r, y_p = np.asarray(y_real, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_r != 0
    return float(np.mean(np.abs((y_r[mask] - y_p[mask]) / y_r[mask])) * 100) if np.any(mask) else np.nan
def calcular_mape_comercial(y_real, y_pred):
    y_r, y_p = np.asarray(y_real, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_p != 0
    return float(np.mean(np.abs((y_r[mask] - y_p[mask]) / y_p[mask])) * 100) if np.any(mask) else np.nan

# ------------------------------------------------------------
# MODELOS DE PRONÓSTICO (versión mejorada, de app10_proyeccion.py)
#  - Holt-Winters: aditivo SIN tendencia, parámetros α y γ optimizados
#    (las series no muestran tendencia; solo estacionalidad anual estable).
#  - ARIMA: ARIMA(1,0,0) con constante + variables dummy mensuales
#    (regresión con errores ARIMA; el ARIMA(2,0,1) anterior no tenía estacionalidad).
#  - SARIMA(1,0,0)(0,1,1)12: se estima con la clase SARIMAX de statsmodels pero SIN
#    variables exógenas, por lo que corresponde a un SARIMA (sin doble diferenciación).
#  - Naive estacional: referencia mínima que todo modelo debe superar.
#  - Selección del mejor modelo por validación cruzada con origen móvil.
# ------------------------------------------------------------
MODELOS = ["Holt-Winters", "ARIMA", "SARIMA"]
ORDEN_ARIMA = (1, 0, 0)
ORDEN_SARIMAX = (1, 0, 0)
ORDEN_SARIMAX_EST = (0, 1, 1, 12)
MIN_TRAIN_CV = 24  # Holt-Winters estacional necesita al menos 2 ciclos completos

def _serie(d):
    s = pd.Series(d, dtype=float)
    s.index = pd.DatetimeIndex(s.index)
    s.index.freq = "MS"
    return s

def _dummies_mes(idx):
    return pd.DataFrame({f"mes_{k}": (idx.month == k).astype(float) for k in range(2, 13)}, index=idx)

def _fechas_futuras(y, h):
    return pd.date_range(y.index[-1] + pd.DateOffset(months=1), periods=h, freq="MS")

def ajustar_hw(y, h):
    fit = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=12,
                               initialization_method="estimated").fit(optimized=True)
    p = {"alpha": float(fit.params["smoothing_level"]), "beta": 0.0,
         "gamma": float(fit.params["smoothing_seasonal"])}
    return np.asarray(fit.forecast(h)), p

def ajustar_arima_mensual(y, h):
    fit = ARIMA(y, exog=_dummies_mes(y.index), order=ORDEN_ARIMA, trend="c").fit()
    pred = fit.forecast(h, exog=_dummies_mes(_fechas_futuras(y, h)))
    return np.asarray(pred), "(1,0,0) + dummies mes", {"aic": float(fit.aic)}

def ajustar_sarimax(y, h):
    fit = SARIMAX(y, order=ORDEN_SARIMAX, seasonal_order=ORDEN_SARIMAX_EST).fit(disp=False)
    return np.asarray(fit.forecast(h))

def naive_estacional(y, h):
    return np.asarray(y.iloc[-12:].values[:h], dtype=float)

def pronosticar_todos(y, h):
    pred_hw, p_hw = ajustar_hw(y, h)
    pred_ar, orden_ar, info_ar = ajustar_arima_mensual(y, h)
    pred_sar = ajustar_sarimax(y, h)
    preds = {"Holt-Winters": pred_hw, "ARIMA": pred_ar, "SARIMA": pred_sar,
             "Naive estacional": naive_estacional(y, h)}
    return {k: np.maximum(v, 0) for k, v in preds.items()}, p_hw, orden_ar, info_ar

def mape_validacion_cruzada(y, h_max):
    """MAPE promedio con origen móvil: se reentrena en cada origen y se pronostica
    hasta h_max meses, sin pasar el último dato disponible.
    Devuelve (MAPE operativo CV, MAPE comercial CV). El operativo es el criterio de
    selección del modelo; el comercial se añade para el módulo de inventario."""
    errores = {m: [] for m in MODELOS + ["Naive estacional"]}
    errores_com = {m: [] for m in MODELOS + ["Naive estacional"]}  # [CAMBIO INVENTARIO]
    for corte in range(MIN_TRAIN_CV, len(y)):
        y_tr, y_te = y.iloc[:corte], y.iloc[corte:corte + h_max]
        preds, _, _, _ = pronosticar_todos(y_tr, len(y_te))
        for m, pr in preds.items():
            errores[m].append(calcular_mape_operativo(y_te, pr))
            errores_com[m].append(calcular_mape_comercial(y_te, pr))
    return ({m: float(np.mean(v)) for m, v in errores.items()},
            {m: float(np.mean(v)) for m, v in errores_com.items()})

@st.cache_data
def ajustar_modelos_validacion(_train_dict, _test_dict, cats):
    error_summary = []
    for cat in cats:
        y_tr, y_te = _serie(_train_dict[cat]), _serie(_test_dict[cat])
        predicciones, _, orden_ar, params_ar = pronosticar_todos(y_tr, N_TEST)

        metricas = {mod: {"MAE": calcular_mae(y_te, pr), "MSE": calcular_mse(y_te, pr),
                          "MAPE Op": calcular_mape_operativo(y_te, pr), "MAPE Com": calcular_mape_comercial(y_te, pr)}
                    for mod, pr in predicciones.items()}

        # Selección robusta: MAPE promedio de la validación cruzada con origen móvil
        y_total = pd.concat([y_tr, y_te])
        cv, cv_com = mape_validacion_cruzada(y_total, N_TEST)
        mejor = min(MODELOS, key=lambda m: cv[m])

        error_summary.append({
            "Subcategoria": cat, "Mejor modelo": mejor,
            "HW MAPE Op (%)": round(metricas["Holt-Winters"]["MAPE Op"], 2), "HW MAPE Com (%)": round(metricas["Holt-Winters"]["MAPE Com"], 2),
            "HW MAE": round(metricas["Holt-Winters"]["MAE"], 2), "HW MSE": round(metricas["Holt-Winters"]["MSE"], 2),
            "ARIMA Orden": orden_ar, "ARIMA AIC": round(params_ar["aic"], 2),
            "ARIMA MAPE Op (%)": round(metricas["ARIMA"]["MAPE Op"], 2), "ARIMA MAPE Com (%)": round(metricas["ARIMA"]["MAPE Com"], 2),
            "ARIMA MAE": round(metricas["ARIMA"]["MAE"], 2), "ARIMA MSE": round(metricas["ARIMA"]["MSE"], 2),
            "SARIMA MAPE Op (%)": round(metricas["SARIMA"]["MAPE Op"], 2), "SARIMA MAPE Com (%)": round(metricas["SARIMA"]["MAPE Com"], 2),
            "SARIMA MAE": round(metricas["SARIMA"]["MAE"], 2), "SARIMA MSE": round(metricas["SARIMA"]["MSE"], 2),
            "Naive MAPE Op (%)": round(metricas["Naive estacional"]["MAPE Op"], 2),
            "HW MAPE CV (%)": round(cv["Holt-Winters"], 2), "ARIMA MAPE CV (%)": round(cv["ARIMA"], 2),
            "SARIMA MAPE CV (%)": round(cv["SARIMA"], 2), "Naive MAPE CV (%)": round(cv["Naive estacional"], 2),
            "MAPE Operativo mejor (%)": round(metricas[mejor]["MAPE Op"], 2), "MAPE Comercial mejor (%)": round(metricas[mejor]["MAPE Com"], 2),
            # [CAMBIO INVENTARIO] error del modelo elegido con el MISMO criterio con que se eligió (CV)
            "MAPE CV Operativo mejor (%)": round(cv[mejor], 2), "MAPE CV Comercial mejor (%)": round(cv_com[mejor], 2),
        })
    return pd.DataFrame(error_summary)

@st.cache_data
def calcular_parametros_hw(_full_dict, cats):
    hw_params = []
    for cat in cats:
        _, p_hw = ajustar_hw(_serie(_full_dict[cat]), 1)
        hw_params.append({"Subcategoria": cat, "Alpha (α - Nivel)": p_hw["alpha"],
                          "Gamma (γ - Estacionalidad)": p_hw["gamma"]})
    return pd.DataFrame(hw_params)

@st.cache_data
def proyectar_futuro(_full_dict, cats, n_forecast):
    future_dates = pd.date_range("2023-07-01", periods=n_forecast, freq="MS")
    futuros = {"Holt-Winters": {}, "ARIMA": {}, "SARIMA": {}}
    for cat in cats:
        preds, _, _, _ = pronosticar_todos(_serie(_full_dict[cat]), n_forecast)
        for m in MODELOS:
            futuros[m][cat] = np.round(preds[m], 0).astype(int)
    return future_dates, futuros

# ------------------------------------------------------------
# FUNCIONES DE INVENTARIO (sin cambios de lógica)
# ------------------------------------------------------------
@st.cache_data
def calcular_estadisticas_inventario(_df, cats, fecha_corte):
    hist = _df[_df["date"] <= fecha_corte].copy()
    calendario = pd.date_range(hist["date"].min().normalize(), fecha_corte, freq="D")
    filas = []
    for cat in cats:
        sub = hist[hist["subcategory"] == cat].copy()
        dem_dia = sub.groupby("date")["units_sold"].sum().reindex(calendario, fill_value=0)
        lead = sub["lead_time_days"].dropna().astype(float)
        filas.append({"Subcategoria": cat, "Demanda diaria promedio": float(dem_dia.mean()), "Desv. demanda diaria": float(dem_dia.std(ddof=1)),
                      "Lead time promedio (días)": float(lead.mean()), "Desv. lead time (días)": float(lead.std(ddof=1)) if len(lead)>1 else 0.0})
    return pd.DataFrame(filas)

def construir_politica_inventario(df_error, df_stats, futuros_modelos, nivel_min_pct, nivel_max_pct):
    # [CAMBIO INVENTARIO] el índice de error usa el MAPE de validación cruzada (criterio de selección
    # del nuevo módulo de proyección). El MAPE ene-jun 2023 se conserva solo como referencia.
    out = df_error[["Subcategoria", "Mejor modelo", "MAPE CV Operativo mejor (%)", "MAPE CV Comercial mejor (%)",
                    "MAPE Operativo mejor (%)", "MAPE Comercial mejor (%)"]].merge(df_stats, on="Subcategoria", how="left")
    out["Índice de error MAPE (%)"] = (out["MAPE CV Operativo mejor (%)"] + out["MAPE CV Comercial mejor (%)"]) / 2
    err_min, err_max = out["Índice de error MAPE (%)"].min(), out["Índice de error MAPE (%)"].max()
    out["Riesgo relativo"] = 0.0 if np.isclose(err_min, err_max) else (out["Índice de error MAPE (%)"] - err_min) / (err_max - err_min)

    ns_min, ns_max = nivel_min_pct / 100, nivel_max_pct / 100
    out["Nivel de servicio"] = ns_min + out["Riesgo relativo"] * (ns_max - ns_min)
    out["Z"] = out["Nivel de servicio"].apply(norm.ppf)
    out["Stock de seguridad"] = np.ceil(out["Z"] * np.sqrt(out["Lead time promedio (días)"] * out["Desv. demanda diaria"] ** 2 + out["Demanda diaria promedio"] ** 2 * out["Desv. lead time (días)"] ** 2)).astype(int)

    pronostico_prom, pronostico_total, inventario_prom = [], [], []
    for _, row in out.iterrows():
        # "Mejor modelo" ahora puede ser "Holt-Winters", "ARIMA" o "SARIMA" (mismas claves que futuros)
        pron = np.asarray(futuros_modelos[row["Mejor modelo"]][row["Subcategoria"]], dtype=float)
        pronostico_prom.append(float(np.mean(pron)))
        pronostico_total.append(float(np.sum(pron)))
        inventario_prom.append(float(np.mean(pron + int(row["Stock de seguridad"]))))

    out["Pronóstico mensual promedio"] = pronostico_prom
    out["Pronóstico total jul-dic"] = pronostico_total
    out["Inventario objetivo promedio"] = inventario_prom
    return out

# ------------------------------------------------------------
# CARGA DE DATOS COMPARTIDA (pronóstico + inventario)
# [CAMBIO INVENTARIO] app10_proyeccion.py ya no leía "lead_time_days" ni devolvía
# las estadísticas de inventario; se reincorporan aquí para que el módulo 2 funcione.
# ------------------------------------------------------------
@st.cache_data
def load_data_forecast():
    df = leer_dataset()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["lead_time_days"] = pd.to_numeric(df["lead_time_days"], errors="coerce")  # [CAMBIO INVENTARIO]
    # Solo se exigen fecha, subcategoría y unidades: una venta sin lead time sigue siendo demanda.
    # El lead time se depura dentro de calcular_estadisticas_inventario (dropna).
    df = df.dropna(subset=["date", "subcategory", "units_sold"])

    cats_disponibles = [c for c in TARGET_CATS if c in set(df["subcategory"].unique())]
    demanda = df.pivot_table(index=pd.Grouper(key="date", freq="MS"), columns="subcategory", values="units_sold", aggfunc="sum").fillna(0)
    demanda = demanda.asfreq("MS", fill_value=0)

    demanda_estudio = demanda.loc[:FECHA_CORTE_MENSUAL, cats_disponibles]
    train, test = demanda_estudio.iloc[:-N_TEST], demanda_estudio.iloc[-N_TEST:]

    df_err_all = ajustar_modelos_validacion({c: train[c].to_dict() for c in cats_disponibles}, {c: test[c].to_dict() for c in cats_disponibles}, cats_disponibles)
    full_dict = {c: demanda_estudio[c].to_dict() for c in cats_disponibles}
    df_hw_all = calcular_parametros_hw(full_dict, cats_disponibles)
    future_dates, futuros = proyectar_futuro(full_dict, cats_disponibles, N_FORECAST)
    df_inv_stats = calcular_estadisticas_inventario(df, cats_disponibles, FECHA_CORTE_MODELO)  # [CAMBIO INVENTARIO]

    return demanda, cats_disponibles, df_err_all, df_hw_all, future_dates, futuros, df_inv_stats


# ============================================================
# MÓDULO 1: PROYECCIÓN DE LA DEMANDA (versión de app10_proyeccion.py)
# ============================================================
def mostrar_modulo_proyeccion(df_err_all, df_hw_all, futuros, selected_cat, future_dates, demanda):
    st.title("📊 Pronóstico de Demanda")
    st.caption("Validación de modelos temporales y proyección para los próximos 6 meses.")
    tab_resumen, tab_detalle = st.tabs(["🌍 Resumen general", "🔍 Detalle por subcategoría"])

    with tab_resumen:
        st.subheader("⚙️ Parámetros de Holt-Winters — todas las subcategorías")
        st.caption("Holt-Winters aditivo sin tendencia: las series no presentan tendencia, por lo que solo se estiman α (nivel) y γ (estacionalidad).")
        st.dataframe(df_hw_all.style.format({c: "{:.8f}" for c in df_hw_all.columns[1:]}), use_container_width=True, hide_index=True)
        st.markdown("---")

        st.subheader("🎯 Comparativa de errores (MAPE) — todas las subcategorías")
        st.caption("Selección del mejor modelo: menor MAPE promedio en validación cruzada con origen móvil (CV). "
                   "El naive estacional (mismo mes del año anterior) es la referencia mínima.")
        df_err_show = df_err_all[["Subcategoria", "HW MAPE CV (%)", "ARIMA MAPE CV (%)", "SARIMA MAPE CV (%)", "Naive MAPE CV (%)",
                                  "HW MAPE Op (%)", "ARIMA MAPE Op (%)", "SARIMA MAPE Op (%)", "Naive MAPE Op (%)",
                                  "Mejor modelo", "MAPE Operativo mejor (%)", "MAPE Comercial mejor (%)"]].copy()
        estilo = df_err_show.style.format({c: "{:.2f}%" for c in df_err_show.columns if "%" in c}).highlight_min(subset=["HW MAPE CV (%)", "ARIMA MAPE CV (%)", "SARIMA MAPE CV (%)"], axis=1, props="background-color: #c8f2d4; color: #111111; font-weight: 700;")
        st.dataframe(estilo, use_container_width=True, hide_index=True)

    with tab_detalle:
        st.subheader(f"📌 Diagnóstico y proyección: {selected_cat}")
        fila = df_err_all[df_err_all["Subcategoria"] == selected_cat].iloc[0]
        mejor = fila["Mejor modelo"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modelo seleccionado", mejor)
        c2.metric("MAPE operativo", f"{fila['MAPE Operativo mejor (%)']:.2f}%")
        c3.metric("MAPE comercial", f"{fila['MAPE Comercial mejor (%)']:.2f}%")
        c4.metric("Orden ARIMA", fila["ARIMA Orden"])

        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.write("**⚙️ Parámetros Holt-Winters**")
            fila_hw = df_hw_all[df_hw_all["Subcategoria"] == selected_cat].iloc[0]
            st.metric("Alpha (α — Nivel)", f"{fila_hw['Alpha (α - Nivel)']:.4f}")
            st.caption("Modelo sin componente de tendencia (no se estima β).")
            st.metric("Gamma (γ — Estacionalidad)", f"{fila_hw['Gamma (γ - Estacionalidad)']:.4f}")
        with col2:
            st.write("**🎯 Errores de validación (ene-jun 2023)**")
            tabla_err_cat = pd.DataFrame({"Modelo": ["Holt-Winters", "ARIMA", "SARIMA"], "MAE": [fila["HW MAE"], fila["ARIMA MAE"], fila["SARIMA MAE"]], "MSE": [fila["HW MSE"], fila["ARIMA MSE"], fila["SARIMA MSE"]], "MAPE operativo (%)": [fila["HW MAPE Op (%)"], fila["ARIMA MAPE Op (%)"], fila["SARIMA MAPE Op (%)"]], "MAPE comercial (%)": [fila["HW MAPE Com (%)"], fila["ARIMA MAPE Com (%)"], fila["SARIMA MAPE Com (%)"]], "MAPE CV (%)": [fila["HW MAPE CV (%)"], fila["ARIMA MAPE CV (%)"], fila["SARIMA MAPE CV (%)"]]})
            st.dataframe(tabla_err_cat.style.format({c: "{:.2f}" for c in tabla_err_cat.columns[1:3]}).format({c: "{:.2f}%" for c in tabla_err_cat.columns[3:]}).highlight_min(subset=["MAPE CV (%)"], props="background-color: #c8f2d4; color: #111111; font-weight: 700;"), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.write(f"**📈 Demanda real vs. proyecciones (julio–diciembre 2023): {selected_cat}**")

        col_graph, col_table = st.columns([2, 1])
        with col_graph:
            y_real = demanda[selected_cat]
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(y_real.index, y_real.values, label="Demanda Real", color="black", linewidth=2.2, marker="o", markersize=4.5)
            ax.plot(future_dates, futuros["Holt-Winters"][selected_cat], "--^", label="Holt-Winters", color="#2563eb", linewidth=1.5)
            ax.plot(future_dates, futuros["ARIMA"][selected_cat], "--o", label="ARIMA", color="#9ca3af", linewidth=1.5)
            ax.plot(future_dates, futuros["SARIMA"][selected_cat], "--s", label="SARIMA", color="#dc2626", linewidth=2)
            ax.plot(future_dates, futuros[mejor][selected_cat], "-", color="#16a34a", linewidth=3.5, alpha=0.35, label=f"🏆 Recomendado: {mejor}")
            ax.set_xlim(max(y_real.index.max(), future_dates.max()) - pd.DateOffset(months=18), max(y_real.index.max(), future_dates.max()))
            ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=8); plt.xticks(rotation=45)
            st.pyplot(fig); plt.close(fig)

        with col_table:
            st.write("**Pronóstico recomendado**")
            st.dataframe(pd.DataFrame({"Unidades Pronosticadas": futuros[mejor][selected_cat]}, index=future_dates.strftime("%Y-%m")), use_container_width=True)


# ============================================================
# MÓDULO 2: MAPE E INVENTARIO
# ============================================================
def mostrar_modulo_inventario(df_politica, futuros, selected_cat, future_dates):
    st.title("📦 Métricas de Inventario (MAPE y Stock)")
    st.info("El MAPE ajusta el nivel de servicio: un producto con mayor error recibe mayor nivel de servicio, lo que incrementa Z y el stock de seguridad.")
    # [CAMBIO INVENTARIO] aclaración del criterio, coherente con el nuevo módulo de proyección
    st.caption("El mejor modelo se elige por MAPE en validación cruzada con origen móvil (CV), y ese mismo MAPE CV "
               "(operativo y comercial) es el que define el nivel de servicio. El MAPE ene–jun 2023 se muestra solo como referencia.")

    fila_inv = df_politica[df_politica["Subcategoria"] == selected_cat].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mejor modelo", fila_inv["Mejor modelo"])
    c2.metric("MAPE Op. (CV)", f"{fila_inv['MAPE CV Operativo mejor (%)']:.2f}%",
              help=f"MAPE operativo ene–jun 2023: {fila_inv['MAPE Operativo mejor (%)']:.2f}%")
    c3.metric("MAPE Com. (CV)", f"{fila_inv['MAPE CV Comercial mejor (%)']:.2f}%",
              help=f"MAPE comercial ene–jun 2023: {fila_inv['MAPE Comercial mejor (%)']:.2f}%")
    c4.metric("Z asignado", f"{fila_inv['Z']:.3f}")
    c5.metric("Stock de seguridad", f"{int(fila_inv['Stock de seguridad']):,} u.")

    st.write("")
    st.markdown("#### 🔎 Variables que determinan el stock de seguridad")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Demanda diaria promedio (D)", f"{fila_inv['Demanda diaria promedio']:,.2f} u./día")
    d2.metric("Desv. demanda diaria (σD)", f"{fila_inv['Desv. demanda diaria']:,.2f} u./día")
    d3.metric("Lead time promedio (L)", f"{fila_inv['Lead time promedio (días)']:,.2f} días")
    d4.metric("Desv. lead time (σL)", f"{fila_inv['Desv. lead time (días)']:,.2f} días")

    # Análisis avanzado de Varianza
    termino_demanda = fila_inv["Lead time promedio (días)"] * (fila_inv["Desv. demanda diaria"] ** 2)
    termino_lead = (fila_inv["Demanda diaria promedio"] ** 2) * (fila_inv["Desv. lead time (días)"] ** 2)
    var_total_lt = termino_demanda + termino_lead
    sigma_durante_lt = np.sqrt(var_total_lt)
    aporte_demanda = (termino_demanda / var_total_lt * 100) if var_total_lt > 0 else 0.0
    aporte_lead = (termino_lead / var_total_lt * 100) if var_total_lt > 0 else 0.0

    with st.expander("Ver diagnóstico de variabilidad del stock de seguridad"):
        e1, e2, e3 = st.columns(3)
        e1.metric("Variabilidad durante el lead time", f"{sigma_durante_lt:,.2f} u.")
        e2.metric("Aporte de variabilidad de demanda", f"{aporte_demanda:.1f}%")
        e3.metric("Aporte de variabilidad del lead time", f"{aporte_lead:.1f}%")

    st.markdown("---")
    st.subheader("🌍 Resumen de inventario — todos los productos")
    df_res = df_politica.copy()
    df_res["Nivel de servicio"] = (df_res["Nivel de servicio"]*100).map("{:.2f}%".format)
    df_res["Z"] = df_res["Z"].map("{:.3f}".format)
    st.dataframe(df_res.drop(columns=["Índice de error MAPE (%)", "Riesgo relativo"]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"📅 Inventario objetivo mensual — {selected_cat}")
    ss_cat = int(fila_inv["Stock de seguridad"])
    pron_cat = np.asarray(futuros[fila_inv["Mejor modelo"]][selected_cat], dtype=int)

    col_g, col_t = st.columns([1.6, 1])
    with col_g:
        fig2, ax2 = plt.subplots(figsize=(8, 4.2))
        ax2.plot(future_dates, pron_cat, marker="o", linewidth=2, label="Demanda pronosticada")
        ax2.plot(future_dates, pron_cat + ss_cat, marker="s", linewidth=2.5, label="Inventario objetivo")
        ax2.fill_between(future_dates, pron_cat, pron_cat + ss_cat, alpha=0.15, label="Stock de seguridad")
        ax2.grid(True, linestyle=":", alpha=0.6); ax2.legend(); plt.xticks(rotation=45)
        st.pyplot(fig2); plt.close(fig2)
    with col_t:
        st.dataframe(pd.DataFrame({"Mes": future_dates.strftime("%Y-%m"), "Pronóstico": pron_cat, "SS": ss_cat, "Inv Objetivo": pron_cat + ss_cat}), use_container_width=True, hide_index=True)


# ============================================================
# MÓDULO 3: PCA (LIMPIEZA / IMPORTANCIA DE VARIABLES)
# Código de Streamlitspca.py sin cambios en datos ni cálculos; solo se envolvió
# en una función para el menú y se quitó st.set_page_config (ya está al inicio).
# ============================================================
@st.cache_data
def cargar_datos_pca():
    # Carga el dataset de la misma carpeta o permite subirlo
    try:
        df = pd.read_csv('Trabajo modelamiento DataSetFRANCIA.csv', sep=None, engine='python')
        return df.dropna()
    except FileNotFoundError:
        return None

def mostrar_modulo_pca():
    st.title("📊 Análisis de Componentes Principales (PCA): Estructura Comercial")
    st.markdown("""
    Esta aplicación presenta el diagnóstico multivariado de las variables comerciales y operativas, 
    evidenciando el proceso de depuración dimensional basado en el aporte de varianza y cargas factoriales (*loadings*).
    """)

    # =========================================================
    # 1. CARGA Y PREPARACIÓN DE DATOS
    # =========================================================
    df = cargar_datos_pca()

    if df is None:
        archivo_subido = st.sidebar.file_uploader("Sube 'Trabajo modelamiento DataSetFRANCIA.csv'", type=["csv"])
        if archivo_subido is not None:
            df = pd.read_csv(archivo_subido, sep=None, engine='python').dropna()
        else:
            st.warning("⚠️ No se encontró 'dataset modelos.csv' en la carpeta. Súbelo en el panel izquierdo.")
            st.stop()

    # Diccionario completo de variables
    diccionario_completo = {
        'units_sold': 'Unidades Vendidas',
        'list_price': 'Precio de Lista ($)',
        'discount_pct': 'Descuento (%)',
        'net_sales': 'Ventas Netas ($)',
        'purchase_cost': 'Costo de Compra ($)',
        'margin_pct': 'Margen de Ganancia (%)',
        'lead_time_days': 'Lead time (días)',
        'stock_on_hand': 'Stock (und)'
    }

    # =========================================================
    # CONTROLES EN LA BARRA LATERAL (SIDEBAR)
    # =========================================================
    st.sidebar.header("Configuración")

    escenario = st.sidebar.radio(
        "Seleccione la etapa del análisis:",
        [
            "1. Diagnóstico Inicial (8 Variables)",
            "2. Modelo Depurado (6 Variables Comerciales)"
        ]
    )

    muestra_size = st.sidebar.slider("Tamaño de muestra visual en Biplot:", 500, min(5000, len(df)), 2000, step=500)
    escala_flechas = st.sidebar.slider("Escala visual de vectores (Loadings):", 2.0, 5.0, 3.5, step=0.5)

    # Filtrar variables según el escenario elegido
    if "1." in escenario:
        encabezados = diccionario_completo.copy()
        st.info("📌 **Etapa 1 (Diagnóstico):** Se incluyen todas las variables disponibles (comerciales + logísticas) para evaluar su peso en los componentes principales.")
    else:
        encabezados = {k: v for k, v in diccionario_completo.items() if k not in ['lead_time_days', 'stock_on_hand']}
        st.success("✅ **Etapa 2 (Depuración):** Se segregan `Lead time (días)` y `Stock (und)` por tener vectores cercanos al origen (aporte nulo a PC1 y PC2).")

    features = list(encabezados.keys())
    df_num = df[features]

    # =========================================================
    # 2. CÁLCULO DE ESTANDARIZACIÓN Y PCA
    # =========================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    pca = PCA()
    pca.fit(X_scaled)  # <-- Asegúrate de que esta línea esté presente antes de transform
    componentes = pca.transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100
    cum_var_exp = np.cumsum(var_exp)

    # =========================================================
    # 3. MÉTRICAS CLAVE (KPIs SUPERIORES)
    # =========================================================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables Activas", f"{len(features)} variables")
    col2.metric("Varianza PC1", f"{var_exp[0]:.2f}%")
    col3.metric("Varianza PC2", f"{var_exp[1]:.2f}%")
    col4.metric("Varianza Acumulada (PC1 + PC2)", f"{cum_var_exp[1]:.2f}%", 
                delta=f"{cum_var_exp[1] - 53.62:.2f}% vs Inicial" if "2." in escenario else None)

    st.markdown("---")

    # =========================================================
    # 4. PESTAÑAS DE VISUALIZACIÓN PARA LA EXPOSICIÓN
    # =========================================================
    tab_biplot, tab_scree, tab_cargas = st.tabs([
        "🎯 Biplot Comercial (Vectores)", 
        "📉 Sedimentación (Scree Plot)", 
        "📋 Tabla de Cargas (Justificación Matemática)"
    ])

    # --- PESTAÑA 1: BIPLOT ---
    with tab_biplot:
        st.subheader(f"Proyección de Datos y Vectores — {escenario}")
        
        np.random.seed(42)
        idx_muestra = np.random.choice(range(componentes.shape[0]), size=muestra_size, replace=False)
        pc1_muestra = componentes[idx_muestra, 0]
        pc2_muestra = componentes[idx_muestra, 1]
        categorias_muestra = df['category'].iloc[idx_muestra].values

        fig_biplot, ax = plt.subplots(figsize=(11, 6.5))
        
        sns.scatterplot(
            x=pc1_muestra, y=pc2_muestra, hue=categorias_muestra,
            palette='Set2', alpha=0.55, s=35, edgecolor=None, ax=ax
        )

        loadings = pca.components_.T * escala_flechas

        for i, feature in enumerate(features):
            ax.arrow(
                0, 0, loadings[i, 0], loadings[i, 1],
                color='darkred', alpha=0.9, head_width=0.08, head_length=0.08, linewidth=2
            )
            etiqueta = encabezados[feature]
            ax.text(
                loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                etiqueta, color='black', ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray', boxstyle='round,pad=0.25')
            )

        ax.set_xlim(-3.5, 4.5)
        ax.set_ylim(-2.7, 3.5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel(f'Componente Principal 1 ({var_exp[0]:.2f}%)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Componente Principal 2 ({var_exp[1]:.2f}%)', fontsize=11, fontweight='bold')
        ax.set_title('Biplot PCA: Estructura Comercial por Categoría', fontsize=13, fontweight='bold')
        ax.legend(title='Categoría', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.35)
        
        st.pyplot(fig_biplot)

    # --- PESTAÑA 2: SCREE PLOT ---
    with tab_scree:
        st.subheader("Varianza Explicada por Componente (Scree Plot)")
        
        fig_scree, ax2 = plt.subplots(figsize=(10, 4.5))
        x_pos = range(1, len(var_exp) + 1)
        
        ax2.bar(x_pos, var_exp, alpha=0.75, align='center', label='Varianza Individual (%)', color='steelblue')
        ax2.plot(x_pos, cum_var_exp, marker='o', label='Varianza Acumulada (%)', color='darkorange', linewidth=2.5)
        
        for i, txt in enumerate(cum_var_exp):
            ax2.annotate(f"{txt:.1f}%", (x_pos[i], cum_var_exp[i] + 2), ha='center', fontsize=9, fontweight='bold')

        ax2.set_title('Gráfico de Sedimentación', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Número de Componentes Principales')
        ax2.set_ylabel('Porcentaje de Varianza Explicada (%)')
        ax2.set_xticks(x_pos)
        ax2.set_ylim(0, 110)
        ax2.legend(loc='center right')
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        st.pyplot(fig_scree)

    # --- PESTAÑA 3: TABLA DE CARGAS FACTORIALES ---
    with tab_cargas:
        st.subheader("Matriz de Cargas Factoriales (Loadings) y Magnitud del Vector")
        st.write("La magnitud en el plano (PC1, PC2) mide qué tan larga es la flecha en el Biplot. Valores cercanos a 0 justifican su exclusión.")
        
        df_loadings = pd.DataFrame(
            pca.components_[:2, :].T,
            columns=['Carga PC1', 'Carga PC2'],
            index=[encabezados[f] for f in features]
        )
        # Magnitud euclidiana del vector en el plano 2D
        df_loadings['Longitud Flecha (PC1-PC2)'] = np.sqrt(df_loadings['Carga PC1']**2 + df_loadings['Carga PC2']**2)
        df_loadings = df_loadings.sort_values(by='Longitud Flecha (PC1-PC2)', ascending=False)
        
        st.dataframe(
            df_loadings.style.format("{:.4f}").background_gradient(subset=['Longitud Flecha (PC1-PC2)'], cmap='YlOrRd'),
            use_container_width=True
        )


# ============================================================
# MÓDULO 4: ELASTICIDAD
# ============================================================
@st.cache_data
def load_data_elasticity():
    df = leer_dataset()
    df = df.dropna(subset=['date', 'subcategory', 'units_sold', 'net_sales'])
    df = df[df['units_sold'] > 0].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['Month'] = df['date'].dt.to_period('M').astype(str)
    df['EffectivePrice'] = df['net_sales'] / df['units_sold']
    return df

def mostrar_modulo_elasticidad():
    st.title("📈 Hub Analítico: Modelos de Elasticidad")
    st.markdown("Plataforma interactiva adaptada al dataset de Francia para analizar la sensibilidad de precios.")

    df = load_data_elasticity()
    top_products = ['Soda', 'Chocolate', 'Energy drink', 'Toothpaste', 'Water']

    monthly = df[df['subcategory'].isin(top_products)].groupby(['subcategory', 'Month']).agg({'EffectivePrice': 'mean', 'units_sold': 'sum'}).reset_index()
    monthly['ln_P'] = np.log(monthly['EffectivePrice'])
    monthly['ln_Q'] = np.log(monthly['units_sold'])
    monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna()
    colores = {'Soda': '#1f77b4', 'Chocolate': '#8c564b', 'Energy drink': '#ff7f0e', 'Toothpaste': '#2ca02c', 'Water': '#17becf'}

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Regresión Log-Log", "2. Regresión Lineal (X-Y)", "3. Elasticidad Individual (Arco)", "4. Elasticidad Cruzada", "5. Machine Learning (SHAP)"])

    with tab1:
        st.subheader("Modelo Log-Log (Elasticidad Constante)")
        resultados_log = []
        for name in top_products:
            df_p = monthly[monthly['subcategory'] == name]
            if len(df_p) > 3 and df_p['ln_P'].nunique() > 1:
                slope, _, r_value, _, _ = stats.linregress(df_p['ln_P'], df_p['ln_Q'])
                resultados_log.append({'Producto': name, 'Unidades': int(df[df['subcategory']==name]['units_sold'].sum()), 'Elasticidad': round(slope, 2), 'R2': round(r_value**2, 4)})

        col1a, col1b = st.columns([2, 1])
        with col1a:
            fig1 = go.Figure()
            for _, row in pd.DataFrame(resultados_log).iterrows():
                fig1.add_trace(go.Scatter(x=[row['Unidades']], y=[row['Elasticidad']], mode='markers+text', name=row['Producto'], text=[row['Producto']], textposition="top center", marker=dict(size=18, color=colores.get(row['Producto'], '#17becf'))))
            fig1.add_hline(y=-1, line_dash="dash", line_color="red")
            fig1.update_layout(title="Log-Log Elasticity", xaxis_title="Unidades Vendidas", yaxis_title="Elasticidad Precio")
            st.plotly_chart(fig1, use_container_width=True)
        with col1b: st.dataframe(pd.DataFrame(resultados_log), hide_index=True)

    with tab2:
        st.subheader("Modelo de Regresión Lineal X-Y")
        resultados_xy = []
        for name in top_products:
            df_p = monthly[monthly['subcategory'] == name]
            if len(df_p) > 3 and df_p['EffectivePrice'].nunique() > 1:
                slope, _, r_value, _, _ = stats.linregress(df_p['EffectivePrice'], df_p['units_sold'])
                resultados_xy.append({'Producto': name, 'Elasticidad X-Y': round(slope * (df_p['EffectivePrice'].mean() / df_p['units_sold'].mean()), 2), 'R2': round(r_value**2, 4)})
        st.table(pd.DataFrame(resultados_xy))

    with tab3:
        st.subheader("Elasticidad de Arco (Variación Mes a Mes)")
        resultados_arco = []
        for name in top_products:
            df_p = monthly[monthly['subcategory'] == name].sort_values('Month').copy()
            df_p['Arc_Elasticity'] = df_p['units_sold'].pct_change() / df_p['EffectivePrice'].pct_change()
            resultados_arco.append({'Producto': name, 'Elasticidad Promedio (Arco)': round(df_p.replace([np.inf, -np.inf], np.nan).dropna()['Arc_Elasticity'].mean(), 2)})
        st.table(pd.DataFrame(resultados_arco))

    with tab4:
        st.subheader("Elasticidad Cruzada de la Demanda")
        pivot_p = monthly.pivot(index='Month', columns='subcategory', values='ln_P')
        pivot_q = monthly.pivot(index='Month', columns='subcategory', values='ln_Q')

        st.markdown("#### Matriz General de Canibalización y Complementariedad")
        cross_matrix = pd.DataFrame(index=top_products, columns=top_products, dtype=float)
        for target in top_products:
            for comp in top_products:
                if target != comp:
                    cross_df_heat = pd.DataFrame({'Q': pivot_q[target], 'P': pivot_p[comp]}).dropna()
                    if len(cross_df_heat) > 3: cross_matrix.loc[target, comp] = LinearRegression().fit(cross_df_heat[['P']], cross_df_heat['Q']).coef_[0]

        fig_heat = px.imshow(cross_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", color_continuous_midpoint=0, labels=dict(x="Si CAMBIA el precio de este producto...", y="...cómo afecta la DEMANDA de este", color="Coef. Cruzado"))
        fig_heat.update_xaxes(side="top")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        col4a, col4b = st.columns(2)
        target_prod = col4a.selectbox("Producto Objetivo (Demanda):", top_products, index=0)
        comp_prod = col4b.selectbox("Producto Competidor (Precio):", top_products, index=1)
        if target_prod != comp_prod:
            cross_df = pd.DataFrame({'Q': pivot_q[target_prod], 'P': pivot_p[comp_prod]}).dropna()
            if len(cross_df) > 3:
                cross_elasticity = LinearRegression().fit(cross_df[['P']], cross_df['Q']).coef_[0]
                st.metric(label=f"Elasticidad Cruzada ({target_prod} vs {comp_prod})", value=round(cross_elasticity, 3))
                if cross_elasticity > 0: st.success("Son productos SUSTITUTOS.")
                else: st.info("Son productos COMPLEMENTARIOS.")

    with tab5:
        st.subheader("Análisis de Impacto con SHAP Values (Random Forest)")
        shap_prod = st.selectbox("Seleccionar producto para entrenar el modelo:", top_products)
        df_shap = df[df['subcategory'] == shap_prod].copy()
        df_shap['Month_Num'] = df_shap['date'].dt.month

        X = df_shap[['EffectivePrice', 'discount_pct', 'promo_flag', 'Month_Num']].dropna()
        y = df_shap.loc[X.index, 'units_sold']
        if len(X) > 50:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
            X_sample = shap.sample(X, 500) if len(X) > 500 else X
            shap_values = shap.TreeExplainer(model).shap_values(X_sample)
            st.write(f"**R² del Bosque Aleatorio:** {round(model.score(X, y), 3)}")
            fig_shap, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig_shap); plt.clf()


# ============================================================
# MÓDULO 5: ESCENARIOS (PROYECCIÓN, INVENTARIO, ELASTICIDAD Y COMBINADO)
# Tres escenarios por parte: optimista, realista y pesimista.
# El realista siempre coincide con los módulos 1 y 2. Los otros dos se arman con
# variación que YA ocurrió en los datos: error del modelo (MAPE CV), proveedores
# más rápidos/lentos y elasticidad trimestral mínima/máxima.
# No modifica ningún cálculo de los módulos 1 a 4.
# ============================================================
ESCENARIOS = ["Optimista", "Realista", "Pesimista"]
COLOR_ESC = {"Optimista": "#16a34a", "Realista": "#2563eb", "Pesimista": "#dc2626"}
FACTOR_DEMANDA_BASE = {"Optimista": 1.0, "Realista": 0.0, "Pesimista": -2.0}
MIN_ENTREGAS_PROVEEDOR = 30  # proveedores con pocas entregas no son representativos

@st.cache_data
def cargar_datos_escenarios(fecha_corte):
    df = leer_dataset()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["units_sold", "net_sales", "purchase_cost", "lead_time_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    hist = df[(df["date"] <= fecha_corte) & df["subcategory"].isin(TARGET_CATS)]

    # 1) Precio efectivo (ventas netas / unidades) y costo unitario ponderado
    v = hist[hist["units_sold"] > 0].dropna(subset=["units_sold", "net_sales", "purchase_cost"])
    v = v.assign(costo_total=v["purchase_cost"] * v["units_sold"])
    g = v.groupby("subcategory")[["net_sales", "units_sold", "costo_total"]].sum()
    precios = pd.DataFrame({"Precio": g["net_sales"] / g["units_sold"], "Costo": g["costo_total"] / g["units_sold"]})

    # 2) Lead time de cada proveedor (solo los que tienen suficientes entregas)
    proveedores = (hist.dropna(subset=["lead_time_days", "supplier_id"])
                       .groupby(["subcategory", "supplier_id"])["lead_time_days"].agg(["mean", "std", "count"]).reset_index())
    proveedores = proveedores[proveedores["count"] >= MIN_ENTREGAS_PROVEEDOR]

    # 3) Elasticidad log-log con datos diarios: estimación total y rango entre trimestres
    v = v.assign(lnP=np.log(v["net_sales"] / v["units_sold"]), lnQ=np.log(v["units_sold"]))
    filas = []
    for cat, x in v.groupby("subcategory"):
        tot = stats.linregress(x["lnP"], x["lnQ"])
        trim = [stats.linregress(t["lnP"], t["lnQ"]).slope
                for _, t in x.groupby(x["date"].dt.to_period("Q")) if t["lnP"].nunique() > 1]
        filas.append({"Subcategoria": cat, "Elasticidad total": tot.slope, "R2": tot.rvalue ** 2,
                      "Elasticidad mín. trimestral": min(trim), "Elasticidad máx. trimestral": max(trim), "Trimestres": len(trim)})
    elasticidades = pd.DataFrame(filas).set_index("Subcategoria")
    return precios, proveedores, elasticidades

def _proveedor_escenario(r, proveedores, esc):
    """Lead time del escenario. Realista = promedio de todos los proveedores (módulo 2).
    Optimista / pesimista = el proveedor que exige MENOS / MÁS stock de seguridad, es decir,
    el de menor / mayor variabilidad durante el lead time: sqrt(L·σD² + D²·σL²)."""
    cat = r["Subcategoria"]
    x = proveedores[proveedores["subcategory"] == cat]
    if esc == "Realista" or x.empty:
        return "Todos (promedio)", r["Lead time promedio (días)"], r["Desv. lead time (días)"]
    D, sD = r["Demanda diaria promedio"], r["Desv. demanda diaria"]
    sigma_lt = np.sqrt(x["mean"] * sD ** 2 + D ** 2 * x["std"] ** 2)
    p = x.loc[sigma_lt.idxmin() if esc == "Optimista" else sigma_lt.idxmax()]
    return p["supplier_id"], float(p["mean"]), float(p["std"])

def escenarios_proyeccion(df_politica, futuros, precios, factores):
    filas, mensual = [], {}
    for _, r in df_politica.iterrows():
        cat = r["Subcategoria"]
        pron = np.asarray(futuros[r["Mejor modelo"]][cat], dtype=float)
        mape = r["MAPE CV Operativo mejor (%)"] / 100
        precio, costo = precios.loc[cat, "Precio"], precios.loc[cat, "Costo"]
        for esc in ESCENARIOS:
            cambio = factores[esc] * mape
            dem = pron * (1 + cambio)
            filas.append({"Subcategoria": cat, "Escenario": esc, "Modelo": r["Mejor modelo"], "MAPE CV (%)": mape * 100,
                          "Cambio demanda (%)": cambio * 100, "Ventas jul-dic (u.)": float(dem.sum()),
                          "Ventas jul-dic ($)": float(dem.sum() * precio), "Margen bruto ($)": float(dem.sum() * (precio - costo))})
            mensual[(cat, esc)] = dem
    return pd.DataFrame(filas), mensual

def escenarios_inventario(df_politica, proveedores, precios):
    filas = []
    for _, r in df_politica.iterrows():
        cat = r["Subcategoria"]
        D, sD = r["Demanda diaria promedio"], r["Desv. demanda diaria"]
        ss_actual = int(r["Stock de seguridad"])
        for esc in ESCENARIOS:
            prov, L, sL = _proveedor_escenario(r, proveedores, esc)
            sigma_lt = np.sqrt(L * sD ** 2 + D ** 2 * sL ** 2)
            ss = int(np.ceil(r["Z"] * sigma_lt))
            filas.append({"Subcategoria": cat, "Escenario": esc, "Proveedor": prov,
                          "Lead time (días)": L, "Desv. lead time (días)": sL,
                          "Nivel de servicio objetivo (%)": r["Nivel de servicio"] * 100,
                          "Stock de seguridad requerido (u.)": ss, "Cambio vs realista (u.)": ss - ss_actual,
                          "Capital en stock de seguridad ($)": ss * precios.loc[cat, "Costo"],
                          "Nivel de servicio con SS actual (%)": float(norm.cdf(ss_actual / sigma_lt) * 100)})
    return pd.DataFrame(filas)

def escenarios_elasticidad(df_politica, futuros, precios, elasticidades, cambio_precio_pct):
    p = cambio_precio_pct / 100
    filas = []
    for _, r in df_politica.iterrows():
        cat = r["Subcategoria"]
        Q0 = float(np.sum(futuros[r["Mejor modelo"]][cat]))  # unidades jul-dic del escenario realista (módulo 1)
        P0, C = precios.loc[cat, "Precio"], precios.loc[cat, "Costo"]
        e = elasticidades.loc[cat]
        margen_actual = Q0 * (P0 - C)

        def resultado(eps):
            Q1 = Q0 * (1 + p) ** eps  # elasticidad constante (modelo log-log)
            return Q1, Q1 * P0 * (1 + p), Q1 * (P0 * (1 + p) - C)

        extremos = [e["Elasticidad mín. trimestral"], e["Elasticidad máx. trimestral"]]
        extremos.sort(key=lambda eps: resultado(eps)[2], reverse=True)  # el de mejor margen es el optimista
        eps_esc = {"Optimista": extremos[0], "Realista": e["Elasticidad total"], "Pesimista": extremos[1]}
        for esc in ESCENARIOS:
            Q1, ventas, margen = resultado(eps_esc[esc])
            filas.append({"Subcategoria": cat, "Escenario": esc, "Elasticidad": eps_esc[esc],
                          "Precio actual ($)": P0, "Precio nuevo ($)": P0 * (1 + p),
                          "Unidades jul-dic": Q1, "Cambio unidades (%)": (Q1 / Q0 - 1) * 100,
                          "Ventas jul-dic ($)": ventas, "Margen bruto ($)": margen,
                          "Cambio margen vs precio actual ($)": margen - margen_actual})
    return pd.DataFrame(filas)

def escenarios_combinado(df_politica, futuros, precios, factores, proveedores):
    filas, mensual = [], {}
    for _, r in df_politica.iterrows():
        cat = r["Subcategoria"]
        pron = np.asarray(futuros[r["Mejor modelo"]][cat], dtype=float)
        mape = r["MAPE CV Operativo mejor (%)"] / 100
        plan = pron + int(r["Stock de seguridad"])  # inventario objetivo del módulo 2
        precio, costo = precios.loc[cat, "Precio"], precios.loc[cat, "Costo"]
        for esc in ESCENARIOS:
            cambio = factores[esc] * mape
            dem = pron * (1 + cambio)
            prov, L, sL = _proveedor_escenario(r, proveedores, esc)
            D, sD = r["Demanda diaria promedio"] * (1 + cambio), r["Desv. demanda diaria"] * (1 + cambio)
            ss = int(np.ceil(r["Z"] * np.sqrt(L * sD ** 2 + D ** 2 * sL ** 2)))
            exceso = float(np.maximum(plan - dem, 0).mean())
            filas.append({"Subcategoria": cat, "Escenario": esc, "Cambio demanda (%)": cambio * 100, "Proveedor": prov,
                          "Lead time (días)": L, "Ventas jul-dic (u.)": float(dem.sum()), "Ventas jul-dic ($)": float(dem.sum() * precio),
                          "Margen bruto ($)": float(dem.sum() * (precio - costo)), "Stock de seguridad requerido (u.)": ss,
                          "Inventario objetivo prom. (u./mes)": float(np.mean(dem + ss)),
                          "Faltante vs plan realista (u.)": float(np.maximum(dem - plan, 0).sum()),
                          "Exceso vs plan realista (u./mes)": exceso, "Capital inmovilizado ($/mes)": exceso * costo})
            mensual[(cat, esc)] = dem
    return pd.DataFrame(filas), mensual

def _delta(valor, base, esc, fmt):
    return None if esc == "Realista" else fmt.format(valor - base)

def _tabla(df, formatos):
    st.dataframe(df.style.format(formatos), use_container_width=True, hide_index=True)

def mostrar_modulo_escenarios(df_politica, futuros, selected_cat, future_dates, demanda, datos_esc):
    precios, proveedores, elasticidades = datos_esc
    st.title("🎲 Análisis de Escenarios: Optimista, Realista y Pesimista")
    st.caption("Tres escenarios para julio–diciembre 2023 en cada parte del trabajo. El realista coincide siempre con los módulos 1 y 2; "
               "los otros dos se arman con variaciones que ya ocurrieron en los datos.")
    fila_pol = df_politica[df_politica["Subcategoria"] == selected_cat].iloc[0]

    tab_proy, tab_inv, tab_elas, tab_comb = st.tabs(["📊 Proyección de demanda", "📦 Inventario", "📈 Elasticidad", "🔗 Combinado (demanda + inventario)"])

    # ------------------------------------------------------------------ PROYECCIÓN
    with tab_proy:
        st.subheader("📊 Escenarios de demanda")
        st.write("La demanda se mueve en múltiplos del **MAPE CV** del modelo elegido para cada producto, es decir, del error típico que ese modelo ya tuvo.")
        factores = {}
        cols = st.columns(3)
        for col, esc in zip(cols, ESCENARIOS):
            factores[esc] = col.number_input(f"{esc}: demanda (× MAPE)", -5.0, 5.0, FACTOR_DEMANDA_BASE[esc], 0.5, key=f"fac_{esc}")

        df_p, mensual_p = escenarios_proyeccion(df_politica, futuros, precios, factores)
        tot = df_p.groupby("Escenario", sort=False)[["Ventas jul-dic (u.)", "Ventas jul-dic ($)", "Margen bruto ($)"]].sum()
        base = tot.loc["Realista"]
        st.markdown("##### Total de los 5 productos")
        cols = st.columns(3)
        for col, esc in zip(cols, ESCENARIOS):
            t = tot.loc[esc]
            with col:
                st.markdown(f"**{esc}**")
                st.metric("Unidades jul–dic", f"{t['Ventas jul-dic (u.)']:,.0f} u.", _delta(t["Ventas jul-dic (u.)"], base["Ventas jul-dic (u.)"], esc, "{:+,.0f} u."))
                st.metric("Ventas jul–dic", f"${t['Ventas jul-dic ($)']:,.0f}", _delta(t["Ventas jul-dic ($)"], base["Ventas jul-dic ($)"], esc, "{:+,.0f} $"))
                st.metric("Margen bruto", f"${t['Margen bruto ($)']:,.0f}", _delta(t["Margen bruto ($)"], base["Margen bruto ($)"], esc, "{:+,.0f} $"))

        st.markdown(f"##### Detalle — {selected_cat}")
        col_g, col_t = st.columns([1.6, 1])
        with col_g:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            hist = demanda[selected_cat].loc[:FECHA_CORTE_MENSUAL].iloc[-12:]
            ax.plot(hist.index, hist.values, color="black", marker="o", markersize=4, linewidth=2, label="Demanda real (últimos 12 meses)")
            for esc in ESCENARIOS:
                ax.plot(future_dates, mensual_p[(selected_cat, esc)], marker="o", linewidth=2, color=COLOR_ESC[esc], label=esc)
            ax.fill_between(future_dates, mensual_p[(selected_cat, "Pesimista")], mensual_p[(selected_cat, "Optimista")], color="#9ca3af", alpha=0.15)
            ax.set_ylabel("Unidades"); ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=8); plt.xticks(rotation=45)
            st.pyplot(fig); plt.close(fig)
        with col_t:
            st.dataframe(pd.DataFrame({esc: np.round(mensual_p[(selected_cat, esc)], 0).astype(int) for esc in ESCENARIOS},
                                      index=future_dates.strftime("%Y-%m")), use_container_width=True)
        _tabla(df_p, {"MAPE CV (%)": "{:.2f}%", "Cambio demanda (%)": "{:+.2f}%", "Ventas jul-dic (u.)": "{:,.0f}",
                      "Ventas jul-dic ($)": "${:,.0f}", "Margen bruto ($)": "${:,.0f}"})

    # ------------------------------------------------------------------ INVENTARIO
    with tab_inv:
        st.subheader("📦 Escenarios de inventario (proveedores)")
        st.write(f"La demanda se mantiene; lo que cambia es el **proveedor**. Optimista: todo lo trae el proveedor que menos stock de seguridad exige "
                 f"(el de menor variabilidad durante el lead time); pesimista: el que más exige. Solo se consideran proveedores con al menos "
                 f"{MIN_ENTREGAS_PROVEEDOR} entregas.")
        df_i = escenarios_inventario(df_politica, proveedores, precios)
        tot = df_i.groupby("Escenario", sort=False).agg({"Stock de seguridad requerido (u.)": "sum", "Capital en stock de seguridad ($)": "sum",
                                                         "Nivel de servicio con SS actual (%)": "mean"})
        base = tot.loc["Realista"]
        st.markdown("##### Total de los 5 productos")
        cols = st.columns(3)
        for col, esc in zip(cols, ESCENARIOS):
            t = tot.loc[esc]
            with col:
                st.markdown(f"**{esc}**")
                st.metric("Stock de seguridad requerido", f"{t['Stock de seguridad requerido (u.)']:,.0f} u.",
                          _delta(t["Stock de seguridad requerido (u.)"], base["Stock de seguridad requerido (u.)"], esc, "{:+,.0f} u."), delta_color="inverse")
                st.metric("Capital en stock de seguridad", f"${t['Capital en stock de seguridad ($)']:,.0f}",
                          _delta(t["Capital en stock de seguridad ($)"], base["Capital en stock de seguridad ($)"], esc, "{:+,.0f} $"), delta_color="inverse")
                st.metric("Nivel de servicio con el SS actual", f"{t['Nivel de servicio con SS actual (%)']:.2f}%",
                          _delta(t["Nivel de servicio con SS actual (%)"], base["Nivel de servicio con SS actual (%)"], esc, "{:+.2f} pp"))

        st.markdown(f"##### Detalle — {selected_cat}")
        det = df_i[df_i["Subcategoria"] == selected_cat].set_index("Escenario")
        cols = st.columns(3)
        for col, esc in zip(cols, ESCENARIOS):
            d = det.loc[esc]
            with col:
                st.markdown(f"**{esc}** · proveedor {d['Proveedor']}")
                st.metric("Lead time (L ± σL)", f"{d['Lead time (días)']:.2f} ± {d['Desv. lead time (días)']:.2f} días")
                st.metric("Stock de seguridad requerido", f"{d['Stock de seguridad requerido (u.)']:,} u.",
                          None if esc == "Realista" else f"{d['Cambio vs realista (u.)']:+,} u.", delta_color="inverse")
                st.metric("Nivel de servicio con el SS actual", f"{d['Nivel de servicio con SS actual (%)']:.2f}%")
        malo = det.loc["Pesimista"]
        st.info(f"En el escenario pesimista, **{selected_cat}** se abastece del proveedor {malo['Proveedor']}. Con el stock de seguridad actual "
                f"({int(fila_pol['Stock de seguridad']):,} u.), el nivel de servicio pasa de {fila_pol['Nivel de servicio']*100:.2f}% a "
                f"**{malo['Nivel de servicio con SS actual (%)']:.2f}%**. Mantener el nivel de servicio objetivo requiere "
                f"{int(malo['Cambio vs realista (u.)']):+,} u. de stock de seguridad.")
        _tabla(df_i, {"Lead time (días)": "{:.2f}", "Desv. lead time (días)": "{:.2f}", "Nivel de servicio objetivo (%)": "{:.2f}%",
                      "Stock de seguridad requerido (u.)": "{:,}", "Cambio vs realista (u.)": "{:+,}",
                      "Capital en stock de seguridad ($)": "${:,.0f}", "Nivel de servicio con SS actual (%)": "{:.2f}%"})

    # ------------------------------------------------------------------ ELASTICIDAD
    with tab_elas:
        st.subheader("📈 Escenarios de elasticidad (cambio de precio)")
        st.write("Se evalúa el efecto de un cambio de precio sobre las ventas y el margen de jul–dic. Se usa una **elasticidad log-log con datos diarios** "
                 "(precio efectivo vs. unidades). Realista: la elasticidad de todo el periodo; optimista y pesimista: la elasticidad "
                 "mínima y máxima que se observó entre trimestres (la que da mejor y peor margen).")
        st.caption("Se usan datos diarios porque el precio promedio mensual casi no varía (2–5 %), y con tan poca variación la regresión mensual "
                   "del módulo 4 no logra estimar la elasticidad (R² cercano a 0). Con datos diarios los descuentos dan la variación de precio necesaria.")
        cambio_precio = st.slider("Cambio de precio a evaluar (%)", -30, 30, 5, 1)
        st.dataframe(elasticidades.reset_index().style.format({"Elasticidad total": "{:.3f}", "R2": "{:.3f}",
                     "Elasticidad mín. trimestral": "{:.3f}", "Elasticidad máx. trimestral": "{:.3f}"}), use_container_width=True, hide_index=True)

        df_e = escenarios_elasticidad(df_politica, futuros, precios, elasticidades, cambio_precio)
        tot = df_e.groupby("Escenario", sort=False)[["Unidades jul-dic", "Ventas jul-dic ($)", "Margen bruto ($)", "Cambio margen vs precio actual ($)"]].sum()
        st.markdown(f"##### Total de los 5 productos — precio {cambio_precio:+d}%")
        cols = st.columns(3)
        for col, esc in zip(cols, ESCENARIOS):
            t = tot.loc[esc]
            with col:
                st.markdown(f"**{esc}**")
                st.metric("Unidades jul–dic", f"{t['Unidades jul-dic']:,.0f} u.")
                st.metric("Ventas jul–dic", f"${t['Ventas jul-dic ($)']:,.0f}")
                st.metric("Margen bruto", f"${t['Margen bruto ($)']:,.0f}", f"{t['Cambio margen vs precio actual ($)']:+,.0f} $ vs precio actual")

        st.markdown(f"##### Detalle — {selected_cat}")
        det = df_e[df_e["Subcategoria"] == selected_cat].set_index("Escenario")
        cols = st.columns(3)
        for col, esc in zip(cols, ESCENARIOS):
            d = det.loc[esc]
            with col:
                st.markdown(f"**{esc}** · elasticidad {d['Elasticidad']:.3f}")
                st.metric("Precio", f"${d['Precio nuevo ($)']:.2f}", f"{cambio_precio:+d}%" if cambio_precio else None, delta_color="off")
                st.metric("Unidades jul–dic", f"{d['Unidades jul-dic']:,.0f} u.", f"{d['Cambio unidades (%)']:+.2f}%")
                st.metric("Margen bruto", f"${d['Margen bruto ($)']:,.0f}", f"{d['Cambio margen vs precio actual ($)']:+,.0f} $")
        _tabla(df_e, {"Elasticidad": "{:.3f}", "Precio actual ($)": "${:.2f}", "Precio nuevo ($)": "${:.2f}", "Unidades jul-dic": "{:,.0f}",
                      "Cambio unidades (%)": "{:+.2f}%", "Ventas jul-dic ($)": "${:,.0f}", "Margen bruto ($)": "${:,.0f}",
                      "Cambio margen vs precio actual ($)": "{:+,.0f}"})

    # ------------------------------------------------------------------ COMBINADO
    with tab_comb:
        st.subheader("🔗 Demanda + inventario a la vez")
        st.write("Ocurren al mismo tiempo el escenario de demanda (pestaña Proyección) y el de proveedor (pestaña Inventario). "
                 "Se compara contra el plan realista: cada mes se repone hasta el inventario objetivo del módulo 2 (pronóstico + stock de seguridad).")
        df_c, mensual_c = escenarios_combinado(df_politica, futuros, precios, factores, proveedores)
        tot = df_c.groupby("Escenario", sort=False)[["Ventas jul-dic ($)", "Margen bruto ($)", "Faltante vs plan realista (u.)", "Capital inmovilizado ($/mes)"]].sum()
        base = tot.loc["Realista"]
        st.markdown("##### Total de los 5 productos")
        cols = st.columns(3)
        for col, esc in zip(cols, ESCENARIOS):
            t = tot.loc[esc]
            with col:
                st.markdown(f"**{esc}**")
                st.metric("Ventas jul–dic", f"${t['Ventas jul-dic ($)']:,.0f}", _delta(t["Ventas jul-dic ($)"], base["Ventas jul-dic ($)"], esc, "{:+,.0f} $"))
                st.metric("Margen bruto", f"${t['Margen bruto ($)']:,.0f}", _delta(t["Margen bruto ($)"], base["Margen bruto ($)"], esc, "{:+,.0f} $"))
                st.metric("Unidades sin atender", f"{t['Faltante vs plan realista (u.)']:,.0f} u.")
                st.metric("Capital inmovilizado en stock", f"${t['Capital inmovilizado ($/mes)']:,.0f} /mes",
                          _delta(t["Capital inmovilizado ($/mes)"], base["Capital inmovilizado ($/mes)"], esc, "{:+,.0f} $"), delta_color="inverse")
        colchon = fila_pol["Stock de seguridad"] / fila_pol["Pronóstico mensual promedio"] * 100
        st.info(f"Con el plan realista, el stock de seguridad de **{selected_cat}** ({int(fila_pol['Stock de seguridad']):,} u.) cubre una demanda "
                f"hasta **{colchon:.1f}%** superior al pronóstico sin generar quiebre de stock. En el escenario realista, el capital inmovilizado es el del stock de seguridad.")
        col_g, _ = st.columns([1.6, 1])
        with col_g:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            plan = np.asarray(futuros[fila_pol["Mejor modelo"]][selected_cat], dtype=float) + int(fila_pol["Stock de seguridad"])
            ax.plot(future_dates, plan, "--", color="black", linewidth=1.8, label="Inventario objetivo (plan realista)")
            for esc in ESCENARIOS:
                ax.plot(future_dates, mensual_c[(selected_cat, esc)], marker="o", linewidth=2, color=COLOR_ESC[esc], label=esc)
            ax.set_ylabel("Unidades"); ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=8); plt.xticks(rotation=45)
            st.pyplot(fig); plt.close(fig)
        _tabla(df_c, {"Cambio demanda (%)": "{:+.2f}%", "Lead time (días)": "{:.2f}", "Ventas jul-dic (u.)": "{:,.0f}",
                      "Ventas jul-dic ($)": "${:,.0f}", "Margen bruto ($)": "${:,.0f}", "Stock de seguridad requerido (u.)": "{:,}",
                      "Inventario objetivo prom. (u./mes)": "{:,.0f}", "Faltante vs plan realista (u.)": "{:,.0f}",
                      "Exceso vs plan realista (u./mes)": "{:,.0f}", "Capital inmovilizado ($/mes)": "${:,.0f}"})


# ============================================================
# NÚCLEO DEL PROGRAMA (MENU LATERAL)
# ============================================================
def main():
    st.sidebar.title("Menú de Navegación")
    st.sidebar.info("Seleccione el análisis a visualizar:")

    opciones = [
        "1. Proyección de demanda",
        "2. Métricas de inventario (MAPE)",
        "3. Importancia de variables (PCA)",
        "4. Análisis de elasticidad",
        "5. Escenarios"
    ]

    eleccion = st.sidebar.radio("Módulos del Proyecto:", opciones)

    # Lógica de carga global para los módulos 1 y 2
    if eleccion in ["1. Proyección de demanda", "2. Métricas de inventario (MAPE)", "5. Escenarios"]:
        demanda, cats_disponibles, df_err_all, df_hw_all, future_dates, futuros, df_inv_stats = load_data_forecast()

        st.sidebar.markdown("---")
        st.sidebar.subheader("🕹️ Filtros de Pronóstico/Inventario")
        selected_cat = st.sidebar.selectbox("Selecciona una subcategoría", cats_disponibles)

        if eleccion == "1. Proyección de demanda":
            mostrar_modulo_proyeccion(df_err_all, df_hw_all, futuros, selected_cat, future_dates, demanda)

        elif eleccion == "2. Métricas de inventario (MAPE)":
            st.sidebar.markdown("---")
            st.sidebar.subheader("📦 Ajuste del stock de seguridad")
            ns_min = st.sidebar.slider("Nivel de servicio mínimo (%)", 90.0, 99.0, 95.0, 0.5)
            ns_max = st.sidebar.slider("Nivel de servicio máximo (%)", 95.0, 99.9, 99.0, 0.1)

            df_politica = construir_politica_inventario(df_err_all, df_inv_stats, futuros, ns_min, ns_max)
            mostrar_modulo_inventario(df_politica, futuros, selected_cat, future_dates)

        elif eleccion == "5. Escenarios":
            st.sidebar.markdown("---")
            st.sidebar.subheader("📦 Nivel de servicio (igual que el módulo 2)")
            ns_min = st.sidebar.slider("Nivel de servicio mínimo (%)", 90.0, 99.0, 95.0, 0.5)
            ns_max = st.sidebar.slider("Nivel de servicio máximo (%)", 95.0, 99.9, 99.0, 0.1)

            df_politica = construir_politica_inventario(df_err_all, df_inv_stats, futuros, ns_min, ns_max)
            datos_esc = cargar_datos_escenarios(FECHA_CORTE_MODELO)
            mostrar_modulo_escenarios(df_politica, futuros, selected_cat, future_dates, demanda, datos_esc)

    elif eleccion == "3. Importancia de variables (PCA)":
        mostrar_modulo_pca()

    elif eleccion == "4. Análisis de elasticidad":
        mostrar_modulo_elasticidad()

if __name__ == "__main__":
    main()
