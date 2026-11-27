"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. El informe,
docs/informe.md, cita las cifras y las figuras que produce este guion.
"""
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from torch import nn  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from comun import (RAIZ, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, mae, mape, rmse)

# Después de importar statsmodels, que al cargarse fija sus propios avisos.
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 2026
N_PRUEBA = 24          # los meses del backtest, los mismos de comun.backtest
H = 12                 # horizonte, el mismo de comun.backtest
SEMILLAS = [SEMILLA, 7, 13, 42, 123]

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}
# Pronósticos por origen, para el análisis de errores por fecha objetivo.
PRONOSTICOS = {}
RESULTADO_SEMILLAS = None

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
# La serie del miniproyecto 1: ventas residenciales de electricidad en EEUU.
ARCHIVO = "datos/electricidad_residencial_eeuu.csv"
FECHA = "fecha"
VALOR = "ventas_miles_mwh"
PERIODO = 12


def preparar(serie):
    """Conserva los 120 valores originales, sin huecos ni recortes.

    El CSV verificado trae los meses consecutivos de 2016-01 a 2025-12.
    La revisión del miniproyecto 1 señaló valores inusuales, sin evidencia
    de errores de registro, así que no se imputan ni se eliminan.
    """
    return serie


def exogenas_calendario(serie):
    """Días calendario de cada mes, conocidos de antemano, como exógena.

    El índice cubre la serie y 12 meses más, necesarios para el horizonte
    del último origen del backtest.
    """
    indice = serie.index.append(fechas_siguientes(serie, PERIODO))
    return pd.DataFrame({"dias": indice.days_in_month}, index=indice)


def modelo_clasico(serie):
    """SARIMAX (0,1,0)(0,1,1)_12 con los días del mes del miniproyecto 1."""
    x = exogenas_calendario(serie)

    def modelo(entrenamiento, h):
        ajuste = SARIMAX(entrenamiento, order=(0, 1, 0),
                         seasonal_order=(0, 1, 1, PERIODO),
                         exog=x.loc[entrenamiento.index]).fit(disp=False)
        ULTIMO["modelo"] = ajuste
        futuro = fechas_siguientes(entrenamiento, h)
        return np.asarray(ajuste.get_forecast(h, exog=x.loc[futuro]).predicted_mean)

    return modelo


NOMBRE_CLASICO = "SARIMAX (0,1,0)(0,1,1)12 + días"


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--")
    eje.set_title(f"{VALOR}: serie mensual; a la derecha de la línea, los meses del backtest")
    eje.set_ylabel(VALOR)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Los modelos aprenden la diferencia contra el mismo mes del año anterior.
# Así la Tendencia anual no tiene que aprenderse de nuevo en cada origen.
OBJETIVO = "dif12"
REZAGOS = 12
VENTANAS = (3, 12, 24)


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t).

    `y` ya incluye la fecha objetivo t como última celda (NaN), así que
    t-12 es la posición -PERIODO-1, no -PERIODO.
    """
    return y.iloc[-PERIODO - 1] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """Una fila por mes con 12 rezagos, mes civil y estadísticas de ventana.

    Las ventanas se calculan sobre z.shift(1): para la fila t solo pueden
    entrar valores hasta t-1. El mes se lee del índice, no de la serie.
    """
    X = pd.DataFrame(index=z.index)
    for k in range(1, REZAGOS + 1):
        X[f"rezago_{k}"] = z.shift(k)
    X["mes"] = z.index.month
    X["mes_sin"] = np.sin(2 * np.pi * z.index.month / PERIODO)
    X["mes_cos"] = np.cos(2 * np.pi * z.index.month / PERIODO)
    corrida = z.shift(1)
    for w in VENTANAS:
        X[f"media_{w}"] = corrida.rolling(w, min_periods=w).mean()
        X[f"desv_{w}"] = corrida.rolling(w, min_periods=w).std()
    return X


def comprobar_sin_fuga(serie):
    """Cambia el valor de cada mes y comprueba que su fila no cambia."""
    z = objetivo(serie)
    original = tabla_supervisada(z)
    for fecha in original.index:
        alterada = z.copy()
        alterada.loc[fecha] += 1000.0
        fila_alterada = tabla_supervisada(alterada).loc[fecha]
        if not np.allclose(original.loc[fecha].to_numpy(dtype=float),
                           fila_alterada.to_numpy(dtype=float),
                           equal_nan=True):
            raise AssertionError(f"la fila de {fecha:%Y-%m} ve su propio valor")
    print(f"  sin fuga: {len(original)} filas verificadas una a una")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo: primero se abre la fecha objetivo y luego se predice.

    En cada paso se agrega la fecha siguiente como NaN, se construye con esa
    fecha la fila de predictores, se predice su dif12 y se reconstruye el nivel
    sumando el valor de esa misma fecha 12 meses antes. Después se guarda el
    nivel pronosticado en la celda que estaba vacía.
    """
    y_trabajo = y.astype(float).copy()
    predicciones = []
    for _ in range(h):
        fecha = fechas_siguientes(y_trabajo, 1)[0]
        y_trabajo = pd.concat([y_trabajo, pd.Series([np.nan], index=[fecha])])
        X = tabla_supervisada(objetivo(y_trabajo)).iloc[[-1]]
        z_hat = np.asarray(ajustado.predict(X)).reshape(-1)[0]
        nivel = float(np.asarray(a_nivel(y_trabajo, np.array([z_hat])))[0])
        predicciones.append(nivel)
        y_trabajo.loc[fecha] = nivel
    return np.asarray(predicciones)


def comprobar_recursivo(serie):
    """Un origen en diciembre debe pronosticar enero con rezago 1 de diciembre."""
    entrenamiento = serie.iloc[-36:]          # termina en diciembre
    meses = pd.date_range(entrenamiento.index[-1], periods=2, freq="MS")[1:]
    filas = []

    class ModeloPrueba:
        def predict(self, X):
            filas.append(X.iloc[-1].copy())
            return np.zeros(len(X))

    pronostico = recursivo(ModeloPrueba(), entrenamiento, 1)
    primera = filas[0]
    enero, diciembre = meses[0], entrenamiento.index[-1]
    z_diciembre = objetivo(entrenamiento).iloc[-1]
    assert primera.name == enero, "la primera fila no es la fecha objetivo"
    assert int(primera["mes"]) == enero.month, "la primera fila no es enero"
    assert np.isclose(primera["rezago_1"], z_diciembre), \
        "rezago_1 no es la diferencia de diciembre"
    assert np.isclose(pronostico[0], entrenamiento.iloc[-12]), \
        "el nivel no usa la misma fecha del año anterior"
    print(f"  recursivo: primera fila {enero:%Y-%m} con rezago_1 de "
          f"{diciembre:%Y-%m} ({z_diciembre:.2f}) y nivel de "
          f"{entrenamiento.index[-12]:%Y-%m}")


def con_tabla(fabrica):
    """Modelo para el backtest: tabla, ajuste por origen y escalador por origen."""
    def modelo(entrenamiento, h):
        X, z = filas_completas(entrenamiento)
        ajustado = fabrica()
        ajustado.fit(X, z)
        ULTIMO["modelo"] = ajustado
        return recursivo(ajustado, entrenamiento, h)
    return modelo


# ============================================================
# 2b. Backtest con pronósticos guardados (comun.py no se toca)
# ============================================================

def backtest_detallado(serie, modelo, n_prueba=N_PRUEBA, h=H):
    """Mismo origen móvil y métricas de comun.backtest, con los pronósticos.

    comun.py queda intacto: esta función solo repite su bucle para poder
    guardar las predicciones de cada origen y agruparlas por fecha objetivo.
    """
    filas = []
    predicciones = []
    primero = len(serie) - n_prueba
    for corte in range(primero, len(serie) - h + 1):
        entrenamiento = serie.iloc[:corte]
        real = serie.iloc[corte:corte + h].to_numpy()
        fechas = fechas_siguientes(entrenamiento, h)
        pronostico = np.asarray(modelo(entrenamiento, h))
        filas.append({"origen": serie.index[corte - 1],
                      "RMSE": rmse(real, pronostico),
                      "MAE": mae(real, pronostico),
                      "MAPE": mape(real, pronostico)})
        predicciones.append({"origen": serie.index[corte - 1],
                             "fechas": fechas,
                             "pronostico": pronostico})
    por_origen = pd.DataFrame(filas)
    return por_origen, por_origen[["RMSE", "MAE", "MAPE"]].mean(), predicciones


def conteo_filas_supervisadas(serie):
    """Filas supervisadas en el primer y en el último origen del backtest."""
    primero = serie.iloc[:len(serie) - N_PRUEBA]
    ultimo = serie.iloc[:-H]
    n_primero = len(filas_completas(primero)[0])
    n_ultimo = len(filas_completas(ultimo)[0])
    print(f"  filas supervisadas: primer origen {n_primero} de "
          f"{len(primero)} meses; último origen {n_ultimo} de {len(ultimo)} meses")
    pd.DataFrame([
        {"origen": "primero", "meses_entrenamiento": len(primero), "filas": n_primero},
        {"origen": "ultimo", "meses_entrenamiento": len(ultimo), "filas": n_ultimo},
    ]).to_csv(RESULTADOS / "filas_supervisadas.csv", index=False)
    return n_primero, n_ultimo


def peor_mes_por_modelo(serie, pronosticos):
    """Media de |real - pronóstico| por fecha objetivo y peor fecha por modelo."""
    filas = []
    for nombre, orígenes in pronosticos.items():
        for origen in orígenes:
            fechas = origen["fechas"]
            real = serie.reindex(fechas).to_numpy(dtype=float)
            for fecha, y, y_hat in zip(fechas, real, origen["pronostico"]):
                filas.append({"modelo": nombre, "fecha": fecha,
                              "error_abs": abs(y - y_hat)})
    errores = pd.DataFrame(filas)
    por_fecha = (errores.groupby(["modelo", "fecha"], as_index=False)
                 .agg(error_abs=("error_abs", "mean"), n_origenes=("error_abs", "size")))
    por_fecha.to_csv(RESULTADOS / "errores_por_fecha.csv", index=False)
    peor = (por_fecha.loc[por_fecha.groupby("modelo")["error_abs"].idxmax()]
            .sort_values("modelo")
            .rename(columns={"fecha": "peor_mes", "error_abs": "error_medio",
                             "n_origenes": "origenes"}))
    peor["peor_mes"] = peor["peor_mes"].dt.strftime("%Y-%m")
    peor.to_csv(RESULTADOS / "peor_mes.csv", index=False)
    return peor


# ============================================================
# 3. Aprendizaje automático
# ============================================================

def knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))


def svr(C=1.0, epsilon=0.1):
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon))


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


REJILLAS = {
    "KNN": (knn, {"kneighborsregressor__n_neighbors": [3, 5, 9]}),
    "SVR": (svr, {"svr__C": [0.3, 1.0, 3.0], "svr__epsilon": [0.05, 0.1]}),
    "Random Forest": (bosque, {"max_depth": [3, 6, None],
                               "min_samples_leaf": [1, 3]}),
    "XGBoost": (xgboost, {"max_depth": [2, 3, 4],
                          "learning_rate": [0.05, 0.1]}),
}


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def ajustar_hiperparametros(serie):
    """GridSearchCV con TimeSeriesSplit sobre los meses anteriores al backtest."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    elegidos = {}
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        busqueda = GridSearchCV(fabrica(), rejilla, cv=TimeSeriesSplit(n_splits=3),
                                scoring="neg_mean_squared_error", n_jobs=1)
        busqueda.fit(X, z)
        elegidos[nombre] = sin_prefijo(busqueda.best_params_)
        print(f"  {nombre}: {elegidos[nombre]} | RMSE validación "
              f"{np.sqrt(-busqueda.best_score_):.2f}")
    return elegidos


def graficar_importancia(serie, elegidos):
    """Importancia de las variables según Random Forest y XGBoost."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    modelos = {
        "Random Forest": bosque(**elegidos["Random Forest"]).fit(X, z),
        "XGBoost": xgboost(**elegidos["XGBoost"]).fit(X, z),
    }
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for eje, (nombre, modelo) in zip(ejes, modelos.items()):
        importancia = pd.Series(modelo.feature_importances_, index=X.columns)
        importancia.sort_values().tail(8).plot.barh(ax=eje)
        eje.set_title(nombre)
        eje.set_xlabel("importancia")
    figura.suptitle("Variables con más peso (entrenamiento previo al backtest)")
    figura.tight_layout()
    figura.savefig(FIGURAS / "importancia.png", dpi=120)
    plt.close(figura)


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
EPOCAS_MAXIMAS = 150
TASA = 0.01
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


class Red:
    """Red de PyTorch con fit y predict, usable como modelo de scikit-learn."""

    def __init__(self, construir, secuencia=False, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA):
        self.construir = construir
        self.secuencia = secuencia
        self.epocas = epocas
        self.semilla = semilla

    def entradas(self, X):
        if self.secuencia:
            columnas = [f"rezago_{k}" for k in range(REZAGOS, 0, -1)]
            x = (X[columnas].to_numpy() - self.media) / self.desv
            return torch.tensor(x[:, :, None], dtype=torch.float32, device=DISPOSITIVO)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32, device=DISPOSITIVO)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32, device=DISPOSITIVO)
        self.red = self.construir(x.shape[-1]).to(DISPOSITIVO)
        optimizador = torch.optim.Adam(self.red.parameters(), lr=TASA)
        perdida = nn.MSELoss()
        self.perdidas = []
        for _ in range(self.epocas):
            self.red.train()
            optimizador.zero_grad()
            error = perdida(self.red(x), objetivo_t)
            error.backward()
            optimizador.step()
            fila = [error.item()]
            if X_val is not None:
                fila.append(float(np.mean(((self.predict(X_val) - z_val) / self.desv) ** 2)))
            self.perdidas.append(fila)
        return self

    def predict(self, X):
        self.red.eval()
        with torch.no_grad():
            salida = self.red(self.entradas(X)).cpu().numpy()[:, 0]
        return salida * self.desv + self.media


class MLP(nn.Module):
    def __init__(self, entradas, ocultas=16):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                                   nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.capas(x)


class LSTM(nn.Module):
    """Una capa LSTM sobre (lote, 12, 1) y una lineal sobre el último estado."""

    def __init__(self, ocultas=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=ocultas, batch_first=True)
        self.lineal = nn.Linear(ocultas, 1)

    def forward(self, x):
        _, (estado, _) = self.lstm(x)
        return self.lineal(estado[-1])


class Conv1DRed(nn.Module):
    """Tercera red: convolución 1D sobre la secuencia de 12 rezagos."""

    def __init__(self, canales=8):
        super().__init__()
        self.capas = nn.Sequential(
            nn.Conv1d(1, canales, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(canales * REZAGOS, 1),
        )

    def forward(self, x):
        return self.capas(x.transpose(1, 2))


CANDIDATAS = {
    "MLP (8 ocultas)": (lambda n: MLP(n, 8), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
    "LSTM (16 ocultas)": (lambda n: LSTM(16), True),
    "Conv1D (8 canales)": (lambda n: Conv1DRed(8), True),
}


def validar_redes(serie):
    """Curvas de pérdida y épocas de mínima validación de cada candidata."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    corte = len(X) - H
    X_ent, z_ent = X.iloc[:corte], z.iloc[:corte]
    X_val, z_val = X.iloc[corte:], z.iloc[corte:]
    perdidas = {}
    epocas_candidatas = {}
    for nombre, (construir, secuencia) in CANDIDATAS.items():
        red = Red(construir, secuencia, EPOCAS_MAXIMAS).fit(X_ent, z_ent, X_val, z_val)
        perdidas[nombre] = np.asarray(red.perdidas, dtype=float)
        epocas_candidatas[nombre] = int(np.argmin(perdidas[nombre][:, 1]) + 1)
        print(f"  {nombre}: {perdidas[nombre].shape[0]} épocas, mínimo validación "
              f"en la época {epocas_candidatas[nombre]} "
              f"({perdidas[nombre][:, 1].min():.3f})")

    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for nombre, valores in perdidas.items():
        ejes[0].plot(valores[:, 0], label=nombre)
        ejes[1].plot(valores[:, 1], label=nombre)
    ejes[0].set_title("Pérdida de entrenamiento")
    ejes[1].set_title("Pérdida de validación")
    for eje in ejes:
        eje.set_xlabel("época")
        eje.set_ylabel("MSE sobre z estandarizada")
        eje.grid(alpha=0.25)
    ejes[1].legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=120)
    plt.close(figura)

    mlps = [n for n in CANDIDATAS if n.startswith("MLP")]
    mejor_mlp = min(mlps, key=lambda n: perdidas[n][:, 1].min())
    finales = [mejor_mlp, "LSTM (16 ocultas)", "Conv1D (8 canales)"]
    print(f"  MLP elegida: {mejor_mlp}")
    return {nombre: epocas_candidatas[nombre] for nombre in finales}


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


def repetir_con_semillas(serie, nombre, epocas):
    """La mejor red con cinco semillas, cada una por el backtest completo."""
    global RESULTADO_SEMILLAS
    filas = []
    for semilla in SEMILLAS:
        _, promedio, _ = backtest_detallado(serie, con_tabla(red(nombre, epocas, semilla)))
        filas.append({"semilla": semilla, "RMSE": round(promedio["RMSE"], 2)})
        print(f"  semilla {semilla}: RMSE {promedio['RMSE']:.2f}")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "semillas.csv", index=False)
    media = tabla["RMSE"].mean()
    desviacion = tabla["RMSE"].std(ddof=1)
    print(f"  media {media:.2f}; desviación estándar {desviacion:.2f}")
    RESULTADO_SEMILLAS = {"tabla": tabla, "media": media, "desviacion": desviacion}
    return RESULTADO_SEMILLAS


# ============================================================
# 5. La tabla y la elección
# ============================================================

def contar_parametros(ajustado):
    """Cuántos números guarda el modelo ajustado."""
    if hasattr(ajustado, "params"):
        return len(ajustado.params)
    if isinstance(ajustado, Red):
        return sum(p.numel() for p in ajustado.red.parameters())
    final = ajustado.steps[-1][1] if hasattr(ajustado, "steps") else ajustado
    if isinstance(final, KNeighborsRegressor):
        return final.n_samples_fit_ * final.n_features_in_
    if isinstance(final, SVR):
        return final.support_vectors_.size + final.dual_coef_.size + 1
    if isinstance(final, RandomForestRegressor):
        return sum(arbol.tree_.node_count for arbol in final.estimators_)
    if isinstance(final, XGBRegressor):
        return len(final.get_booster().trees_to_dataframe())
    return 0


def fila(nombre, serie, modelo):
    ULTIMO.clear()
    inicio = time.perf_counter()
    _, promedio, predicciones = backtest_detallado(serie, modelo)
    segundos = time.perf_counter() - inicio
    PRONOSTICOS[nombre] = predicciones
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s "
          f"({parametros} parámetros)")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9, 3.6))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy(), color="black", label="real")
    for nombre, modelo in modelos.items():
        pronostico = modelo(entrenamiento, 12)
        eje.plot(real.index, pronostico, marker="o", markersize=3,
                 label=f"{nombre} (RMSE {rmse(real, pronostico):.1f})")
    eje.set_ylabel(VALOR)
    eje.set_title("Pronóstico a 12 meses desde el último origen")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=120)
    plt.close(figura)


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    inicio = time.perf_counter()
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    print("1. Línea base y modelo a superar")
    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    try:
        clasico = modelo_clasico(serie)
        filas.append(fila(NOMBRE_CLASICO, serie, clasico))
        elegidos_para_figura[NOMBRE_CLASICO] = clasico
    except NotImplementedError as error:
        pendiente("parte 1", error)

    print("2. Tabla supervisada")
    try:
        comprobar_sin_fuga(serie)
        comprobar_recursivo(serie)
        conteo_filas_supervisadas(serie)
        tabla_lista = True
    except NotImplementedError as error:
        pendiente("parte 2", error)
        tabla_lista = False

    if tabla_lista:
        print("3. Aprendizaje automático")
        try:
            elegidos = ajustar_hiperparametros(serie)
            graficar_importancia(serie, elegidos)
            modelos_aa = {nombre: con_tabla(lambda f=fabrica, p=elegidos[nombre]: f(**p))
                          for nombre, (fabrica, _) in REJILLAS.items()}
            filas_aa = [fila(nombre, serie, modelo) for nombre, modelo in modelos_aa.items()]
            filas += filas_aa
            mejor_aa = min(filas_aa, key=lambda f: f["RMSE"])["modelo"]
            elegidos_para_figura[mejor_aa] = modelos_aa[mejor_aa]
        except NotImplementedError as error:
            pendiente("parte 3", error)

        print("4. Aprendizaje profundo")
        try:
            epocas = validar_redes(serie)
            filas_red = [fila(f"{nombre}, {e} épocas", serie, con_tabla(red(nombre, e)))
                         for nombre, e in epocas.items()]
            filas += filas_red
            mejor = min(zip(epocas, filas_red), key=lambda par: par[1]["RMSE"])[0]
            elegidos_para_figura[mejor] = con_tabla(red(mejor, epocas[mejor]))
            print(f"  mejor red: {mejor} con {epocas[mejor]} épocas")
            repetir_con_semillas(serie, mejor, epocas[mejor])
        except NotImplementedError as error:
            pendiente("parte 4", error)

    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print("\n5. La tabla")
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])

    print("\nAnálisis de la peor fecha objetivo por modelo")
    peor = peor_mes_por_modelo(serie, PRONOSTICOS)
    print(peor.to_string(index=False))

    if len(elegidos_para_figura) > 1:
        graficar_pronosticos(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
