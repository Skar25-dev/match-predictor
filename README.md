# ⚽ Match Predictor

Predictor de resultados de fútbol (1X2) para **La Liga** y **Segunda División**. El objetivo del proyecto no es
"acertar quinielas", sino practicar un flujo de ML serio sobre datos reales: validación
temporal sin fuga de información, probabilidades calibradas y comparación contra el mercado.

> **Estado:** en construcción. Ahora mismo están hechos la **carga y limpieza de datos**, la
> **descarga automática**, el **análisis por equipo** y la infraestructura (tests, lint, Docker, CI).
> Aún no hay modelo.

## 🗺️ Hoja de ruta

- [x] Carga y limpieza de los CSV (tolerante a temporadas antiguas)
- [x] Script de descarga de datos históricos
- [x] Tests, lint, Docker y CI
- [x] Descargar y validar el histórico completo (10 temporadas; 462 partidos por temporada en Segunda)
- [x] Análisis por equipo: resumen, casa/fuera, forma reciente y evolución por temporada
- [x] Ratings Elo entre temporadas y divisiones (previos al partido, sin fuga de información)
- [x] Features de forma reciente, tiros y descanso (calculadas antes de cada partido, sin fuga)
- [x] Línea base: probabilidades implícitas de las cuotas y regresión logística sobre el Elo
- [x] Modelo de gradient boosting (`HistGradientBoosting` de scikit-learn) y regresión logística con esas features
- [x] La web usa el mejor modelo del backtesting: Elo + tiros a puerta recientes (logística)
- [x] Próxima jornada en la web: probabilidades del modelo junto a las del mercado
- [x] Registro de predicciones: se guardan antes del partido y se comparan con el resultado y el mercado
- [x] Backtesting temporal frente a las cuotas del mercado (log-loss, Brier)
- [x] Web con FastAPI + Jinja2: ficha de equipo y predictor de partido (con selector de equipos)
- [x] Registro de experimentos con MLflow (opcional, `--track`)
- [x] Actualización diaria: una GitHub Action guarda las predicciones en `data/live_predictions.csv`
- [ ] PostgreSQL (solo si el registro crece lo bastante para justificarlo)

## 📊 Datos

Los resultados y cuotas vienen de [football-data.co.uk](https://www.football-data.co.uk/),
un CSV por temporada y división (`SP1` = La Liga, `SP2` = Segunda). Los datos **no se versionan**
en el repositorio: se descargan con el script (ver abajo).

Observaciones de los CSV de la temporada 2025-26 que condicionan el diseño:

- **Cádiz (Segunda) y Sevilla / Real Madrid (La Liga) no se enfrentan nunca en una misma
  temporada.** Para compararlos hace falta un rating que se arrastre entre temporadas y
  divisiones, aprovechando los ascensos y descensos.
- **Las cuotas de Pinnacle faltan en cerca de la mitad de los partidos** (191 de 380 en La Liga
  y 266 de 462 en Segunda). Como referencia del mercado se usarán Bet365 y la media de las
  casas, que vienen completas.
- Temporadas antiguas: menos columnas y fechas con año de dos dígitos. El cargador lo tolera
  (las columnas opcionales que falten quedan como `NaN`), pero esto está cubierto con tests
  sintéticos; el histórico real se validará al descargarlo.

## 🚀 Uso

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# Descargar las últimas 10 temporadas de La Liga y Segunda a data/raw/
python -m matchpredictor.data.download

# Cargar todo en un DataFrame limpio
python -c "from matchpredictor.data import load_all; print(load_all('data/raw').tail())"
```

Opciones de la descarga: `--from-season 2018`, `--divisions SP1`, `--force` (ver `--help`).

### 📈 Análisis de un equipo

```python
from matchpredictor.data import load_all
from matchpredictor.analysis import list_teams, team_summary, form_string, team_by_season

df = load_all("data/raw")
list_teams(df)  # todos los equipos de los datos
team_summary(df, "Cadiz")  # balance global, en casa y fuera
form_string(df, "Sevilla", n=5)  # p. ej. "WDLWW" (el último a la derecha)
team_by_season(df, "Cadiz")  # una fila por temporada y división (ascensos/descensos)
```

Los nombres son los de football-data.co.uk (`Cadiz`, `Ath Madrid`...); si te equivocas, el error
sugiere el más parecido.

### 🧮 Ratings Elo

```python
from matchpredictor.data import load_all
from matchpredictor.features import compute_elo, latest_ratings

games, ratings = compute_elo(load_all("data/raw"))
latest_ratings(ratings).head(10)  # ranking actual
games[["home_team", "away_team", "home_elo", "away_elo", "elo_expected_home"]].tail()
```

Los ratings de cada partido son los **previos** a jugarse, así que valen como features sin
fuga de información. Los parámetros (`k`, `home_advantage`, `season_regression`...) se ajustarán
con backtesting temporal, no a ojo.

### 🎯 Backtesting contra el mercado

```bash
python -m matchpredictor.evaluation.backtest --data data/raw --last 5
```

Cada temporada de test se predice con un modelo entrenado **solo con temporadas anteriores**.
Se comparan frecuencias históricas (`frecuencias`), regresión logística sobre el Elo
(`elo_logit`) y las probabilidades implícitas de las cuotas sin margen (`market`: media de
cierre si existe, y si no Bet365 de cierre, media de apertura o Bet365 de apertura). La tabla
solo cuenta los partidos que tienen cuotas, para que la comparación sea justa.

Referencias: predecir 1/3-1/3-1/3 da log-loss 1,0986 y Brier 0,667; cuanto más bajo, mejor.
El mercado es un rival duro: el objetivo realista es acercarse a él, no superarlo.

### 🧩 Features y modelos

`add_form_features(games)` añade, por cada partido y equipo, la forma reciente (puntos y
diferencia de goles de los últimos 5), la diferencia de tiros y de tiros a puerta (últimos 10) y
los días de descanso, siempre con el historial **anterior** al partido. Hay tests que comparan
cada valor con una implementación ingenua y que comprueban que cambiar un partido no altera sus
propias features. Los modelos `FeatureLogit` y `GradientBoosting` las usan junto al Elo, y el
backtesting (`python -m matchpredictor.evaluation.backtest`) los compara con el mercado.

### 🔧 Ajuste del Elo (sin hacer trampa)

```bash
python -m matchpredictor.evaluation.tuning --data data/raw
```

Prueba combinaciones de `k`, ventaja de campo, regresión entre temporadas y rating inicial de
Segunda. Los parámetros se **eligen** con temporadas de validación y se **comprueban** en
otras de holdout que no se han usado para elegir; la tabla final compara el Elo actual, el
ajustado, las frecuencias y el mercado en el holdout. Si el mejor valor cae en el borde de la
rejilla, el programa avisa de que conviene ampliarla.

### 🔀 ¿Misma escala en La Liga y en Segunda?

```bash
python -m matchpredictor.evaluation.divisions --data data/raw
```

El Elo solo conecta las dos divisiones a través de los equipos que suben y bajan, y no hay
partidos entre ellas con los que comprobarlo directamente. Este análisis mira los primeros
partidos de cada ascendido y descendido en su nueva división y estima un desfase `d`
(*un rating de Segunda equivale a rating + d en La Liga*) con intervalo de confianza por
bootstrap. Si el intervalo incluye 0, no hay evidencia de que haga falta corregir la escala.

Ese desfase se puede usar como parámetro `division_offset` del Elo (se aplica al ascender y
descender) y de la web (al comparar un equipo de Segunda con uno de La Liga). Parte del
desfase medido puede ser regresión a la media (los ascendidos suben tras una buena temporada,
con algo de suerte incluida), así que se valida con `evaluation.tuning` antes de fiarse de él.

### 🌐 Web

```bash
python -m matchpredictor.web --data data/raw      # http://127.0.0.1:8000
python -m matchpredictor.web --data data/raw --division-offset -30   # ajuste entre divisiones
```

- `/`: selector de equipo y de partido (desplegables agrupados por división).
- `/team/<equipo>`: Elo y puesto, forma reciente, balance de la temporada en casa y fuera,
  últimos partidos y recorrido por temporadas (ascensos y descensos).
- `/predict?home=Sevilla&away=Cadiz`: probabilidades 1X2 y cuotas justas. Si los equipos no
  juegan en la misma división, se avisa de que el partido es hipotético.
- `/fixtures`: próxima jornada con las probabilidades del modelo junto a las del mercado
  (filtros `division` y `team`). Avisa si el calendario está desfasado.
- API JSON: `/api/teams`, `/api/team/<equipo>`, `/api/predict?home=..&away=..`,
  `/api/fixtures`, `/health` y
  documentación interactiva en `/api/docs`.

Al arrancar carga los CSV y entrena el modelo una vez; después responde al instante.

**Modelo de la web:** regresión logística sobre la diferencia de Elo y la de tiros a puerta de
los últimos 10 partidos (más la división). Se eligió con un análisis por features en el
backtesting de 5 temporadas: los tiros a puerta son lo único que mejora al Elo (log-loss
1,0199 → 1,0177); la forma por puntos o goles y el descanso no aportan, y el gradient boosting
rinde peor con tan pocos datos. El mercado sigue en 1,0011.

**Partidos entre divisiones** (p. ej. Sevilla–Cádiz): solo se usa el Elo. Los tiros de dos
equipos de divisiones distintas están medidos contra rivales distintos y no son comparables, y
no hay un ajuste validado para ellos; con Elo y tiros juntos el modelo reparte el peso entre
ambos y el Elo, que es lo que separa a los equipos, pesaría menos. El resultado indica qué
modelo se ha usado (`model`: `elo` o `elo_sot`).

**Próxima jornada:** descarga el calendario y arranca la web con él:

```bash
python -m matchpredictor.data.download --fixtures      # guarda data/fixtures.csv
python -m matchpredictor.web --data data/raw --fixtures data/fixtures.csv
```

El mercado son las cuotas medias de apertura (o Bet365 si faltan) sin el margen de la casa.
La hora se muestra tal como viene en el archivo, que no indica la zona horaria. Un partido con
un equipo sin historial aparece como «Sin predicción».

### 📒 Registro de predicciones

```bash
python -m matchpredictor.evaluation.predictions_log record   # guarda las del calendario
python -m matchpredictor.evaluation.predictions_log report   # compara con los resultados
python -m matchpredictor.evaluation.predictions_log replay --since 2026-08-01
```

`record` añade a `data/predictions.csv` las predicciones del calendario (modelo y mercado) y
nunca reescribe un partido ya registrado: se conserva la predicción hecha antes de jugarse.
Cuando descargas los resultados nuevos, `report` calcula log-loss y Brier del modelo y del
mercado sobre los mismos partidos. `replay` simula el registro con partidos ya jugados: cada
semana se entrena solo con lo anterior (sin mirar el futuro) y usa las cuotas de apertura,
como harían los partidos futuros. Esas filas quedan marcadas con `mode = replay`.

`report` incluye además la diferencia de log-loss modelo − mercado con un intervalo de confianza
del 95 % (bootstrap sobre partidos), global, por división y por modelo usado. Si el intervalo
contiene el 0, la diferencia no se distingue del ruido con esa muestra.

### ⏰ Registro diario automático

`.github/workflows/record.yml` se ejecuta cada día (y a mano desde la pestaña *Actions*):
descarga resultados y calendario, guarda las predicciones de los partidos aún sin jugar y
commitea `data/live_predictions.csv`. Como `record` nunca reescribe un partido ya registrado,
cada predicción queda fijada con la información disponible el primer día que apareció en el
calendario. El resumen de cada ejecución muestra la comparación con el mercado. GitHub
desactiva las ejecuciones programadas tras 60 días sin actividad en el repositorio.

### 🔬 Registro de experimentos (MLflow)

MLflow es opcional. Instálalo y añade `--track` al backtesting o al informe del registro:

```bash
pip install -r requirements-tracking.txt
python -m matchpredictor.evaluation.backtest --data data/raw --track
python -m matchpredictor.evaluation.predictions_log report --track
mlflow ui --backend-store-uri file:mlruns        # http://127.0.0.1:5000
```

`requirements-tracking.txt` instala solo el cliente (`mlflow-skinny`), suficiente para registrar.
La interfaz web `mlflow ui` necesita el paquete completo: `pip install mlflow==3.5.0`.

Cada ejecución guarda en `mlruns/` (ignorada por git) los parámetros, el log-loss y Brier de
cada modelo (o la diferencia con el mercado y su intervalo) y el commit del código, para
comparar versiones del modelo sin perder el rastro. Con `MLFLOW_TRACKING_URI` se cambia el destino.

### 🐳 Con Docker

```bash
docker build -t match-predictor .
docker run --rm -v "$(pwd)/data:/app/data" match-predictor   # descarga los datos en ./data/raw

# O con Compose: descarga y web en http://localhost:8000
docker compose run --rm download
docker compose up web
```

### 🧪 Tests y lint

```bash
python -m pytest
ruff check . && ruff format --check .
```

Los tests usan muestras reales recortadas (`tests/fixtures/`) y casos sintéticos para los
formatos antiguos, y nunca acceden a internet. GitHub Actions ejecuta lint, tests y el build
de Docker en cada push y pull request.

## 📂 Estructura

```text
├── matchpredictor/
│   ├── data/
│   │   ├── loader.py      # Carga, limpieza y validación de los CSV
│   │   └── download.py    # Descarga de temporadas desde football-data.co.uk
│   ├── service.py         # Junta datos, Elo y modelo; responde a la web
│   ├── names.py           # Nombres para mostrar (Cadiz -> Cádiz)
│   ├── web/               # FastAPI + plantillas Jinja2
│   ├── models/
│   │   ├── baseline.py    # Frecuencias históricas y regresión logística sobre el Elo
│   │   └── feature_models.py  # Regresión logística y gradient boosting con varias features
│   ├── evaluation/
│   │   ├── metrics.py     # Log-loss, Brier y probabilidades implícitas de las cuotas
│   │   ├── backtest.py    # Backtesting temporal y tabla de resultados
│   │   ├── tuning.py      # Ajuste del Elo con validación y holdout
│   │   └── divisions.py   # Desfase de escala entre La Liga y Segunda (ascensos y descensos)
│   ├── features/
│   │   ├── elo.py         # Elo continuo entre temporadas y divisiones
│   │   └── form.py        # Forma reciente, tiros y descanso (sin fuga de información)
│   └── analysis/
│       └── team.py        # Historial, resumen y forma de un equipo (formato largo)
├── tests/                 # pytest (+ fixtures/ con muestras reales recortadas)
├── data/raw/              # CSV descargados (ignorados por git)
├── .github/workflows/     # CI: lint + tests + build de la imagen Docker
├── Dockerfile
├── docker-compose.yml      # web y descarga de datos
├── requirements.txt       # Dependencias de producción (versiones fijadas)
└── requirements-dev.txt   # + pytest, httpx y ruff
```
