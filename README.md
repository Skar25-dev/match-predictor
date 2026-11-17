# ⚽ Match Predictor

Predictor de resultados de fútbol (1X2) para **La Liga** y **Segunda División**, pensado para
seguir a equipos como el Cádiz, el Sevilla y el Real Madrid. El objetivo del proyecto no es
"acertar quinielas", sino practicar un flujo de ML serio sobre datos reales: validación
temporal sin fuga de información, probabilidades calibradas y comparación contra el mercado.

> **Estado:** en construcción. Ahora mismo están hechos la **carga y limpieza de datos**, la
> **descarga automática** y la infraestructura (tests, lint, Docker, CI). Aún no hay modelo.

## 🗺️ Hoja de ruta

- [x] Carga y limpieza de los CSV (tolerante a temporadas antiguas)
- [x] Script de descarga de datos históricos
- [x] Tests, lint, Docker y CI
- [ ] Descargar y validar el histórico completo (varias temporadas)
- [ ] Ratings Elo entre temporadas y divisiones, forma reciente, descanso entre partidos
- [ ] Línea base: probabilidades implícitas de las cuotas y regresión logística
- [ ] Modelo de gradient boosting (XGBoost / LightGBM) con probabilidades calibradas
- [ ] Backtesting temporal frente a las cuotas del mercado (log-loss, Brier)
- [ ] API con FastAPI, registro de experimentos con MLflow, Docker Compose + PostgreSQL

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

### 🐳 Con Docker

```bash
docker build -t match-predictor .
docker run --rm -v "$(pwd)/data:/app/data" match-predictor   # descarga los datos en ./data/raw
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
│   └── data/
│       ├── loader.py      # Carga, limpieza y validación de los CSV
│       └── download.py    # Descarga de temporadas desde football-data.co.uk
├── tests/                 # pytest (+ fixtures/ con muestras reales recortadas)
├── data/raw/              # CSV descargados (ignorados por git)
├── .github/workflows/     # CI: lint + tests + build de la imagen Docker
├── Dockerfile
├── requirements.txt       # Dependencias de producción (versiones fijadas)
└── requirements-dev.txt   # + pytest y ruff
```
