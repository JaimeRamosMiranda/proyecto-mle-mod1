# Identificación de Fenotipos de Pacientes con Insuficiencia Cardíaca mediante Análisis de Conglomerados

> **Proyecto Curso I — Especialización Machine Learning Engineering**  
Ramos Miranda, Jaime

---

## 📋 Tabla de Contenidos

1. [Problema de ML](#1-problema-de-ml)
2. [Diagrama de flujo del proyecto](#2-diagrama-de-flujo-del-proyecto)
3. [Descripción del dataset y diccionario de datos](#3-descripción-del-dataset-y-diccionario-de-datos)
4. [Model Card](#4-model-card)
5. [Resultados y métricas de evaluación](#5-resultados-y-métricas-de-evaluación)
6. [Conclusiones](#6-conclusiones)

---

## 1. Problema de ML

### Contexto clínico

La insuficiencia cardíaca es una de las principales causas de mortalidad cardiovascular a nivel mundial. Los pacientes que la padecen presentan perfiles clínicos muy heterogéneos: diferente edad, función renal, fracción de eyección, comorbilidades y ritmo de deterioro. Esta heterogeneidad dificulta el diseño de estrategias de tratamiento y seguimiento personalizadas.

### Definición del problema

**Tipo:** Aprendizaje no supervisado — Clustering  
**Objetivo:** Identificar subgrupos homogéneos de pacientes (fenotipos) a partir de sus variables clínicas, de modo que cada fenotipo represente un perfil de riesgo diferenciado.

A diferencia del uso supervisado del dataset (predicción de fallecimiento), aquí **no se utiliza `DEATH_EVENT` como variable de entrenamiento**. En cambio, se emplea a posteriori como validación clínica externa para verificar si los conglomerados capturan diferencias reales en la evolución de los pacientes.

### Hipótesis

> Existen subgrupos de pacientes con insuficiencia cardíaca que comparten patrones clínicos similares y que presentan tasas de mortalidad diferenciadas, los cuales pueden ser identificados mediante técnicas de clustering.

### Valor clínico esperado

Cada fenotipo identificado puede orientar decisiones clínicas: priorización de seguimiento, ajuste de tratamiento farmacológico e intensidad de monitoreo.

---

## 2. Diagrama de flujo del proyecto

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATOS CRUDOS                             │
│          heart_failure_clinical_records_dataset1.csv            │
│                    299 pacientes · 13 variables                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESAMIENTO                             │
│  • Separar DEATH_EVENT (validación posterior)                   │
│  • Verificación de valores nulos → ninguno encontrado           │
│  • Estandarización (StandardScaler)                             │
│  • Detección de outliers multivariantes (Mahalanobis, p=0.975)  │
│    → 16 outliers eliminados → 283 pacientes finales             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              DETERMINACIÓN DEL Nº DE CLUSTERS                   │
│  • Método de la Silueta    → k=2 (máximo estadístico)           │
│  • Método del Codo (WSS)   → k=3–4 (punto de inflexión)         │
│  • Gap Statistic           → confirma rango k=3–4               │
│  • Criterio clínico        → k=4 (fenotipos diferenciados)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CLUSTERING                                │
│  • K-Means (k=4, nstart=20, random_state=2020)                  │
│  • Clustering Jerárquico Ward (confirmatorio)                   │
│  • Visualización PCA (2 componentes)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               CARACTERIZACIÓN DE FENOTIPOS                      │
│  • Perfiles de centroides por fenotipo                          │
│  • Tablas de variables continuas y categóricas                  │
│  • Heatmap de centroides                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               VALIDACIÓN CLÍNICA EXTERNA                        │
│  • Tasa de mortalidad (DEATH_EVENT) por fenotipo                │
│  • Fenotipo 4: 73.3% mortalidad (alto riesgo confirmado)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Descripción del dataset y diccionario de datos

**Nombre:** Heart Failure Clinical Records Dataset  
**Fuente:** [Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) / [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)  
**Formato:** CSV tabular  
**Tamaño:** 299 filas × 13 columnas (~22 KB)  
**Valores nulos:** Ninguno  
**Periodo:** Registro clínico de seguimiento entre 4 y 285 días

### Diccionario de datos

| Variable | Tipo | Rango | Descripción |
|---|---|---|---|
| `age` | Continua | 40 – 95 años | Edad del paciente |
| `anaemia` | Binaria | 0 / 1 | Presencia de anemia (descenso de glóbulos rojos) |
| `creatinine_phosphokinase` | Continua | 23 – 7861 mcg/L | Nivel de la enzima CPK en sangre |
| `diabetes` | Binaria | 0 / 1 | Diagnóstico de diabetes mellitus |
| `ejection_fraction` | Continua | 14 – 80 % | Porcentaje de sangre expulsado por el corazón en cada contracción |
| `high_blood_pressure` | Binaria | 0 / 1 | Diagnóstico de hipertensión arterial |
| `platelets` | Continua | 25 100 – 850 000 /mL | Conteo de plaquetas en sangre |
| `serum_creatinine` | Continua | 0.5 – 9.4 mg/dL | Nivel de creatinina sérica (indicador de función renal) |
| `serum_sodium` | Continua | 113 – 148 mEq/L | Nivel de sodio en sangre |
| `sex` | Binaria | 0 / 1 | Sexo (0: mujer, 1: hombre) |
| `smoking` | Binaria | 0 / 1 | Hábito tabáquico activo |
| `time` | Continua | 4 – 285 días | Días de seguimiento clínico |
| `DEATH_EVENT` ⚠️ | Binaria | 0 / 1 | Fallecimiento durante el seguimiento *(no se usa en el entrenamiento)* |

> ⚠️ `DEATH_EVENT` se excluye del clustering y se usa únicamente como validador clínico externo.

---

## 4. Model Card

### Información general

| Campo | Detalle |
|---|---|
| **Nombre del modelo** | K-Means Fenotipos — Insuficiencia Cardíaca |
| **Versión** | 1.0.0 |
| **Tipo** | Clustering no supervisado (K-Means) |
| **Librería** | scikit-learn 1.x |
| **Lenguaje** | Python 3.10+ |
| **Fecha de entrenamiento** | Junio 2026 |

### Uso previsto

**Propósito:** Herramienta exploratoria para estratificación de pacientes con insuficiencia cardíaca en subgrupos clínicamente diferenciados (fenotipos).

**Usuarios objetivo:** Investigadores clínicos, equipos de cardiología, científicos de datos en salud.

**Uso NO previsto:** Este modelo no debe utilizarse para diagnóstico clínico individual, ni como sustituto de criterio médico. No está validado para uso en producción clínica.

### Arquitectura y parámetros

| Parámetro | Valor |
|---|---|
| Algoritmo | K-Means |
| Número de clusters (k) | 4 |
| `n_init` | 20 |
| `max_iter` | 300 |
| `random_state` | 2020 |
| Preprocesamiento | StandardScaler + eliminación outliers Mahalanobis (p=0.975) |
| Método confirmatorio | Clustering Jerárquico Ward |

### Datos de entrenamiento

- 299 pacientes originales
- 16 outliers multivariantes eliminados
- **283 pacientes finales** para entrenamiento
- 12 variables clínicas (sin `DEATH_EVENT`)

### Limitaciones

- Dataset de tamaño pequeño (n=283), lo que puede afectar la estabilidad de los clusters.
- Los fenotipos son descriptivos y exploratorios, no tienen validación prospectiva.
- El coeficiente de silueta bajo (0.116) indica solapamiento entre grupos, esperable en datos clínicos continuos.
- No se incluyen variables de tratamiento farmacológico que podrían enriquecer la segmentación.

### Consideraciones éticas

- Los datos son anónimos y de uso académico-investigativo.
- No se realizan predicciones individuales; el análisis es poblacional y agregado.

---

## 5. Resultados y métricas de evaluación

### 5.1 Métricas de evaluación offline (clustering)

Las métricas de clustering evalúan la calidad interna de los conglomerados sin referencia a una etiqueta externa.

| Métrica | Valor | Interpretación |
|---|---|---|
| **Coeficiente de Silueta** | 0.1163 | Separación moderada entre clusters (valores bajos son esperables en datos clínicos de alta dimensión) |
| **Índice Davies-Bouldin** | 2.3085 | Menor es mejor; indica distancia promedio entre clusters |
| **Índice Calinski-Harabasz** | 29.79 | Mayor es mejor; mide compacidad vs separación |
| **Inercia (WSS)** | 2194.36 | Suma de cuadrados intra-cluster |

> **Nota sobre la selección de k=4:** aunque k=2 maximiza la silueta (0.1270), se optó por k=4 dado que ofrece una diferenciación clínica superior — especialmente por la identificación del Fenotipo 4 de altísimo riesgo (73.3% de mortalidad), lo cual no es visible con k=2 (donde la mortalidad es 31.3% vs 28.7%, prácticamente indistinguible).

### 5.2 Métricas de evaluación online (validación clínica externa)

Se usa `DEATH_EVENT` como validador externo — si los clusters capturan diferencias reales de riesgo, deben mostrar tasas de mortalidad diferenciadas.

| Fenotipo | N pacientes | Fallecidos | Tasa de mortalidad |
|---|---|---|---|
| **Fenotipo 1** | 69 | 19 | 27.5% |
| **Fenotipo 2** | 79 | 19 | 24.1% |
| **Fenotipo 3** | 90 | 15 | 16.7% |
| **Fenotipo 4** | 45 | 33 | **73.3%** ⚠️ |

El Fenotipo 4 presenta una tasa de mortalidad 4.4× mayor que el Fenotipo 3, lo que valida la utilidad clínica del clustering.

### 5.3 Perfiles de fenotipos

#### Variables continuas (medias por fenotipo)

| Variable | Fenotipo 1 | Fenotipo 2 | Fenotipo 3 | Fenotipo 4 |
|---|---|---|---|---|
| Edad (años) | 60.5 | 59.7 | 56.9 | **71.7** |
| CPK (mcg/L) | 438.9 | 469.0 | 528.9 | 390.0 |
| Fracción eyección (%) | 37.7 | 40.1 | 43.3 | **28.4** |
| Plaquetas (/mL) | 244 913 | 265 380 | 278 992 | 241 111 |
| Creatinina sérica (mg/dL) | 1.32 | 1.16 | 1.09 | **1.99** |
| Sodio sérico (mEq/L) | 137.0 | 137.4 | 137.8 | **133.6** |
| Tiempo seguimiento (días) | 117.1 | 142.7 | **161.7** | 81.5 |

#### Variables categóricas (% con la condición)

| Condición | Fenotipo 1 | Fenotipo 2 | Fenotipo 3 | Fenotipo 4 |
|---|---|---|---|---|
| Anemia | 46.4% | 35.4% | 45.6% | 46.7% |
| Diabetes | 44.9% | 31.6% | 55.6% | 24.4% |
| Hipertensión | **100%** | 29.1% | 0% | 20.0% |
| Sexo masculino | 39.1% | **100%** | 42.2% | **88.9%** |
| Tabaquismo | 2.9% | **100%** | 0% | 17.8% |

### 5.4 Interpretación clínica de fenotipos

| Fenotipo | Nombre descriptivo | Características clave | Riesgo |
|---|---|---|---|
| **F1** | Paciente hipertenso de riesgo moderado | 100% hipertensión, mayoría mujeres, sin tabaquismo | Moderado (27.5%) |
| **F2** | Hombre fumador activo | 100% masculino, 100% fumador, función cardíaca conservada | Moderado-bajo (24.1%) |
| **F3** | Paciente joven diabético de bajo riesgo | Menor edad (56.9 años), mayor fracción de eyección, mayor tiempo de seguimiento | Bajo (16.7%) |
| **F4** | Paciente anciano con deterioro multiorgánico | Mayor edad (71.7 años), baja fracción de eyección, creatinina elevada, sodio bajo | **Muy alto (73.3%)** ⚠️ |

---

## 6. Conclusiones

1. **Se identificaron 4 fenotipos clínicamente diferenciados** en pacientes con insuficiencia cardíaca mediante K-Means, confirmados por clustering jerárquico Ward.

2. **El Fenotipo 4** representa un perfil de altísimo riesgo: pacientes de mayor edad, con función cardíaca deteriorada (baja fracción de eyección), daño renal (creatinina elevada) e hiponatremia. Su tasa de mortalidad de 73.3% lo distingue claramente del resto y valida la utilidad del modelo.

3. **El Fenotipo 3** agrupa a pacientes más jóvenes con mejor función cardíaca y mayor tiempo de seguimiento, con la menor mortalidad (16.7%), lo que sugiere un perfil de evolución más favorable.

4. **Los fenotipos 1 y 2** se diferencian principalmente por comorbilidades específicas: hipertensión exclusiva en F1 y tabaquismo masculino en F2, con mortalidades similares (~24–28%).

5. **El solapamiento entre clusters** (silueta = 0.116) es esperable en datos clínicos continuos y no invalida el análisis; la validación clínica mediante mortalidad diferencial confirma la relevancia de la segmentación.

6. **Perspectivas futuras:** incorporar variables de tratamiento farmacológico, aplicar técnicas de reducción de dimensionalidad (UMAP) para mejor separación visual, y validar los fenotipos en cohortes externas.

---

## 🗂️ Estructura del repositorio

```
proyecto-mle-mod1/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_preprocesamiento.ipynb
│   └── 02_clustering_y_analisis.ipynb
│
├── data/
│   ├── .gitkeep
│   └── heart_failure_clinical_records_dataset1.csv
│
├── artifacts/
│   ├── cuadros_fenotipos.xlsx
│   ├── heart_failure_con_fenotipos.csv
│   ├── distancias_euclideas.png
│   ├── dendrograma_k4.png
│   ├── metodo_silueta.png
│   ├── metodo_codo.png
│   ├── gap_statistic.png
│   ├── clusters_kmeans_k4.png
│   └── perfiles_centroides.png
│
└── src/
    └── analisis_conglomerados_heart_failure.py
```

---

## ⚙️ Instalación y ejecución

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/proyecto-mle-mod1.git
cd proyecto-mle-mod1

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el análisis
python src/analisis_conglomerados_heart_failure.py
```

---

*Especialización Machine Learning Engineering · Junio 2026*
