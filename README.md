# Identificación de Fenotipos de Pacientes con Insuficiencia Cardíaca mediante Análisis de Conglomerados

> **Proyecto Curso I — Especialización Machine Learning Engineering**  
> Autor: Ramos Miranda, Jaime

---

## 📋 Tabla de Contenidos

1. [Problema de ML](#1-problema-de-ml)
2. [Diagrama de flujo del proyecto](#2-diagrama-de-flujo-del-proyecto)
3. [Descripción del dataset y diccionario de datos](#3-descripción-del-dataset-y-diccionario-de-datos)
4. [Librerías utilizadas](#4-librerías-utilizadas)
5. [Model Card](#5-model-card)
6. [Resultados y métricas de evaluación](#6-resultados-y-métricas-de-evaluación)
7. [Conclusiones](#7-conclusiones)

---

## 1. Problema de ML

### Contexto clínico

La insuficiencia cardíaca es una de las principales causas de mortalidad cardiovascular a nivel mundial. Los pacientes que la padecen presentan perfiles clínicos muy heterogéneos: diferente edad, función renal, fracción de eyección, comorbilidades y ritmo de deterioro. Esta heterogeneidad dificulta el diseño de estrategias de tratamiento y seguimiento personalizadas.

### Definición del problema

**Tipo:** Aprendizaje no supervisado — Clustering  
**Objetivo:** Identificar subgrupos homogéneos de pacientes (fenotipos) a partir de sus variables clínicas, de modo que cada fenotipo represente un perfil de riesgo diferenciado.

A diferencia del uso supervisado del dataset (predicción de fallecimiento), aquí **no se utiliza `DEATH_EVENT` como variable de entrenamiento**. En cambio, se emplea a posteriori como validación clínica externa para verificar si los conglomerados capturan diferencias reales en la evolución de los pacientes.

---

## 2. Diagrama de flujo del proyecto

![Diagrama de Flujo del Proyecto](https://raw.githubusercontent.com/jramosm-ds/proyecto-mle-mod1/main/artifacts/flujo_proyecto_heart_failure.png)
*(Nota: El diagrama visual resume las etapas de adquisición, preprocesamiento con Mahalanobis, optimización de K, entrenamiento y validación clínica).*

---

## 3. Descripción del dataset y diccionario de datos

**Nombre:** Heart Failure Clinical Records Dataset  
**Fuente:** [Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)  
**Formato:** CSV tabular (299 filas, 13 columnas)

### Diccionario de Datos

| Variable | Tipo | Rango | Descripción |
|---|---|---|---|
| `age` | Continua | 40.0 – 95.0 | Edad del paciente (años) |
| `anaemia` | Binaria | 0 / 1 | Presencia de anemia (disminución de glóbulos rojos) |
| `creatinine_phosphokinase` | Continua | 23.0 – 7861.0 | Nivel de la enzima CPK en sangre (mcg/L) |
| `diabetes` | Binaria | 0 / 1 | Diagnóstico de diabetes mellitus |
| `ejection_fraction` | Continua | 14.0 – 80.0 | Porcentaje de sangre expulsado por el corazón en cada contracción |
| `high_blood_pressure` | Binaria | 0 / 1 | Diagnóstico de hipertensión arterial |
| `platelets` | Continua | 25 100 – 850 000 | Conteo de plaquetas en sangre (/mL) |
| `serum_creatinine` | Continua | 0.5 – 9.4 | Nivel de creatinina sérica (mg/dL) |
| `serum_sodium` | Continua | 113.0 – 148.0 | Nivel de sodio en sangre (mEq/L) |
| `sex` | Binaria | 0 / 1 | Sexo (0: mujer, 1: hombre) |
| `smoking` | Binaria | 0 / 1 | Hábito tabáquico activo |
| `time` | Continua | 4.0 – 285.0 | Días de seguimiento clínico |
| `DEATH_EVENT` | Binaria | 0 / 1 | Fallecimiento durante el seguimiento (Variable de validación) |

---

## 4. Librerías utilizadas

Para el desarrollo del proyecto se utilizaron las siguientes librerías de Python:

- **Procesamiento de datos:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (KMeans, StandardScaler, PCA, metrics)
- **Análisis Estadístico:** `scipy` (distancia Mahalanobis, clustering jerárquico)
- **Visualización:** `matplotlib`, `seaborn`
- **Exportación:** `openpyxl`
- **LLM Integration:** `google-generativeai`

---

## 5. Model Card

### 5.1 Información General

| Campo | Detalle |
|---|---|
| **Nombre del modelo** | K-Means Fenotipos — Insuficiencia Cardíaca |
| **Versión** | 1.0.0 |
| **Tipo de modelo** | Clustering no supervisado (K-Means) |
| **Librería** | scikit-learn 1.x |
| **Framework** | Python 3.10+ |

### 5.2 Arquitectura y Parámetros

| Parámetro | Valor | Descripción |
|---|---|---|
| `n_clusters` | 4 | Número óptimo determinado por métodos de Silueta y Codo |
| `init` | k-means++ | Método de inicialización inteligente de centroides |
| `n_init` | 20 | Número de veces que el algoritmo se ejecuta con diferentes semillas |
| `max_iter` | 300 | Máximo de iteraciones por ejecución |
| `random_state` | 2020 | Semilla para asegurar la reproducibilidad |

---

## 6. Resultados y métricas de evaluación

### 6.1 Métricas de evaluación offline (clustering)

| Métrica | Valor | Interpretación |
|---|---|---|
| **Coeficiente de Silueta** | 0.1163 | Separación moderada entre clusters (típico en datos clínicos) |
| **Índice Davies-Bouldin** | 2.3085 | Proximidad entre clusters (menor es mejor) |
| **Índice Calinski-Harabasz** | 29.79 | Relación entre dispersión inter e intra-cluster |
| **Inercia (WSS)** | 2194.36 | Suma de distancias al cuadrado dentro de los clusters |

### 6.2 Métricas de evaluación online (validación clínica)

Se utiliza la variable `DEATH_EVENT` para verificar si los clusters capturan diferencias reales en el riesgo de mortalidad.

| Fenotipo | N Pacientes | Fallecidos | Tasa de Mortalidad |
|---|---|---|---|
| **Fenotipo 1** | 69 | 19 | 27.5% |
| **Fenotipo 2** | 79 | 19 | 24.1% |
| **Fenotipo 3** | 90 | 15 | 16.7% |
| **Fenotipo 4** | 45 | 33 | **73.3%** ⚠️ |

### 6.3 Perfiles de fenotipos

#### Variables Continuas (Promedios)

| Variable | Fenotipo 1 | Fenotipo 2 | Fenotipo 3 | Fenotipo 4 |
|---|---|---|---|---|
| Edad (años) | 60.5 | 59.7 | 56.9 | **71.7** |
| CPK (mcg/L) | 438.9 | 469.0 | 528.9 | 390.0 |
| Fracción eyección (%) | 39.2 | 37.7 | 40.7 | **31.1** |
| Plaquetas (/mL) | 268 881 | 260 512 | 265 005 | 229 136 |
| Creatinina sérica (mg/dL) | 1.11 | 1.17 | 1.18 | **1.99** |
| Sodio sérico (mEq/L) | 137.0 | 137.4 | 137.8 | **133.6** |
| Tiempo seguimiento (días) | 117.1 | 142.6 | **161.7** | 81.5 |

#### Variables Categóricas (% con condición)

| Condición | Fenotipo 1 | Fenotipo 2 | Fenotipo 3 | Fenotipo 4 |
|---|---|---|---|---|
| Anemia | 46.4% | 35.4% | 45.6% | 46.7% |
| Diabetes | 44.9% | 31.6% | 55.6% | 24.4% |
| Hipertensión | **100.0%** | 29.1% | 0.0% | 20.0% |
| Sexo Masculino | 39.1% | **100.0%** | 42.2% | 88.9% |
| Tabaquismo | 2.9% | **100.0%** | 0.0% | 17.8% |

### 6.4 Interpretación Clínica

| Fenotipo | Nombre descriptivo | Perfil clave | Riesgo |
|---|---|---|---|
| **F1** | Hipertenso controlado | 100% hipertensión, mayoría mujeres | Moderado |
| **F2** | Hombre fumador activo | 100% masculino, 100% tabaquismo | Moderado-bajo |
| **F3** | Joven diabético estable | Menor edad, mejor función cardíaca | Bajo |
| **F4** | Anciano con fallo multiorgánico | Mayor edad, creatinina alta, sodio bajo | **Crítico** |

---

## 7. Conclusiones

1. **Estratificación efectiva:** El algoritmo K-Means identificó 4 fenotipos con tasas de mortalidad significativamente diferentes, validando la hipótesis inicial.
2. **Fenotipo de Alto Riesgo:** El Fenotipo 4 es el más crítico (73.3% mortalidad), caracterizado por edad avanzada y deterioro de la función renal.
3. **Utilidad de IA Generativa:** La integración con LLM permite una interpretación rápida y experta de los perfiles estadísticos, facilitando la toma de decisiones clínicas.
