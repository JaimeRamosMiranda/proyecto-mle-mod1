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

![Diagrama de Flujo del Proyecto](https://raw.githubusercontent.com/JaimeRamosMiranda/proyecto-mle-mod1/main/artifacts/flujo_proyecto_heart_failure.png)
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

### 6.2 Caracterización clínica de los fenotipos

**Cuadro N°1: Características clínicas de acuerdo a los pacientes en cada conglomerado**

| Variables | Fenotipo 1 | Fenotipo 2 | Fenotipo 3 | Fenotipo 4 |
|---|---|---|---|---|
| | (n= 69) | (n= 79) | (n= 90) | (n= 45) |
| **Variables numéricas (Medias)** | | | | |
| Edad (años) | 60.5 | 59.7 | 56.9 | 71.7 |
| CPK (mcg/L) | 438.9 | 469.0 | 528.9 | 390.0 |
| Fracción de eyección (%) | 39.2 | 37.7 | 40.7 | 31.1 |
| Plaquetas (/mL) | 268,881 | 260,512 | 265,005 | 229,136 |
| Creatinina sérica (mg/dL) | 1.11 | 1.17 | 1.18 | 1.99 |
| Sodio sérico (mEq/L) | 137.0 | 137.4 | 137.8 | 133.6 |
| Seguimiento (días) | 117.1 | 142.6 | 161.7 | 81.5 |
| **Variables categóricas (Frecuencias)** | | | | |
| Anemia | 46.4% | 35.4% | 45.6% | 46.7% |
| Diabetes | 44.9% | 31.6% | 55.6% | 24.4% |
| Hipertensión | 100.0% | 29.1% | 0.0% | 20.0% |
| Sexo (Hombre) | 39.1% | 100.0% | 42.2% | 88.9% |
| Tabaquismo | 2.9% | 100.0% | 0.0% | 17.8% |
| **Sobrevivencia (Mortalidad)** | | | | |
| Fallecidos | 27.5% | 24.1% | 16.7% | 73.3% |

---

## 7. Conclusiones e Interpretación de Fenotipos

### Interpretación por Conglomerado

**Fenotipo 1 (n=69): "Paciente hipertenso de riesgo moderado"**  
Este grupo se caracteriza por una prevalencia del 100% de hipertensión arterial. Presenta una edad media de 60.5 años. La fracción de eyección media es de 39.2%, situándose en el límite de la disfunción sistólica moderada (referencia < 40%). Los niveles de creatinina (1.11 mg/dL) y sodio (137 mEq/L) se encuentran dentro de rangos normales. Es un grupo mayoritariamente femenino (60.9% mujeres) y con muy baja incidencia de tabaquismo (2.9%). La mortalidad registrada fue del 27.5%, un nivel intermedio comparado con el resto de los fenotipos.

**Fenotipo 2 (n=79): "Hombre fumador de riesgo moderado-bajo"**  
Este fenotipo está compuesto exclusivamente por hombres (100%) y fumadores activos (100%). La edad media es de 59.7 años. Presentan la segunda menor fracción de eyección media (37.7%), indicando un compromiso cardíaco notable. Sin embargo, su función renal es estable (creatinina 1.17 mg/dL) y los niveles de sodio son normales (137.4 mEq/L). A pesar de los factores de riesgo conductuales (tabaquismo), presentan una mortalidad del 24.1%, menor que la del grupo hipertenso (F1).

**Fenotipo 3 (n=90): "Paciente joven con perfil clínico estable"**  
Es el grupo más numeroso y el de menor edad media (56.9 años). Se caracteriza por tener la mayor fracción de eyección media (40.7%) y el mayor tiempo de seguimiento clínico (161.7 días), lo que sugiere una mejor estabilidad. Curiosamente, tiene la mayor frecuencia de diabetes (55.6%) pero una nula incidencia de hipertensión (0%) y tabaquismo (0%). Este perfil metabólico pero no vascular se traduce en la mortalidad más baja de todo el estudio (16.7%).

**Fenotipo 4 (n=45): "Paciente anciano con fallo multiorgánico y riesgo crítico"**  
Este fenotipo representa el perfil de mayor gravedad. Posee la edad media más avanzada (71.7 años). Clínicamente, presenta una combinación de factores de mal pronóstico: la fracción de eyección más baja (31.1%), hipercreatininemia (1.99 mg/dL, indicativo de insuficiencia renal) e hiponatremia leve (sodio 133.6 mEq/L, referencia normal > 135). Es un grupo mayoritariamente masculino (88.9%). Estas condiciones se reflejan en una mortalidad crítica del 73.3%, la más alta detectada, validando la capacidad del modelo para identificar pacientes en estado terminal o de alto riesgo.
