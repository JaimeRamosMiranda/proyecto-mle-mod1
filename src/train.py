"""
train.py
--------
Script de entrenamiento del modelo K-Means para identificación de fenotipos
de pacientes con insuficiencia cardíaca.

Uso:
    python train.py
    python train.py --data_dir data --artifacts_dir artifacts --k 4
    python train.py --k 3 --n_init 30 --random_state 42

Requiere haber ejecutado preprocess.py primero.

Salidas (en artifacts/):
    modelo_kmeans.pkl
    scaler.pkl
    heart_failure_con_fenotipos.csv
    cuadros_fenotipos.xlsx
    clusters_kmeans_k{k}.png
    perfiles_centroides.png
    mortalidad_por_fenotipo.png
    metodo_silueta.png
    metodo_codo.png
"""

import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COLORES = ["#E05C5C", "#4C8CBF", "#5BAD72", "#F0A500",
           "#9B59B6", "#1ABC9C", "#E67E22", "#2C3E50"]


# ─── Clases ───────────────────────────────────────────────────────────────────

class ClusterOptimizer:
    """Evalúa múltiples valores de k para determinar el número óptimo de clusters."""

    def __init__(self, k_range: range = range(2, 11), random_state: int = 2020):
        """
        Parameters
        ----------
        k_range : range
            Rango de valores de k a evaluar.
        random_state : int
            Semilla para reproducibilidad.
        """
        self.k_range = k_range
        self.random_state = random_state
        self.silhouette_scores_: list = []
        self.inertias_: list = []

    def fit(self, data: pd.DataFrame) -> "ClusterOptimizer":
        """
        Calcula silueta e inercia para cada k en el rango definido.

        Parameters
        ----------
        data : pd.DataFrame
            Datos estandarizados.
        """
        for k in self.k_range:
            km = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=self.random_state)
            labels = km.fit_predict(data)
            self.silhouette_scores_.append(silhouette_score(data, labels))
            self.inertias_.append(km.inertia_)
            logger.info(f"  k={k} → silueta={self.silhouette_scores_[-1]:.4f}, inercia={self.inertias_[-1]:.2f}")
        return self

    @property
    def best_k_silhouette(self) -> int:
        """k con mayor coeficiente de silueta."""
        return list(self.k_range)[self.silhouette_scores_.index(max(self.silhouette_scores_))]

    def plot_silhouette(self, output_path: str) -> None:
        """Guarda el gráfico del método de la silueta."""
        plt.figure(figsize=(8, 5))
        plt.plot(list(self.k_range), self.silhouette_scores_, "bo-", linewidth=2, markersize=7)
        plt.xlabel("Número de clusters k")
        plt.ylabel("Coeficiente de silueta promedio")
        plt.title("Método de la Silueta — Número óptimo de clusters")
        plt.xticks(list(self.k_range))
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Gráfico silueta guardado: {output_path}")

    def plot_elbow(self, output_path: str) -> None:
        """Guarda el gráfico del método del codo."""
        plt.figure(figsize=(8, 5))
        plt.plot(list(self.k_range), self.inertias_, "bo-", linewidth=2, markersize=7)
        plt.xlabel("Número de clusters k")
        plt.ylabel("Inercia (WSS)")
        plt.title("Método del Codo — Número óptimo de clusters")
        plt.xticks(list(self.k_range))
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Gráfico codo guardado: {output_path}")


class ClusterTrainer:
    """Entrena el modelo K-Means final y genera los artefactos de resultados."""

    def __init__(self, k: int, n_init: int = 20, max_iter: int = 300, random_state: int = 2020):
        """
        Parameters
        ----------
        k : int
            Número de clusters.
        n_init : int
            Número de inicializaciones aleatorias.
        max_iter : int
            Máximo de iteraciones por inicialización.
        random_state : int
            Semilla para reproducibilidad.
        """
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: KMeans = None
        self.metrics_: dict = {}

    def fit(self, data: pd.DataFrame) -> "ClusterTrainer":
        """
        Entrena el modelo K-Means.

        Parameters
        ----------
        data : pd.DataFrame
            Datos estandarizados.
        """
        self.model = KMeans(
            n_clusters=self.k,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(data)

        self.metrics_ = {
            "k": self.k,
            "inertia": round(self.model.inertia_, 2),
            "silhouette": round(silhouette_score(data, self.model.labels_), 4),
            "davies_bouldin": round(davies_bouldin_score(data, self.model.labels_), 4),
            "calinski_harabasz": round(calinski_harabasz_score(data, self.model.labels_), 2),
        }

        logger.info(f"=== Métricas K-Means (k={self.k}) ===")
        for key, val in self.metrics_.items():
            logger.info(f"  {key}: {val}")

        dist = pd.Series(self.model.labels_ + 1).value_counts().sort_index()
        logger.info(f"Distribución de fenotipos:\n{dist.to_string()}")
        return self

    def save(self, output_path: str) -> None:
        """Serializa el modelo entrenado en un archivo .pkl."""
        with open(output_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Modelo guardado: {output_path}")

    def plot_pca(self, data: pd.DataFrame, output_path: str) -> None:
        """Visualiza los clusters proyectados en 2 componentes PCA."""
        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(data)
        var_exp = pca.explained_variance_ratio_
        colores = COLORES[:self.k]

        plt.figure(figsize=(10, 7))
        for i, c in enumerate(colores):
            mask = self.model.labels_ == i
            plt.scatter(coords[mask, 0], coords[mask, 1],
                        color=c, label=f"Fenotipo {i+1}", alpha=0.7,
                        s=50, edgecolors="white", linewidth=0.5)

        centroides_pca = pca.transform(self.model.cluster_centers_)
        plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
                    color="black", marker="X", s=250, zorder=5, label="Centroides")

        plt.xlabel(f"PC1 ({var_exp[0]*100:.1f}% varianza explicada)")
        plt.ylabel(f"PC2 ({var_exp[1]*100:.1f}% varianza explicada)")
        plt.title(f"Clusters K-Means (k={self.k}) — Proyección PCA")
        plt.legend(framealpha=0.9)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Gráfico PCA guardado: {output_path}")

    def plot_centroids_heatmap(self, data: pd.DataFrame, output_path: str) -> None:
        """Genera el heatmap de centroides estandarizados por fenotipo."""
        centroides_df = pd.DataFrame(
            self.model.cluster_centers_,
            columns=data.columns,
            index=[f"Fenotipo {i+1}" for i in range(self.k)]
        )
        plt.figure(figsize=(15, self.k + 2))
        sns.heatmap(centroides_df, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, linewidths=0.5,
                    cbar_kws={"label": "Z-score"})
        plt.title("Perfil de centroides por fenotipo (valores estandarizados)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Heatmap de centroides guardado: {output_path}")

    def plot_mortality(
        self,
        data_with_labels: pd.DataFrame,
        output_path: str
    ) -> None:
        """Grafica la tasa de mortalidad por fenotipo."""
        mort = data_with_labels.groupby("fenotipo")["DEATH_EVENT"].agg(["sum", "count"])
        mort["tasa_%"] = (mort["sum"] / mort["count"] * 100).round(1)
        colores = COLORES[:self.k]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(
            [f"Fenotipo {i}" for i in mort.index],
            mort["tasa_%"],
            color=colores, edgecolor="white", linewidth=1.2
        )
        for bar, val in zip(bars, mort["tasa_%"]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val}%", ha="center", fontsize=11, fontweight="bold")

        plt.xlabel("Fenotipo")
        plt.ylabel("Tasa de mortalidad (%)")
        plt.title("Mortalidad observada por fenotipo (validación clínica externa)")
        plt.ylim(0, 90)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Gráfico de mortalidad guardado: {output_path}")


class ResultsExporter:
    """Exporta las tablas resumen de fenotipos a Excel y CSV."""

    COLS_CONTINUAS = [
        "age", "creatinine_phosphokinase", "ejection_fraction",
        "platelets", "serum_creatinine", "serum_sodium", "time"
    ]
    COLS_BINARIAS = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
    LABELS_BINARIAS = {
        "sex": {0: "mujer", 1: "hombre"},
        "anaemia": {0: "no", 1: "sí"},
        "diabetes": {0: "no", 1: "sí"},
        "high_blood_pressure": {0: "no", 1: "sí"},
        "smoking": {0: "no", 1: "sí"},
        "DEATH_EVENT": {0: "vivo", 1: "fallecido"},
    }

    def export(
        self,
        data_original: pd.DataFrame,
        labels: np.ndarray,
        target: pd.Series,
        artifacts_dir: str
    ) -> pd.DataFrame:
        """
        Genera tablas resumen y exporta a Excel y CSV.

        Parameters
        ----------
        data_original : pd.DataFrame
            Datos en escala original (sin estandarizar).
        labels : np.ndarray
            Etiquetas de cluster asignadas por el modelo (0-based).
        target : pd.Series
            Variable DEATH_EVENT para validación clínica.
        artifacts_dir : str
            Directorio donde se guardarán los archivos.

        Returns
        -------
        df : pd.DataFrame
            Dataset completo con fenotipo y DEATH_EVENT asignados.
        """
        df = data_original.copy()
        df["fenotipo"] = labels + 1
        df["DEATH_EVENT"] = target.values

        # Tablas resumen
        perfil_cont = df.groupby("fenotipo")[self.COLS_CONTINUAS].mean().round(2)
        perfil_bin  = (df.groupby("fenotipo")[self.COLS_BINARIAS].mean() * 100).round(1)
        mort = df.groupby("fenotipo")["DEATH_EVENT"].agg(["sum", "count"])
        mort.columns = ["fallecidos", "total"]
        mort["tasa_%"] = (mort["fallecidos"] / mort["total"] * 100).round(1)

        # Etiquetas legibles para tablas categóricas
        df_labels = df.copy()
        for col, mapping in self.LABELS_BINARIAS.items():
            if col in df_labels.columns:
                df_labels[col] = df_labels[col].map(mapping)

        # Excel
        excel_path = os.path.join(artifacts_dir, "cuadros_fenotipos.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            perfil_cont.to_excel(writer, sheet_name="Variables continuas")
            perfil_bin.to_excel(writer, sheet_name="Variables binarias pct")
            mort.to_excel(writer, sheet_name="Mortalidad")
            for col in self.COLS_BINARIAS + ["DEATH_EVENT"]:
                if col in df_labels.columns:
                    (df_labels.groupby(["fenotipo", col])
                               .size()
                               .reset_index(name="Freq")
                               .to_excel(writer, sheet_name=col[:31], index=False))
        logger.info(f"Excel exportado: {excel_path}")

        # CSV con fenotipos
        csv_path = os.path.join(artifacts_dir, "heart_failure_con_fenotipos.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV exportado: {csv_path}")

        return df


# ─── Función principal ────────────────────────────────────────────────────────

def main(data_dir: str, artifacts_dir: str, k: int, n_init: int, random_state: int) -> None:
    """
    Ejecuta el pipeline de entrenamiento completo.

    Parameters
    ----------
    data_dir : str
        Directorio con los archivos de preprocesamiento.
    artifacts_dir : str
        Directorio donde se guardarán los artefactos.
    k : int
        Número de clusters.
    n_init : int
        Número de inicializaciones K-Means.
    random_state : int
        Semilla para reproducibilidad.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # Carga de datos preprocesados
    data_scaled   = pd.read_csv(os.path.join(data_dir, "datos_estandarizados.csv"))
    data_original = pd.read_csv(os.path.join(data_dir, "datos_originales_limpios.csv"))
    target        = pd.read_csv(os.path.join(data_dir, "target.csv")).squeeze()
    logger.info(f"Datos cargados: {data_scaled.shape[0]} pacientes × {data_scaled.shape[1]} variables")

    # Evaluación de k óptimo
    logger.info("Evaluando número óptimo de clusters...")
    optimizer = ClusterOptimizer(k_range=range(2, 11), random_state=random_state)
    optimizer.fit(data_scaled)
    optimizer.plot_silhouette(os.path.join(artifacts_dir, "metodo_silueta.png"))
    optimizer.plot_elbow(os.path.join(artifacts_dir, "metodo_codo.png"))
    logger.info(f"k óptimo por silueta: {optimizer.best_k_silhouette} | k seleccionado: {k}")

    # Entrenamiento
    logger.info(f"Entrenando K-Means con k={k}...")
    trainer = ClusterTrainer(k=k, n_init=n_init, random_state=random_state)
    trainer.fit(data_scaled)
    trainer.save(os.path.join(artifacts_dir, "modelo_kmeans.pkl"))
    trainer.plot_pca(data_scaled, os.path.join(artifacts_dir, f"clusters_kmeans_k{k}.png"))
    trainer.plot_centroids_heatmap(data_scaled, os.path.join(artifacts_dir, "perfiles_centroides.png"))

    # Exportar resultados
    exporter = ResultsExporter()
    df_final = exporter.export(data_original, trainer.model.labels_, target, artifacts_dir)

    # Gráfico de mortalidad
    trainer.plot_mortality(df_final, os.path.join(artifacts_dir, "mortalidad_por_fenotipo.png"))

    logger.info("=== Entrenamiento finalizado ===")
    logger.info(f"Artefactos guardados en: {artifacts_dir}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo K-Means para identificación de fenotipos."
    )
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directorio con los datos preprocesados (default: data)")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts",
                        help="Directorio de salida para artefactos (default: artifacts)")
    parser.add_argument("--k", type=int, default=4,
                        help="Número de clusters (default: 4)")
    parser.add_argument("--n_init", type=int, default=20,
                        help="Número de inicializaciones K-Means (default: 20)")
    parser.add_argument("--random_state", type=int, default=2020,
                        help="Semilla aleatoria (default: 2020)")

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        k=args.k,
        n_init=args.n_init,
        random_state=args.random_state,
    )
