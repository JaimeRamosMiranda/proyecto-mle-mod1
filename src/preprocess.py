"""
preprocess.py
-------------
Script de preprocesamiento para el análisis de conglomerados de pacientes
con insuficiencia cardíaca.

Uso:
    python preprocess.py --input data/heart_failure_clinical_records_dataset1.csv
    python preprocess.py --input data/heart_failure_clinical_records_dataset1.csv --threshold 0.99

Salidas (en data/):
    datos_estandarizados.csv
    datos_originales_limpios.csv
    target.csv
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Clases ───────────────────────────────────────────────────────────────────

class DataLoader:
    """Carga y valida el dataset de entrada."""

    def __init__(self, filepath: str, target_col: str = "DEATH_EVENT"):
        """
        Parameters
        ----------
        filepath : str
            Ruta al archivo CSV.
        target_col : str
            Nombre de la columna target a separar antes del clustering.
        """
        self.filepath = filepath
        self.target_col = target_col
        self.data: pd.DataFrame = None
        self.target: pd.Series = None

    def load(self) -> "DataLoader":
        """Carga el CSV y separa la columna target."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {self.filepath}")

        self.data = pd.read_csv(self.filepath)
        logger.info(f"Dataset cargado: {self.data.shape[0]} filas × {self.data.shape[1]} columnas")

        if self.target_col not in self.data.columns:
            raise ValueError(f"Columna '{self.target_col}' no encontrada en el dataset.")

        self.target = self.data[self.target_col].copy()
        self.data = self.data.drop(columns=[self.target_col])
        logger.info(f"Variable target '{self.target_col}' separada. Variables para clustering: {list(self.data.columns)}")
        return self

    def validate(self) -> "DataLoader":
        """Verifica valores nulos y los elimina si los hay."""
        nulls = self.data.isna().sum().sum()
        if nulls > 0:
            logger.warning(f"Se encontraron {nulls} valores nulos. Eliminando filas afectadas...")
            mask = self.data.notna().all(axis=1)
            self.data = self.data[mask].reset_index(drop=True)
            self.target = self.target[mask].reset_index(drop=True)
        else:
            logger.info("Sin valores nulos detectados.")
        return self


class OutlierRemover:
    """Detecta y elimina outliers multivariantes mediante distancia de Mahalanobis."""

    def __init__(self, threshold_quantile: float = 0.975):
        """
        Parameters
        ----------
        threshold_quantile : float
            Cuantil chi-cuadrado para definir el umbral de detección (default: 0.975).
        """
        self.threshold_quantile = threshold_quantile
        self.outlier_indices_: np.ndarray = None
        self.n_removed_: int = 0

    def fit_transform(
        self,
        data: pd.DataFrame,
        target: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Detecta y elimina outliers del dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Datos estandarizados.
        target : pd.Series
            Variable target asociada.

        Returns
        -------
        data_clean : pd.DataFrame
            Dataset sin outliers.
        target_clean : pd.Series
            Target sin las filas de outliers.
        """
        cov = np.cov(data.T)
        cov_inv = np.linalg.pinv(cov)
        mean = data.mean(axis=0)
        diff = data - mean
        dist_sq = np.array([d @ cov_inv @ d for d in diff.values])
        cutoff = chi2.ppf(self.threshold_quantile, df=data.shape[1])

        self.outlier_indices_ = np.where(dist_sq > cutoff)[0]
        self.n_removed_ = len(self.outlier_indices_)

        data_clean = data.drop(data.index[self.outlier_indices_]).reset_index(drop=True)
        target_clean = target.drop(target.index[self.outlier_indices_]).reset_index(drop=True)

        logger.info(f"Outliers detectados y eliminados: {self.n_removed_}")
        logger.info(f"Pacientes finales para clustering: {len(data_clean)}")
        return data_clean, target_clean


class Preprocessor:
    """Orquesta el pipeline completo de preprocesamiento."""

    def __init__(self, threshold_quantile: float = 0.975):
        """
        Parameters
        ----------
        threshold_quantile : float
            Umbral para detección de outliers con Mahalanobis.
        """
        self.threshold_quantile = threshold_quantile
        self.scaler = StandardScaler()
        self.outlier_remover = OutlierRemover(threshold_quantile=threshold_quantile)

    def run(
        self,
        data_raw: pd.DataFrame,
        target_raw: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Ejecuta el pipeline: estandarización → detección de outliers.

        Parameters
        ----------
        data_raw : pd.DataFrame
            Datos sin estandarizar (sin columna target).
        target_raw : pd.Series
            Variable target separada.

        Returns
        -------
        data_scaled_clean : pd.DataFrame
            Datos estandarizados y sin outliers.
        data_original_clean : pd.DataFrame
            Datos en escala original y sin outliers.
        target_clean : pd.Series
            Target sin filas de outliers.
        """
        # Estandarización
        scaled_array = self.scaler.fit_transform(data_raw)
        data_scaled = pd.DataFrame(scaled_array, columns=data_raw.columns)
        logger.info("Estandarización completada (media≈0, std≈1).")

        # Eliminación de outliers sobre los datos estandarizados
        data_scaled_clean, target_clean = self.outlier_remover.fit_transform(
            data_scaled, target_raw
        )

        # Aplicar los mismos índices a los datos en escala original
        outlier_idx = self.outlier_remover.outlier_indices_
        data_original_clean = data_raw.drop(data_raw.index[outlier_idx]).reset_index(drop=True)

        return data_scaled_clean, data_original_clean, target_clean


# ─── Función principal ────────────────────────────────────────────────────────

def main(input_path: str, output_dir: str, threshold: float) -> None:
    """
    Ejecuta el pipeline de preprocesamiento y guarda los resultados.

    Parameters
    ----------
    input_path : str
        Ruta al CSV de entrada.
    output_dir : str
        Directorio donde se guardarán los archivos procesados.
    threshold : float
        Cuantil chi-cuadrado para detección de outliers Mahalanobis.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carga y validación
    loader = DataLoader(filepath=input_path)
    loader.load().validate()

    # Preprocesamiento
    preprocessor = Preprocessor(threshold_quantile=threshold)
    data_scaled, data_original, target = preprocessor.run(loader.data, loader.target)

    # Guardar resultados
    path_scaled   = os.path.join(output_dir, "datos_estandarizados.csv")
    path_original = os.path.join(output_dir, "datos_originales_limpios.csv")
    path_target   = os.path.join(output_dir, "target.csv")

    data_scaled.to_csv(path_scaled, index=False)
    data_original.to_csv(path_original, index=False)
    target.to_csv(path_target, index=False, header=True)

    logger.info(f"Archivos guardados en '{output_dir}':")
    logger.info(f"  → {path_scaled}")
    logger.info(f"  → {path_original}")
    logger.info(f"  → {path_target}")
    logger.info("Preprocesamiento finalizado.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de datos para análisis de conglomerados."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al CSV de entrada (ej: data/heart_failure_clinical_records_dataset1.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directorio de salida para los archivos procesados (default: data)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.975,
        help="Cuantil chi-cuadrado para detección de outliers Mahalanobis (default: 0.975)"
    )

    args = parser.parse_args()
    main(input_path=args.input, output_dir=args.output_dir, threshold=args.threshold)
