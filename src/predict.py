"""
predict.py
----------
Script de predicción: asigna un fenotipo a nuevos pacientes con insuficiencia
cardíaca usando el modelo K-Means entrenado.

Uso:
    python predict.py --input data/nuevos_pacientes.csv
    python predict.py --input data/nuevos_pacientes.csv --model artifacts/modelo_kmeans.pkl --output artifacts/predicciones.csv

El CSV de entrada debe contener las mismas 12 variables clínicas usadas en el
entrenamiento (sin la columna DEATH_EVENT).

Salida:
    CSV con las mismas columnas del input más la columna 'fenotipo' asignado.
"""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Columnas requeridas para la predicción
REQUIRED_COLUMNS = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

FENOTYPE_PROFILES = {
    1: "Paciente hipertenso de riesgo moderado (mortalidad ~27.5%)",
    2: "Hombre fumador activo (mortalidad ~24.1%)",
    3: "Paciente joven de bajo riesgo (mortalidad ~16.7%)",
    4: "Paciente anciano con deterioro multiorgánico — ALTO RIESGO (mortalidad ~73.3%)",
}


# ─── Clases ───────────────────────────────────────────────────────────────────

class ModelLoader:
    """Carga el modelo K-Means serializado desde disco."""

    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Ruta al archivo .pkl del modelo entrenado.
        """
        self.model_path = model_path
        self.model = None

    def load(self) -> "ModelLoader":
        """Deserializa el modelo desde el archivo .pkl."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Modelo cargado desde: {self.model_path}")
        logger.info(f"  Número de clusters: {self.model.n_clusters}")
        return self


class PatientPredictor:
    """Asigna fenotipos a nuevos pacientes usando el modelo entrenado."""

    def __init__(self, model, training_data_path: str):
        """
        Parameters
        ----------
        model : KMeans
            Modelo K-Means entrenado.
        training_data_path : str
            Ruta al CSV de datos de entrenamiento estandarizados (para ajustar el scaler).
        """
        self.model = model
        self.scaler = StandardScaler()
        self._fit_scaler(training_data_path)

    def _fit_scaler(self, training_data_path: str) -> None:
        """
        Ajusta el StandardScaler con los datos de entrenamiento originales.

        Parameters
        ----------
        training_data_path : str
            Ruta al CSV de datos originales limpios (sin estandarizar).
        """
        if not os.path.exists(training_data_path):
            raise FileNotFoundError(
                f"No se encontraron los datos de entrenamiento: {training_data_path}\n"
                "Ejecuta primero preprocess.py para generarlos."
            )
        train_data = pd.read_csv(training_data_path)
        self.scaler.fit(train_data[REQUIRED_COLUMNS])
        logger.info(f"Scaler ajustado con datos de entrenamiento: {training_data_path}")

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida que el dataframe de entrada tenga las columnas requeridas.

        Parameters
        ----------
        df : pd.DataFrame
            Datos de nuevos pacientes.

        Returns
        -------
        df : pd.DataFrame
            Datos validados y ordenados.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columnas faltantes en el archivo de entrada: {missing}\n"
                f"Columnas requeridas: {REQUIRED_COLUMNS}"
            )
        if "DEATH_EVENT" in df.columns:
            logger.warning("Se encontró 'DEATH_EVENT' en el input — será ignorada para la predicción.")

        nulls = df[REQUIRED_COLUMNS].isna().sum().sum()
        if nulls > 0:
            logger.warning(f"Se encontraron {nulls} valores nulos. Las filas afectadas tendrán fenotipo NaN.")

        return df[REQUIRED_COLUMNS]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Asigna un fenotipo a cada paciente del dataframe de entrada.

        Parameters
        ----------
        df : pd.DataFrame
            Datos de nuevos pacientes (12 variables clínicas).

        Returns
        -------
        result : pd.DataFrame
            Dataframe original con columnas 'fenotipo' y 'descripcion_fenotipo' añadidas.
        """
        features = self.validate_input(df)

        # Manejar filas con nulos
        mask_valid = features.notna().all(axis=1)
        fenotipos = np.full(len(features), np.nan)

        if mask_valid.any():
            scaled = self.scaler.transform(features[mask_valid])
            labels = self.model.predict(scaled) + 1  # 1-based
            fenotipos[mask_valid] = labels

        result = df.copy()
        result["fenotipo"] = fenotipos.astype("Int64")
        result["descripcion_fenotipo"] = result["fenotipo"].map(FENOTYPE_PROFILES)

        logger.info(f"Pacientes procesados: {len(result)}")
        logger.info("Distribución de fenotipos asignados:")
        dist = result["fenotipo"].value_counts().sort_index()
        for f, n in dist.items():
            logger.info(f"  Fenotipo {f}: {n} pacientes — {FENOTYPE_PROFILES.get(f, '')}")

        return result


# ─── Función principal ────────────────────────────────────────────────────────

def main(input_path: str, model_path: str, training_data_path: str, output_path: str) -> None:
    """
    Ejecuta el pipeline de predicción para nuevos pacientes.

    Parameters
    ----------
    input_path : str
        Ruta al CSV con los nuevos pacientes.
    model_path : str
        Ruta al modelo .pkl entrenado.
    training_data_path : str
        Ruta al CSV de datos originales de entrenamiento (para ajustar el scaler).
    output_path : str
        Ruta donde se guardará el CSV con fenotipos asignados.
    """
    # Carga del modelo
    loader = ModelLoader(model_path=model_path)
    loader.load()

    # Carga de nuevos pacientes
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")
    new_patients = pd.read_csv(input_path)
    logger.info(f"Nuevos pacientes cargados: {len(new_patients)} filas")

    # Predicción
    predictor = PatientPredictor(
        model=loader.model,
        training_data_path=training_data_path
    )
    result = predictor.predict(new_patients)

    # Guardar resultado
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"Predicciones guardadas: {output_path}")
    logger.info("=== Predicción finalizada ===")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asigna fenotipos a nuevos pacientes con insuficiencia cardíaca."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al CSV con nuevos pacientes (ej: data/nuevos_pacientes.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/modelo_kmeans.pkl",
        help="Ruta al modelo entrenado (default: artifacts/modelo_kmeans.pkl)"
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default="data/datos_originales_limpios.csv",
        help="Ruta a los datos originales de entrenamiento para ajustar el scaler (default: data/datos_originales_limpios.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/predicciones.csv",
        help="Ruta de salida del CSV con fenotipos asignados (default: artifacts/predicciones.csv)"
    )

    args = parser.parse_args()
    main(
        input_path=args.input,
        model_path=args.model,
        training_data_path=args.training_data,
        output_path=args.output,
    )
