# core/utils/helpers.py
import yaml
import os
from typing import Dict, Any
from dotenv import load_dotenv
import argparse
import logging
import json

# Configurar logging (si no lo has hecho ya)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config(config_path: str, section=None) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo YAML y la combina con variables de entorno.
    Las variables de entorno tienen prioridad.
    También valida las rutas de directorios especificadas.

    Args:
        config_path: Ruta al archivo YAML.
        section: (Opcional) Si se especifica, devuelve solo la sección indicada del archivo YAML.

    Returns:
        Un diccionario con la configuración.

    Raises:
        FileNotFoundError: Si el archivo YAML no se encuentra.
        yaml.YAMLError: Si hay un error al parsear el archivo YAML.
        ValueError: Si la sección especificada no existe.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Archivo de configuración no encontrado: {config_path}")
        raise  # Re-lanzar la excepción
    except yaml.YAMLError as e:
        logger.error(f"Error al parsear el archivo YAML: {e}")
        raise

    # --- Validación de Rutas (Mejorado) ---
    #   Validar *todas* las rutas que sean relevantes, no solo data_dir y model_save_path
    for key in ["data_dir", "model_save_path", "transformer_data_dir"]:  # Añadir más si es necesario
        if key in config:  # Verificar si la clave existe en la configuración
            path = config[key]
            if path is not None: #Si existe y no es None
              if not os.path.isabs(path):  # Si no es absoluta, se asume relativa al config
                  config_dir = os.path.dirname(config_path)
                  path = os.path.join(config_dir, path)
                  config[key] = path  # Actualizar con la ruta absoluta
              if not os.path.exists(path):
                  logger.warning(f"La ruta '{key}' ({path}) especificada en la configuración no existe.")
                  #Se podría lanzar un error, o usar un valor por defecto

    # --- Sobreescribir valores del YAML con variables de entorno ---
    if section:
        if section not in config:
            raise ValueError(f"La sección '{section}' no existe en el archivo de configuración.")
        for key, value in config[section].items():
            env_var = os.getenv(f"{section.upper()}_{key.upper()}")  # Ej: MODEL_PARAMS_HIDDEN_SIZE
            if env_var:
                try:  # Intentar convertir al tipo correcto
                    if isinstance(value, bool):
                        config[section][key] = env_var.lower() == 'true'
                    elif isinstance(value, int):
                        config[section][key] = int(env_var)
                    elif isinstance(value, float):
                        config[section][key] = float(env_var)
                    elif isinstance(value, list):
                         config[section][key] = json.loads(env_var) #Si es una lista
                    else:
                        config[section][key] = env_var
                except:
                    config[section][key] = env_var #Si falla, lo guarda como string
        return config[section]

    else:
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    env_var = os.getenv(f"{section.upper()}_{key.upper()}")
                    if env_var:
                        try:
                            if isinstance(value, bool):
                                config[section][key] = env_var.lower() == 'true'
                            elif isinstance(value, int):
                                config[section][key] = int(env_var)
                            elif isinstance(value, float):
                                config[section][key] = float(env_var)
                            elif isinstance(value, list):
                                config[section][key] = json.loads(env_var)
                            else:
                                config[section][key] = env_var
                        except:
                            config[section][key] = env_var

        return config

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena y evalúa modelos para IABET.")
    parser.add_argument("--config", type=str, default="config/tft_config.yaml",
                        help="Ruta al archivo de configuración YAML.")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Nombre del experimento a ejecutar (de config.yaml).")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directorio de datos preprocesados.")
    parser.add_argument("--model_type", type=str, default="TFT",
                        help="Tipo de modelo (TFT, TFTIndRNN, Informer, all).")  #  ¡Importante!
    # --- (Opcional) Argumentos para sobreescribir parámetros individuales ---
    # parser.add_argument("--learning_rate", type=float, help="Tasa de aprendizaje.")
    # ... (añadir más argumentos según sea necesario) ...

    return parser.parse_args()

# Cargar variables de entorno (si existe el archivo .env)
load_dotenv()