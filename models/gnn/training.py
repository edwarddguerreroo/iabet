# models/gnn/training.py

#Si se entrena la GNN por separado
import tensorflow as tf
from typing import Callable, Optional, Dict, Tuple, List, Any

def train_one_epoch(model: tf.keras.Model, train_data: tf.data.Dataset,
                   optimizer: tf.keras.optimizers.Optimizer,
                   criterion: Callable, batch_size: int) -> float:
    """Entrena el modelo por una época."""
    total_loss = 0
    num_batches = 0
    return total_loss / (num_batches + 1)

def evaluate_epoch(model: tf.keras.Model, val_data: tf.data.Dataset,
                   criterion: Callable, batch_size: int) -> float:
    """Evalúa el modelo en una época."""
    total_loss = 0
    num_batches = 0

    return total_loss / (num_batches + 1)
def train_gnn(model: tf.keras.Model,
              train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              learning_rate: float,
              batch_size: int,
              epochs: int,
              callbacks: Optional[List[Any]] = None,
              verbose: int = 1) -> tf.keras.callbacks.History:
    pass