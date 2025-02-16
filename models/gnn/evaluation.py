# models/gnn/evaluation.py
import tensorflow as tf
from typing import Dict
#Si se evalua por separado
def evaluate_model(model: tf.keras.Model,
                   test_data: tf.data.Dataset,
                   batch_size:int) -> Dict[str, float]:
  pass