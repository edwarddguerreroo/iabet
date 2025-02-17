# check_imports.py
import sys
import os
import importlib

print("sys.executable:", sys.executable)

print("\nsys.path:")
for path in sys.path:
    print(path)

print("\nVariables de entorno:")
print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'No definida')}")
print(f"  PATH: {os.environ.get('PATH', 'No definida')}")

try:
    import tensorflow as tf
    print("\nTensorFlow:")
    print(f"  Versión: {tf.__version__}")
    print(f"  Ruta: {tf.__file__}")

    try:
        import tensorflow.keras as keras
        print("\ntf.keras:")
        print(f"  Versión: {tf.keras.__version__}")
        print(f"  Ruta: {tf.keras.__file__}")

    except AttributeError as e:
        print(f"\nError al acceder a tf.keras.__version__: {e}")
        # Intenta encontrar el módulo problemático
        try:
            import keras.api._v2.keras as problem_module
            print(f"\nkeras.api._v2.keras importado desde: {problem_module.__file__}")
        except Exception as e2:
            print(f"\nNo se pudo importar keras.api._v2.keras: {e2}")

except ImportError as e:
    print(f"\nError al importar TensorFlow: {e}")

try:
    import keras
    print(f"\nVersión de Keras (si se importa directamente): {keras.__version__}")
    print(f"Ruta de Keras (si se importa directamente): {keras.__file__}")
except ImportError:
    print("\nKeras no se pudo importar directamente.")

try:
    from tensorflow.keras.layers import Dense
    print("\nImportación de tensorflow.keras.layers exitosa.")
except ImportError as e:
    print("\nError al importar tensorflow.keras.layers:", e)