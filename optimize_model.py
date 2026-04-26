import tensorflow as tf
import os

model_path = "model.h5"
tflite_path = "model.tflite"

# ----- Carrega o modelo treinado -----
print(f"Carregando modelo original de: {model_path}...")
model = tf.keras.models.load_model(model_path)

# Função auxiliar para calcular o tamanho do arquivo em KB.
def get_size_kb(file_path):
    return os.path.getsize(file_path) / 1024

original_size = get_size_kb(model_path)
print(f"Tamanho do modelo original (.h5): {original_size:.2f} KB\n")

# ------ Dynamic Range Quantization -------
print(" Aplicando Otimização 1: Dynamic Range Quantization (Int8)")
converter_dynamic = tf.lite.TFLiteConverter.from_keras_model(model)

# A flag Optimize.DEFAULT converte os pesos de float32 para int8, reduzindo o tam. do modelo em quase 4x.
converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic_model = converter_dynamic.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_dynamic_model)

quantized_size = get_size_kb(tflite_path)
print(f" Modelo quantizado principal salvo em: {tflite_path}")
print(f" Tamanho gerado: {quantized_size:.2f} KB")
print(f" Redução de tamanho obtida: {((original_size - quantized_size) / original_size) * 100:.1f}%\n")

# ------ Float16 Quantization (Exploração/Aprofundamento) -------
print(" Aplicando Otimização 2: Float16 Quantization ")
converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_f16.target_spec.supported_types = [tf.float16]
tflite_f16_model = converter_f16.convert()

tflite_f16_path = "model_float16.tflite"
with open(tflite_f16_path, "wb") as f:
    f.write(tflite_f16_model)

f16_size = get_size_kb(tflite_f16_path)
print(f" Modelo Float16 salvo como: {tflite_f16_path}")
print(f" Tamanho gerado (Float16): {f16_size:.2f} KB\n")
print(f"Redução de tamanho obtida: {((original_size - f16_size) / original_size) * 100:.1f}%\n")

# --- Resumo Técnico ---
print("="*60)
print(" RESUMO DE OTIMIZAÇÃO ")
print("="*60)
print(f" Original (.h5)         : {original_size:.2f} KB")
print(f" Quantizado Int8        : {quantized_size:.2f} KB (-{((original_size - quantized_size) / original_size) * 100:.1f}%)")
print(f" Quantizado Float16     : {f16_size:.2f} KB (-{((original_size - f16_size) / original_size) * 100:.1f}%)")
print("="*60)
