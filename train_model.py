import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

print(f"TensorFlow Version: {tf.__version__}")

# 1. Carregamento e Preparação do Dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalização para [0, 1] e adição do canal de cor (necessário para Conv2D)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Usar tf.newaxis evita fixar a resolução (28, 28) no código, tornando-o adaptável a novos datasets.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 2. Construção da Arquitetura da CNN

# Utilização de filtros pequenos (16 e 32) são mais do que suficientes para o MNIST.
model = keras.Sequential([

    layers.Input(shape=(28, 28, 1)),
    
    # 1ª Camada Convolucional
    layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # 2ª Camada Convolucional
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Classificador
    layers.Flatten(),
    layers.Dense(64, activation="relu"),       # Camada densa pequena
    layers.Dense(10, activation="softmax")
])

print("\n--- Resumo da Arquitetura ---")
model.summary()

# --- 3. Compilação ---
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --- 4. Treinamento ---
print("\nIniciando treinamento (Focado em rapidez para Pipeline de CI)...")
start_time = time.time()

# Utilização de apenas 3 épocas. Como a arquitetura é adequada ao MNIST, 
# 3 épocas já são suficientes para passar de 95% de acurácia, respeitando os limites do CI.
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=3, 
    batch_size=128,
    verbose=1
)

training_time = time.time() - start_time
print(f"Treinamento concluído em {training_time:.2f} segundos.")

# --- 5. Avaliação de Métricas Finais ---
print("\nAvaliando modelo no conjunto de teste")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("\n" + "="*50)
print(" MÉTRICAS FINAIS DO MODELO")
print(f"Perda (Loss) no Teste    : {test_loss:.4f}")
print(f"Acurácia (Acc) no Teste  : {test_acc:.2%}")
print("="*50)

# --- 6. Salvamento do Modelo e Gestão de Artefatos ---
# O arquivo model.h5 atende à exigência do repositório (formato legado com ampla compatibilidade).
# O arquivo model.keras garante redundância no formato nativo moderno e otimizado do Keras V3.

model_path_h5 = "model.h5"
model_path_keras = "model.keras"

model.save(model_path_h5)
model.save(model_path_keras)

print("\n Artefatos gerados com sucesso:")
print(f" Formato Legado: {model_path_h5}")
print(f" Formato Nativo: {model_path_keras}")