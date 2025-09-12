"""
ffn_mnist.py
Feedforward neural network (MNIST)
Run: python ffn_mnist.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1. Load & preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten for feedforward net
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)

# One-hot labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Build model
model = Sequential([
    InputLayer(input_shape=(28*28,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
os.makedirs("models", exist_ok=True)
checkpoint_cb = ModelCheckpoint(
    filepath="models/ffn_mnist_best.keras",   # ✅ new format
    save_best_only=True,
    monitor='val_accuracy'
)
earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

# 3. Train
history = model.fit(
    x_train_flat, y_train_cat,
    epochs=30,
    batch_size=128,
    validation_split=0.15,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=2
)

# 4. Evaluate
test_loss, test_acc = model.evaluate(x_test_flat, y_test_cat, verbose=0)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# 5. Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("ffn_mnist_history.png")
print("Saved training plot to ffn_mnist_history.png")
