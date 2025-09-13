import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

# preprocess
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# CNN
input_shape = x_train.shape[1:]  # (32,32,3)
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3,3), activation='relu', padding='same', name='conv1'),
    Conv2D(32, (3,3), activation='relu', padding='same', name='conv2'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same', name='conv3'),
    Conv2D(64, (3,3), activation='relu', padding='same', name='conv4'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

os.makedirs("models", exist_ok=True)
checkpoint_cb = ModelCheckpoint(
    filepath="models/cnn_cifar10_best.keras",  
    save_best_only=True,
    monitor='val_accuracy'
)
earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

# training
history = model.fit(
    x_train, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_split=0.15,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=2
)

#evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# 5. Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig("cnn_cifar10_history.png")
print("Saved training plot to cnn_cifar10_history.png")

idx = 5
img = x_test[idx:idx+1] 

layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
layer_outputs = [model.get_layer(name).output for name in layer_names]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img)

first_layer_activation = activations[0]  
n_features = first_layer_activation.shape[-1]
size = first_layer_activation.shape[1]

# isplay the first 16 feature maps
n_cols = 4
n_rows = min(4, (n_features + n_cols - 1)//n_cols)
plt.figure(figsize=(n_cols*2, n_rows*2))
for i in range(min(16, n_features)):
    ax = plt.subplot(n_rows, n_cols, i+1)
    feature_map = first_layer_activation[0, :, :, i]
    plt.imshow(feature_map, aspect='auto')
    plt.axis('off')
plt.suptitle('Feature maps from layer conv1 (first 16 channels)')
plt.tight_layout()
plt.savefig("cnn_featuremaps_conv1.png")
print("Saved feature map visualization to cnn_featuremaps_conv1.png")
