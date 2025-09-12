
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

# preprocess
vocab_size = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

os.makedirs("models", exist_ok=True)
checkpoint_cb = ModelCheckpoint(
    filepath="models/lstm_imdb_best.keras",   
    save_best_only=True,
    monitor='val_accuracy'
)
earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# train
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.15,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=2
)

# evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# training history
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
plt.savefig("lstm_imdb_history.png")
print("Saved training plot to lstm_imdb_history.png")


word_index = imdb.get_word_index()
index_word = {v+3:k for k,v in word_index.items()}
index_word[0] = "<pad>"
index_word[1] = "<start>"
index_word[2] = "<unknown>"

def decode_review(encoded):
    return ' '.join(index_word.get(i, '?') for i in encoded)

sample_idx = 0
print("\nSample review (decoded):")
print(decode_review(x_test[sample_idx]))
print("True label:", y_test[sample_idx])
pred = model.predict(x_test[sample_idx:sample_idx+1])[0][0]
print("Predicted probability of positive:", float(pred))

