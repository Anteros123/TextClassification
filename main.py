import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import numpy as np

excel_file = "Test.xlsx"
df = pd.read_excel(excel_file)
df = df.sample(frac=1, random_state=42)
print(df.head())

tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

train, val, test = np.split(df.sample(frac=1, random_state=42), [int(0.6*len(df)), int(0.8*len(df))])

def df_to_dataset(dataframe, tokenizer, shuffle=True, batch_size=8):
    df = dataframe.copy()
    labels = df.pop('label')
    df = df["sentence"]
    tokenized = tokenizer(df.tolist(), padding=True, truncation=True, return_tensors="tf")
    ds = tf.data.Dataset.from_tensor_slices(({"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_data = df_to_dataset(train, tokenizer)
valid_data = df_to_dataset(val, tokenizer)
test_data = df_to_dataset(test, tokenizer)

print(list(train_data.take(1)))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids'),
    tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask'),    
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128, mask_zero=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_data,
                    epochs=20,
                    validation_data=valid_data)

model.evaluate(test_data)

sentences_to_predict = []

tokenized_input = tokenizer(sentences_to_predict, padding=True, truncation=True, return_tensors="tf")

predictions = model.predict({
    'input_ids': tokenized_input['input_ids'],
    'attention_mask': tokenized_input['attention_mask']
})

predicted_labels = ["positive" if pred >= 0.5 else "negative" for pred in predictions]

for sentence, label in zip(sentences_to_predict, predicted_labels):
    print(f"Sentence: {sentence}")
    print(f"Predicted Label: {label}")
    print()













