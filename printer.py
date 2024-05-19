import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer

# Определяем размер словаря и размерность эмбеддингов
vocab_size = 10000
embedding_dim = 100

# Создаем новую модель
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))

# Компилируем модель
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Сохраняем модель целиком
model.save('model.h5')
# Загружаем обученную модель
model = load_model('model.weights.h5')

# Функция для обработки ввода ингредиентов
def process_input(ingredients):
    # Токенизация ингредиентов
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(ingredients)
    sequences = tokenizer.texts_to_sequences(ingredients)

    # Получаем максимальную длину последовательности в наборе данных
    max_length = max([len(seq) for seq in sequences])

    # Приводим все последовательности к одинаковой длине
    padded_sequences = np.array([seq + [0] * (max_length - len(seq)) for seq in sequences])

    return padded_sequences

# Функция для получения предсказания от модели
def get_prediction(inputs):
    predictions = model.predict(inputs)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    return model.output_names[predicted_class_idx]

# Главная функция
def main():
    print("Введите ингредиенты через запятую:")
    ingredients = input().split(",")
    ingredients = [item.strip() for item in ingredients]

    inputs = process_input([" ".join(ingredients)])
    prediction = get_prediction(inputs)

    print(f"Предсказанная категория: {prediction}")

if __name__ == "__main__":
    main()
