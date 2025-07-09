
# streamlit run .\UI.py в терминале для запуска
import joblib
import streamlit as st
import numpy as np
import pandas as pd

from input_preprocessing import preprocess  # Предполагается, что функция preprocess определена в отдельном файле


# Функция для предсказания рейтинга фильма
def predict(features):
    model = joblib.load("movies_model.pkl")
    return model.predict(pd.DataFrame([features]))


# Заголовок для приложения
st.title("Предсказание рейтинга фильма")

# Ввод параметров от пользователя
genre = st.selectbox(
    "Жанр",
    options=["Drama", "Comedy", "Action", "Horror", "Sci-Fi", "Thriller", "Romance", "Adventure", "Crime", "Fantasy"]
)

top_directors = [
    'Woody Allen', 'Clint Eastwood', 'Steven Spielberg', 'Joel Schumacher',
    'Ron Howard', 'Barry Levinson', 'Steven Soderbergh', 'Sidney Lumet',
    'Ridley Scott', 'Oliver Stone', 'Tony Scott', 'Wes Craven',
    'Garry Marshall', 'Martin Scorsese', 'Quentin Tarantino',
    'Christopher Nolan', 'James Cameron', 'Tim Burton', 'David Fincher',
    'Alfred Hitchcock', 'Stanley Kubrick', 'Francis Ford Coppola',
    'Peter Jackson', 'Robert Zemeckis', 'George Lucas'
]

top_actors = [
    'Nicolas Cage', 'Robert De Niro', 'Tom Hanks', 'Bruce Willis',
    'John Travolta', 'Denzel Washington', 'Steve Martin', 'Mel Gibson',
    'Sylvester Stallone', 'Robin Williams', 'Tom Cruise', 'Johnny Depp',
    'Eddie Murphy', 'Jeff Bridges', 'Arnold Schwarzenegger', 'Leonardo DiCaprio',
    'Brad Pitt', 'Matt Damon', 'Will Smith', 'Harrison Ford',
    'Al Pacino', 'Jack Nicholson', 'Dustin Hoffman', 'Morgan Freeman',
    'Samuel L. Jackson', 'Christian Bale', 'Russell Crowe', 'Kevin Spacey'
]

# Поля ввода с выпадающими списками
director = st.selectbox(
    "Режиссер",
    options=top_directors,
    index=2,  # Steven Spielberg по умолчанию
    help="Выберите режиссера из списка или введите своего"
)

star = st.selectbox(
    "Актер",
    options=top_actors,
    index=2,  # Tom Hanks по умолчанию
    help="Выберите актера из списка или введите своего"
)

# Добавляем возможность ввода своего варианта
other_director = st.checkbox("Другой режиссер (нет в списке)")
if other_director:
    director = st.text_input("Введите имя режиссера", value="")

other_actor = st.checkbox("Другой актер (нет в списке)")
if other_actor:
    star = st.text_input("Введите имя актера", value="")

budget = st.number_input(
    "Бюджет (в млн. долларов)",
    min_value=0.1,
    max_value=500.0,
    value=50.0,
    step=0.1
)

runtime = st.number_input(
    "Продолжительность (в минутах)",
    min_value=60,
    max_value=240,
    value=120,
    step=1
)

# Сбор всех параметров в словарь
features = {
    'genre': genre,
    'director': director,
    'star': star,
    'budget': budget,
    'runtime': runtime
}

# Кнопка для выполнения предсказания
if st.button("Предсказать рейтинг"):
    # Преобразуем бюджет в логарифмическую шкалу (как в ноутбуке)
    features['budget'] = np.log1p(features['budget'])

    result = predict(features)
    # Ограничение результата от 0 до 10
    result = max(0, min(10, result))

    st.success(f"Предсказанный рейтинг фильма: {result[0]:.2f} из 10")