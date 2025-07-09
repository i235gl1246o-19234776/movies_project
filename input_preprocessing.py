import joblib
import numpy as np
import pandas as pd


def preprocess(features):
    # Загружаем имена колонок из CSV
    column_names = pd.read_csv('movies.csv', nrows=0).columns

    # Удаляем колонку-таргет, если она есть
    column_names = column_names.drop(['score'])

    # Создаем DataFrame из входных данных
    input_series = pd.DataFrame([pd.Series(features, index=column_names)])

    # Обработка категориальных переменных с агрегацией "других"
    top_directors = [
        'Woody Allen', 'Clint Eastwood', 'Steven Spielberg', 'Joel Schumacher',
        'Ron Howard', 'Barry Levinson', 'Steven Soderbergh', 'Sidney Lumet',
        'Ridley Scott', 'Oliver Stone', 'Tony Scott', 'Wes Craven',
        'Garry Marshall', 'Martin Scorsese'
    ]
    top_actors = [
        'Nicolas Cage', 'Robert De Niro', 'Tom Hanks', 'Bruce Willis',
        'John Travolta', 'Denzel Washington', 'Steve Martin', 'Mel Gibson',
        'Sylvester Stallone', 'Robin Williams', 'Tom Cruise', 'Johnny Depp',
        'Eddie Murphy', 'Jeff Bridges', 'Arnold Schwarzenegger'
    ]

    input_series['director'] = input_series['director'].apply(lambda x: x if x in top_directors else 'other')
    input_series['star'] = input_series['star'].apply(lambda x: x if x in top_actors else 'other')

    # Заменяем 'nan' на 'unknown'
    input_series = input_series.replace('nan', 'unknown')

    # Заполняем пропущенные значения
    input_series = input_series.fillna('unknown')

    # Логарифмируем бюджет
    input_series['budget'] = np.log1p(input_series['budget'])

    # Определяем категориальные признаки
    categorical_columns = ['genre', 'director', 'star']
    numerical = ['budget', 'runtime']

    # Загружаем кодировщики
    ohe = joblib.load('one_hot_encoder.joblib')
    scaler = joblib.load('scaler.joblib')

    # Применяем one-hot кодирование
    encoded_data = ohe.transform(input_series[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(categorical_columns))

    # Объединяем с исходным фреймом, убираем старые колонки
    input_series = pd.concat([input_series.reset_index(drop=True), encoded_df], axis=1)
    input_series = input_series.drop(columns=categorical_columns)

    # Обработка бесконечностей
    input_series = input_series.replace([np.inf, -np.inf], np.nan)

    # Масштабирование признаков
    numerical_data = input_series[numerical]
    other_data = input_series.drop(columns=numerical)

    # Масштабируем только числовые признаки
    scaler = joblib.load('scaler.joblib')
    scaled_numerical = scaler.transform(numerical_data)

    # Преобразуем обратно в DataFrame с нужными названиями
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical)

    # Собираем всё обратно
    final_input = pd.concat([scaled_numerical_df.reset_index(drop=True), other_data.reset_index(drop=True)], axis=1)

    return final_input
