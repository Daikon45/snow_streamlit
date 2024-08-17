import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# モデルをロード
model = load_model('tuned_model')

# アプリケーションのタイトル
st.title("Total Snowfall Prediction App")

# 入力フォームの作成
st.header("Enter the following parameters:")

season = st.number_input("Season (0 for Spring, Summer, Autumn; 1 for Winter)", min_value=0, max_value=1, step=1)
lowest_temp = st.number_input("Lowest Temperature (C)")
sunshine_duration = st.number_input("Sunshine Duration (hours)")
max_snow_depth = st.number_input("Maximum Snow Depth (cm)")

# 予測ボタンが押されたときの処理
if st.button("Predict Total Snowfall"):
    # 入力データをデータフレームに変換
    input_data = pd.DataFrame({
        'Season': [season],
        'Lowest temperature (C)': [lowest_temp],
        'Sunshine duration (hours)': [sunshine_duration],
        'Maximum snow depth (cm)': [max_snow_depth]
    })

    # 予測の実行
    prediction_df = predict_model(model, data=input_data)

    # デバッグのためにデータフレームの内容を表示
    st.write(prediction_df)

    # 予測結果の取得
    prediction_column = prediction_df.columns[-1]  # 予測結果が含まれる最後の列を取得
    prediction = prediction_df[prediction_column][0]

    # 予測結果を表示
    st.markdown(f"### Predicted Total Snowfall (cm): {prediction:.3f}")
