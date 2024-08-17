import streamlit as st
import requests
import json
from datetime import datetime

# 背景画像のURL
background_image_url = "https://images.unsplash.com/photo-1709442849678-de8db6539014?q=80&w=2071&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

# カスタムCSSを追加して背景画像を設定
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# アプリケーションのタイトル
st.title("白馬の積雪予想アプリ")

# 今日の日付を取得して表示（小さな文字サイズ）
today = datetime.today().strftime('%Y-%m-%d')
st.markdown(f"<p style='font-size:16px;'>本日の日付: {today}</p>", unsafe_allow_html=True)

# 天候データを取得する関数
@st.cache_data
def get_weather():
    api_key = "ff1a8cecbf26165ba4586b3e728b3607"
    location = "Ōmachi, JP"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&lang=ja&units=metric"
    response = requests.get(url)
    return response.json()

# 今日の天候データを取得
weather_data = get_weather()

# 天候データを表示
st.header("現在の大町/白馬の天候")
if weather_data:
    if 'main' in weather_data:
        temp = weather_data['main']['temp']
        weather_desc = weather_data['weather'][0]['description']
        st.markdown(f"**Temperature:** {temp} °C")
        st.markdown(f"**Weather:** {weather_desc.capitalize()}")
    else:
        st.error("Unable to retrieve weather data: 'main' key not found.")
else:
    st.error("Unable to retrieve weather data.")



# 説明変数の入力フォームを作成
season = st.number_input("シーズンを入力してください。（春, 夏, 秋は0; 冬なら1）", min_value=0, max_value=1, step=1)
lowest_temp = st.number_input("最低気温の予想を入力してください。")
sunshine_duration = st.number_input("おおよその日照時間は？（例: 冬季の曇りなら3時間、雪なら0時間）")
max_snow_depth = st.number_input("現在積雪量は何センチですか？")

# 予測ボタン
if st.button("積雪予想ボタン"):
    # 入力データを辞書形式に変換
    input_data = {
        "season": season,
        "lowest_temp": lowest_temp,
        "sunshine_duration": sunshine_duration,
        "max_snow_depth": max_snow_depth
    }

    # FastAPIエンドポイントにリクエストを送信
    try:
        response = requests.post("http://127.0.0.1:8000/predict/", json=input_data)
        response_data = response.json()

        if response.status_code == 200:
            # 予測結果を表示
            prediction = response_data["prediction"]
            st.markdown(f"### 過去のデータより積雪予想はこのくらいです！（cm）: {prediction}")
        else:
            st.error(f"Error: {response_data['detail']}")

    except Exception as e:
        st.error(f"Error: {e}")

