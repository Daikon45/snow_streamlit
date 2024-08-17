import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# モデルのロード
def load_model():
    with open('tuned_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# データのロード
def load_data():
    data = pd.read_csv('converted_data.csv')
    return data

# ユーザーインターフェースの構築
def main():
    st.title("モデル予測インターフェイス")
    st.write("CSVデータからモデルを使用して予測を行います。")

    # データの表示
    data = load_data()
    st.write("データプレビュー:")
    st.write(data.head())

    # 入力フィーチャーの選択
    st.sidebar.header("入力フィーチャーの選択")
    selected_features = st.sidebar.multiselect(
        "フィーチャーを選択してください",
        data.columns.tolist(),
        default=data.columns.tolist()
    )

    # ユーザー入力データの取得
    user_input = {}
    for feature in selected_features:
        value = st.sidebar.number_input(f'{feature}', value=float(data[feature].mean()))
        user_input[feature] = value

    user_input_df = pd.DataFrame(user_input, index=[0])

    # モデルの予測
    model = load_model()
    if st.sidebar.button("予測を実行"):
        scaler = StandardScaler()
        user_input_scaled = scaler.fit_transform(user_input_df)
        prediction = model.predict(user_input_scaled)
        st.write("予測結果:")
        st.write(prediction)

if __name__ == '__main__':
    main()


