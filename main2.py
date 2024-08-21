from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pycaret.regression import load_model, predict_model

# モデルをロード
model = load_model('tuned_model')

# FastAPIアプリケーションの作成
app = FastAPI()

# ルートエンドポイントの追加
@app.get("/")
async def root():
    return {"message": "Welcome to the Snow Prediction API"}

# 入力データの構造を定義するPydanticモデル
class SnowPredictionInput(BaseModel):
    season: int
    lowest_temp: float
    sunshine_duration: float
    max_snow_depth: float


# 予測エンドポイントを作成
@app.post("/predict/")
async def predict_snow(data: SnowPredictionInput):
    try:
        # 入力データをデータフレームに変換
        input_data = pd.DataFrame({
            'Season': [data.season],
            'Lowest temperature (C)': [data.lowest_temp],
            'Sunshine duration (hours)': [data.sunshine_duration],
            'Maximum snow depth (cm)': [data.max_snow_depth]
        })

        # 予測の実行
        prediction_df = predict_model(model, data=input_data)

        # 予測結果の取得
        prediction_column = prediction_df.columns[-1]  # 予測結果が含まれる最後の列を取得
        prediction = prediction_df[prediction_column][0]

        # 結果を返す
        return {"prediction": round(prediction, 3)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


