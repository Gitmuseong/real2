import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("🫀 심장병 예측 앱")
st.markdown("이 앱은 건강 데이터를 기반으로 심장질환 위험 여부를 예측합니다.")

# 모델 로딩
model = joblib.load("model.pkl")

# 사용자 입력 받기
age = st.slider("나이", 20, 100, 50)
sex = st.radio("성별", [0, 1], format_func=lambda x: "여성" if x == 0 else "남성")
cp = st.selectbox("가슴 통증 유형(cp)", [0, 1, 2, 3])
trtbps = st.number_input("휴식 혈압 (trtbps)", 90, 200, 120)
chol = st.number_input("콜레스테롤 수치 (chol)", 100, 600, 250)
fbs = st.radio("공복혈당 > 120", [0, 1])
restecg = st.selectbox("심전도 결과 (restecg)", [0, 1, 2])
thalachh = st.slider("최대 심박수 (thalachh)", 60, 210, 150)
exng = st.radio("운동 중 협심증 (exng)", [0, 1])

if st.button("예측하기"):
    input_data = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng]],
                              columns=["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng"])
    result = model.predict(input_data)[0]
    if result == 1:
        st.error("⚠️ 심장병 위험이 **있습니다**. 전문가와 상담하세요.")
    else:
        st.success("✅ 심장병 위험이 **없습니다**.")

