import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="심장병 예측 앱", layout="centered")
st.title("🫀 심장병 예측 앱")

# 숫자 → 텍스트 매핑 딕셔너리
cp_map = {
    0: "전형적 협심증",
    1: "비전형적 협심증",
    2: "비협심증성 흉통",
    3: "무증상"
}

fbs_map = {
    0: "공복혈당 정상",
    1: "공복혈당 높음"
}

restecg_map = {
    0: "정상",
    1: "ST-T 이상",
    2: "좌심실 비대"
}

exng_map = {
    0: "운동 중 협심증 없음",
    1: "운동 중 협심증 있음"
}

# 사용자 입력 받기 (글자로 보이도록)
age = st.slider("나이", 15, 80, 30)
sex = st.radio("성별", [0, 1], format_func=lambda x: "여성" if x == 0 else "남성")
cp = st.selectbox("가슴 통증 유형", [0, 1, 2, 3], format_func=lambda x: cp_map[x])
trtbps = st.number_input("휴식 혈압", 90, 200, 120)
chol = st.number_input("콜레스테롤 수치", 100, 600, 250)
fbs = st.radio("공복 혈당 > 120", [0, 1], format_func=lambda x: fbs_map[x])
restecg = st.selectbox("심전도 결과", [0, 1, 2], format_func=lambda x: restecg_map[x])
thalachh = st.slider("최대 심박수", 60, 210, 150)
exng = st.radio("운동 중 협심증 여부", [0, 1], format_func=lambda x: exng_map[x])
oldpeak = st.number_input("Oldpeak (운동 후 ST 저하)", 0.0, 10.0, 1.0, step=0.1)
st_slope_map = {
    0: "오르막 (upsloping)",
    1: "평탄 (flat)",
    2: "내리막 (downsloping)"
}
st_slope = st.selectbox("ST Slope", [0,1,2], format_func=lambda x: st_slope_map[x])

# 입력 요약 출력 (숫자 → 텍스트 변환 포함)
st.markdown("### 입력 요약")
st.write(f"나이: {age}세")
st.write(f"성별: {'여성' if sex == 0 else '남성'}")
st.write(f"가슴 통증 유형: {cp_map[cp]}")
st.write(f"휴식 혈압: {trtbps} mmHg")
st.write(f"콜레스테롤 수치: {chol} mg/dL")
st.write(f"공복 혈당 > 120: {fbs_map[fbs]}")
st.write(f"심전도 결과: {restecg_map[restecg]}")
st.write(f"최대 심박수: {thalachh}")
st.write(f"운동 중 협심증 여부: {exng_map[exng]}")
st.write(f"Oldpeak: {oldpeak}")
st.write(f"ST Slope: {st_slope_map[st_slope]}")

# 모델 로드
model = joblib.load("model.pkl")

# 예측
import numpy as np

input_data = pd.DataFrame([[
    age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, st_slope
]], columns=["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"])

if st.button("예측하기"):
    pred = model.predict(input_data)[0]
    if pred == 1:
        st.error("⚠️ 심장병 위험이 있습니다. 의사와 상담하세요.")
    else:
        st.success("✅ 심장병 위험이 없습니다.")
