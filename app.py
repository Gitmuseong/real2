import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="심장병 예측 앱", layout="centered")
st.title("🫀 심장병 예측")

# ===== 변수 설명용 확장 패널 =====
with st.expander("협심증(Angina)란?"):
    st.write("""
    심장 근육에 혈액 공급이 부족해서 생기는 가슴 통증이나 불편감입니다.  
    운동하거나 스트레스 받을 때 증상이 심해질 수 있고, 심장병 위험 신호 중 하나입니다.
    """)

with st.expander("Oldpeak란?"):
    st.write("""
    운동 중 또는 운동 후 심전도(심장 전기 신호)에서 측정하는 값입니다.

    심장이 운동할 때 더 많은 산소를 필요로 하게 되는데, 심장 근육에 산소 공급이 부족하면 심전도 신호 중 ST 분절이라는 부분이 내려가게 됩니다.
    이 내려가는 정도를 숫자로 나타낸 것이 바로 Oldpeak입니다.
    Oldpeak 값이 클수록 심장에 산소가 부족한 상태가 더 심하다는 뜻입니다.
    ST 저하는 심장 근육에 혈액 공급이 원활하지 않아 발생하는 전기 신호 변화로, 협심증이나 심장 질환의 중요한 징후입니다.

    쉽게 말해, Oldpeak는 운동 후 심장이 얼마나 스트레스를 받았는지를 보여주는 지표입니다.
    """)


with st.expander("ST Slope 의미"):
    st.write("""
    - 오르막 (Upsloping): 정상 신호  
    - 평탄 (Flat): 약간 위험 신호  
    - 내리막 (Downsloping): 심장병 위험 신호  
    """)

# ===== 숫자 → 텍스트 매핑 딕셔너리 =====
cp_map = {0: "전형적 협심증", 1: "비전형적 협심증", 2: "비협심증성 흉통", 3: "무증상"}
fbs_map = {0: "공복혈당 정상", 1: "공복혈당 높음"}
restecg_map = {0: "정상", 1: "ST-T 이상", 2: "좌심실 비대"}
exng_map = {0: "운동 중 협심증 없음", 1: "운동 중 협심증 있음"}
st_slope_map = {0: "오르막 (upsloping)", 1: "평탄 (flat)", 2: "내리막 (downsloping)"}

# ===== 사용자 입력 받기 =====
age = st.slider("나이", 20, 80, 50)
sex = st.radio("성별", [0,1], format_func=lambda x: "여성" if x==0 else "남성")
cp = st.selectbox("가슴 통증 유형", [0,1,2,3], format_func=lambda x: cp_map[x])
trtbps = st.number_input("휴식 혈압", 90, 200, 120)
chol = st.number_input("콜레스테롤", 100, 600, 250)
fbs = st.radio("공복 혈당 > 120", [0,1], format_func=lambda x: fbs_map[x])
restecg = st.selectbox("심전도 결과", [0,1,2], format_func=lambda x: restecg_map[x])
thalachh = st.slider("최대 심박수", 60, 210, 150)
exng = st.radio("운동 중 협심증 여부", [0,1], format_func=lambda x: exng_map[x])
oldpeak = st.number_input("Oldpeak (운동 후 ST 저하)", 0.0, 10.0, 1.0, step=0.1)
st_slope = st.selectbox("ST Slope", [0,1,2], format_func=lambda x: st_slope_map[x])

# ===== 입력 요약 =====
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

# ===== 모델 로드 =====
model = joblib.load("model.pkl")

# ===== 예측용 데이터프레임 생성 =====
input_data = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, st_slope]],
    columns=["Age","Sex","ChestPainType","RestingBP","Cholesterol",
             "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"])

# ===== 인코딩 및 컬럼 정렬 =====
input_data_encoded = pd.get_dummies(input_data)
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0
input_data_encoded = input_data_encoded[model_columns]

# ===== 예측 및 확률 기반 결과 출력 =====
if st.button("예측하기"):
    proba = model.predict_proba(input_data_encoded)[0][1]
    st.write(f"🔎 예측된 심장병 위험 확률: **{proba:.2%}**")

    if proba >= 0.5:
        st.error("⚠️ 심장병 위험이 있습니다. 의사와 상담하세요.")
    else:
        st.success("✅ 심장병 위험이 없습니다.")
