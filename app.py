import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# --- 설정 ---
st.set_page_config(page_title="스마트팜 AI 냉방 시뮬레이터", layout="wide")
st.title("🌡️ 스마트팜 디지털 트윈: 냉방 시뮬레이터")

# --- 모델 로드 함수 ---
@st.cache_resource
def load_model():
    # 모델 파일 경로 (같은 폴더에 두세요)
    model = xgb.XGBRegressor()
    model.load_model("final_xgboost_model.json")
    return model

# --- 사이드바: 설정 ---
st.sidebar.header("환경 설정")
target_temp = st.sidebar.slider("목표 냉방 수온 (°C)", 5.0, 25.0, 12.0)
uploaded_file = st.sidebar.file_uploader("Priva 데이터 업로드 (CSV)", type=['csv'])

# --- 메인 로직 ---
if uploaded_file is not None:
    model = load_model()
    
    # 데이터 로드 및 전처리 (위의 파이썬 코드 로직과 동일하게 구현)
    df = pd.read_csv(uploaded_file, sep=';', skiprows=[1,2]) # 예시
    # ... (전처리 로직) ...
    
    if st.button("시뮬레이션 실행"):
        st.write(f"🚀 **{target_temp}°C** 수온으로 시뮬레이션을 시작합니다...")
        
        # ... (시뮬레이션 루프) ...
        # 결과: res_df 생성
        
        # 결과 그래프
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.plot(res_df['Timestamp'], res_df['Actual'], 'k', label='Actual')
        ax.plot(res_df['Timestamp'], res_df['Simulated'], 'r', label='Simulated')
        ax.fill_between(res_df['Timestamp'], res_df['Actual'], res_df['Simulated'], 
                        where=(res_df['Actual']>res_df['Simulated']), color='blue', alpha=0.1)
        ax.legend()
        st.pyplot(fig)
        
        # 데이터 다운로드
        st.download_button("결과 CSV 다운로드", res_df.to_csv().encode('utf-8'), "sim_result.csv")

else:
    st.info("좌측 사이드바에서 CSV 파일을 업로드해주세요.")