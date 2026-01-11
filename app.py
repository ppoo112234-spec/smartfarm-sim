import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# --- 설정 ---
st.set_page_config(page_title="스마트팜 AI 냉방 시뮬레이터", layout="wide")
st.title("🌡️ 스마트팜 디지털 트윈: 냉방 시뮬레이터")

# --- 모델 로드 함수 ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    # 파일명이 정확한지 꼭 확인하세요! (GitHub에 올린 파일명)
    if os.path.exists("final_xgboost_model.json"):
        model.load_model("final_xgboost_model.json")
    else:
        st.error("❌ 모델 파일(final_xgboost_model.json)을 찾을 수 없습니다.")
        return None
    return model

# --- 물리 엔진 (피처 엔지니어링) ---
def calculate_physics(df_row, prev_row=None, rad_load_state=0):
    row = df_row.copy()
    T_in, T_out = row['Greenhouse_Temp'], row['Outside_Temp']
    Rad, Wind, Pipe = row['Radiation'], row['Wind_Speed'], row['Pipe_Temp']
    
    # Curtain & Vent (결측치 방지)
    c1 = row.get('Curtain_1', 0) / 100.0
    c2 = row.get('Curtain_2', 0) / 100.0
    vent_lee = row.get('Vent_Lee', 0)
    vent_wind = row.get('Vent_Wind', 0)
    
    ins_eff = 1 - (1 - c1 * 0.45) * (1 - c2 * 0.65)
    row['Thermal_Loss_Potential'] = 5.7 * (T_in - T_out) * (1 - ins_eff) * (1 + 0.1 * Wind)
    
    shade_eff = 1 - (1 - c1 * 0.25) * (1 - c2 * 0.65)
    row['Net_Solar_Gain'] = Rad * (1 - shade_eff)
    
    alpha = 2/13
    if prev_row is None: new_rad_load = row['Net_Solar_Gain']
    else: new_rad_load = (row['Net_Solar_Gain'] * alpha) + (rad_load_state * (1 - alpha))
    row['Rad_Thermal_Mass'] = new_rad_load
    
    row['Heating_Force_Lag'] = Pipe - T_in # 음수 허용
    
    vent_avg = (vent_lee + vent_wind) / 2.0
    row['Vent_Cooling_Force_Lag'] = (vent_avg / 100.0) * (T_in - T_out) * np.sqrt(Wind + 1)
    
    return row, new_rad_load

# --- 사이드바: 설정 ---
st.sidebar.header("환경 설정")
target_temp = st.sidebar.slider("목표 냉방 수온 (°C)", 5.0, 25.0, 12.0)
uploaded_file = st.sidebar.file_uploader("Priva 데이터 업로드 (CSV)", type=['csv'])

# --- 메인 로직 ---
if uploaded_file is not None:
    model = load_model()
    
    if model is not None:
        # 데이터 로드
        try:
            # Priva 포맷 (세미콜론) 대응
            df = pd.read_csv(uploaded_file, sep=';', skiprows=[1, 2])
            if df.shape[1] < 2: # 쉼표일 경우 재시도
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', skiprows=[1, 2])
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            st.stop()

        # 전처리
        df.columns = df.columns.str.strip().str.lower()
        col_date = df.columns[0]
        df.rename(columns={col_date: 'Timestamp'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)

        mapping = {
            'outside temp': 'Outside_Temp', 'radiation': 'Radiation', 'wind speed': 'Wind_Speed',
            'meas grh temp': 'Greenhouse_Temp', 'meas rh': 'Greenhouse_RH', 'meas lee': 'Vent_Lee',
            'meas wind': 'Vent_Wind', 'meas wt 3': 'Pipe_Temp', 
            'meas curtain 1': 'Curtain_1', 'meas curtain 2': 'Curtain_2'
        }
        df = df.rename(columns=mapping)
        
        # 필수 컬럼 채우기
        req_cols = ['Greenhouse_Temp', 'Outside_Temp', 'Radiation', 'Wind_Speed', 'Greenhouse_RH', 'Pipe_Temp', 'Curtain_1', 'Curtain_2', 'Vent_Lee', 'Vent_Wind']
        for c in req_cols:
            if c not in df.columns: df[c] = 0.0
            else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        st.success(f"✅ 데이터 로드 성공: {len(df)}행")

        # 시뮬레이션 버튼
        if st.button("🚀 시뮬레이션 실행"):
            with st.spinner('AI가 온실 환경을 시뮬레이션 중입니다...'):
                model_features = model.get_booster().feature_names
                sim_results = []
                curr_sim_temp = df.iloc[0]['Greenhouse_Temp']
                prev_row, rad_state = None, 0
                
                # Simulation Loop
                for i in range(len(df)):
                    row = df.iloc[i].copy()
                    row['Pipe_Temp'] = target_temp # 목표 수온 적용
                    row['Greenhouse_Temp'] = curr_sim_temp
                    
                    row_feat, rad_state = calculate_physics(row, prev_row, rad_state)
                    
                    # Predict
                    X = pd.DataFrame([row_feat])[model_features]
                    pred_delta = model.predict(X)[0]
                    next_temp = curr_sim_temp + (pred_delta * 0.85)
                    
                    sim_results.append({
                        'Timestamp': row['Timestamp'],
                        'Actual': df.iloc[i]['Greenhouse_Temp'],
                        'Simulated': next_temp
                    })
                    curr_sim_temp = next_temp
                    prev_row = row_feat
                
                # 결과 DF 생성
                res_df = pd.DataFrame(sim_results)
                
                # ---------------------------------------------------------
                # ★ [중요] 그래프 그리기 코드가 반드시 이 안에 있어야 함! ★
                # ---------------------------------------------------------
                st.write("### 📊 시뮬레이션 결과")
                
                fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
                ax.plot(res_df['Timestamp'], res_df['Actual'], 'k', alpha=0.5, label='Actual Temp')
                ax.plot(res_df['Timestamp'], res_df['Simulated'], 'r', linewidth=2, label=f'Simulated ({target_temp}°C)')
                
                # 냉방 효과 영역
                ax.fill_between(res_df['Timestamp'], res_df['Actual'], res_df['Simulated'], 
                                where=(res_df['Actual'] > res_df['Simulated']), 
                                color='blue', alpha=0.1, label='Cooling Effect')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                st.pyplot(fig)
                
                # 데이터 다운로드 버튼
                st.download_button(
                    label="📥 결과 CSV 다운로드",
                    data=res_df.to_csv(index=False).encode('utf-8-sig'),
                    file_name="simulation_result.csv",
                    mime="text/csv"
                )

else:
    st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드해주세요.")
