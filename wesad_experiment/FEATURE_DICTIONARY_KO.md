# 특징 사전

60초 구간의 특징이다. 단위·정의가 다른 특징끼리 원시 수치의 크기를 직접 비교하지 않는다. 상관된 특징의 기여도는 설명기 설정에 따라 달라질 수 있다.

| 코드 | 한국어 의미 | 단위·해석 |
|---|---|---|
| eda_mean | 평균 피부전도 | µS, 손목 EDA 1Hz 저역통과 후 평균 |
| eda_std | 피부전도 변동 | µS, 구간 표준편차 |
| eda_range | 피부전도 범위 | µS, 최대−최소 |
| eda_slope | 피부전도 추세 | µS/s, 시간에 대한 선형 기울기 |
| eda_positive_derivative_mean | 피부전도 증가량 | µS/s, 양의 시간 차분 평균; 정교한 SCR 횟수와 다름 |
| temp_mean | 평균 피부온도 | °C |
| temp_std | 피부온도 변동 | °C |
| temp_range | 피부온도 범위 | °C |
| temp_slope | 피부온도 추세 | °C/s |
| acc_magnitude_mean_g | 평균 가속도 크기 | g, 중력 포함 |
| acc_magnitude_std_g | 가속도 크기 변동 | g |
| acc_jerk_mean_g_s | 움직임 변화량 | g/s, 3축 가속도 차분 크기 평균 |
| bvp_filtered_std | 맥파 변동 | BVP 기기 상대 단위; 0.7–3.5Hz 필터 후 표준편차 |
| pulse_bpm | 추정 맥박수 | beats/min, 유효 맥박간격 중앙값으로 계산 |
| pulse_interval_std_ms | 맥박간격 변동 | ms, 유효 맥박간격 표준편차 |
| pulse_rmssd_ms | 인접 맥박간격 차이 변동 | ms, PRV RMSSD. ECG HRV로 단정하지 않음 |
| pulse_quality_ok | 단순 맥박 품질 기준 통과 | 0 또는 1; 임상적 검증 지표가 아님 |

예측 모델에 사용하지 않는 메타데이터: subject, start_s, end_s, condition, target.
