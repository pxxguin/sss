// Explanatory copy is about model output attribution, not causal physiology.
export const definitions={
 eda_mean:'60초 구간에서 측정한 손목 피부전도의 평균입니다.',
 eda_std:'60초 구간의 피부전도가 평균 주위에서 얼마나 변했는지를 나타냅니다.',
 eda_range:'구간 내 피부전도의 최댓값과 최솟값 차이입니다.',
 eda_slope:'구간 안에서 시간에 따라 피부전도가 변한 선형 추세입니다. 양수는 측정값의 상승 추세입니다.',
 eda_positive_derivative_mean:'피부전도 변화 중 증가한 부분의 평균입니다. 정교한 피부전도 반응 횟수와는 다릅니다.',
 temp_mean:'60초 구간에서 측정한 손목 피부온도의 평균입니다.',
 temp_std:'구간 내 피부온도의 변동 정도입니다.',
 temp_range:'구간 내 피부온도의 최댓값과 최솟값 차이입니다.',
 temp_slope:'시간에 따른 피부온도의 선형 추세입니다. 양수는 온도 상승 추세, 음수는 하강 추세입니다.',
 acc_magnitude_mean_g:'중력을 포함한 3축 가속도 벡터 크기의 평균입니다.',
 acc_magnitude_std_g:'가속도 벡터 크기가 구간 안에서 변한 정도입니다.',
 acc_jerk_mean_g_s:'연속한 가속도 값의 변화 속도 크기를 평균한 값입니다.',
 bvp_filtered_std:'대역통과 필터를 적용한 손목 맥파의 변동 정도입니다. 기기 상대 단위입니다.',
 pulse_bpm:'손목 맥파에서 구한 유효 맥박 간격의 중앙값으로 추정한 분당 맥박수입니다.',
 pulse_interval_std_ms:'유효 맥박 간격들의 표준편차입니다.',
 pulse_rmssd_ms:'인접한 유효 맥박 간격 차이의 제곱평균제곱근입니다. 맥파 기반 PRV이며 ECG 기반 HRV와 같다고 단정하지 않습니다.',
 pulse_quality_ok:'단순한 맥박 검출 품질 기준을 통과하면 1입니다. 의료적 신호 품질을 보장하지는 않습니다.'};
export function direction(value,epsilon=1e-8){return Math.abs(value)<epsilon?'neutral':value>0?'up':'down';}
export function attributionText(label,value){const d=direction(value),amount=(Math.abs(value)*100).toFixed(2);return d==='neutral'?`이 입력에서 ${label}에 배분된 스트레스 출력 기여는 거의 0입니다.`:`이 입력에서 ${label}에는 기준 출력에 비해 스트레스 모델 점수를 ${amount}%p ${d==='up'?'높이는':'낮추는'} 기여가 배분되었습니다.`;}
export function deltaText(current,original){const delta=current-original,d=direction(delta),amount=(Math.abs(delta)*100).toFixed(2);return d==='neutral'?'원본 사례와 비교해 이 특징의 기여도 변화가 거의 없습니다.':`원본 사례보다 이 특징에 배분된 기여도가 ${amount}%p ${d==='up'?'증가':'감소'}했습니다. 다른 특징의 기여도도 함께 바뀔 수 있습니다.`;}
export function points(value,digits=2){const v=Math.abs(value)<.5*10**(-digits)/100?0:value;return `${v>0?'+':''}${(v*100).toFixed(digits)}%p`;}
