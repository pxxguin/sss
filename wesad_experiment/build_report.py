"""Build a Korean report and researcher stimulus sheets from real saved outputs."""
from pathlib import Path
import json,html
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'

def main():
    s=json.loads((OUT/'summary.json').read_text());f=pd.read_csv(OUT/'features.csv');m=pd.read_csv(OUT/'fold_metrics.csv');c=json.loads((OUT/'explanations.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    lines=['# WESAD 실제 실행 결과','',
    '2026-11-01 실행. 공식 원본을 다운로드해 특징 추출·모델 학습·LOSO 검증·설명 생성을 수행했다. HCI 사람 대상 실험은 아직 수행하지 않았다. 합성 데이터로 성능을 산출하지 않았다.','',
    '## 데이터와 검증','',
    f'- 원본 ZIP: 2,249,444,501 bytes. 공식 응답의 MD5 e328980e7992502de30c96ba2a7cb3e2와 일치.',
    f'- 15명, 비중첩 60초 구간 {len(f)}개, 입력 특징 {len(f.columns)-5}개.',
    f'- 평상 {(f.condition==1).sum()}개 / 스트레스 {(f.condition==2).sum()}개 / 즐거움 {(f.condition==3).sum()}개.',
    '- 평상·즐거움은 비스트레스로 결합. 나머지 라벨 제외. 각 조건 경계 양끝 5초 제거.',
    '- 14명 학습·1명 평가의 LOSO 15 folds. 튜닝하지 않은 고정 모델 설정.',
    f'- 단순 맥박 품질 기준 실패 구간: {v["pulse_quality_failed_windows"]}개. 이 기준 통과가 의료적 신호 품질을 보증하지는 않는다.','',
    '## 성능','',
    '아래는 참가자별 지표를 먼저 산출한 후 15명에서 평균한 값이다. CI는 참가자 단위 10,000회 percentile bootstrap이다. 같은 사람의 60초 창을 독립 표본으로 부트스트랩하지 않았다.','',
    '| 모델 | Macro-F1 | 95% CI | Balanced accuracy | 스트레스 재현율 | AUROC |',
    '|---|---:|---|---:|---:|---:|']
    names={'dummy':'다수 클래스 기준선','logistic':'로지스틱 회귀','random_forest':'랜덤 포레스트'}
    for key in ['dummy','logistic','random_forest']:
        x=s[key]['subject_mean'];ci=s[key]['macro_f1_subject_bootstrap_95ci']
        lines.append(f'| {names[key]} | {x["macro_f1"]:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {x["balanced_accuracy"]:.3f} | {x["stress_recall"]:.3f} | {x["roc_auc"]:.3f} |')
    cm=s['random_forest']['confusion_matrix_true_rows_pred_columns']
    lines+=['','로지스틱 회귀의 평균 Macro-F1이 근소하게 높다. 이 차이의 유의성이나 우월성을 입증하지 않았다. 설명 비교 대상은 사전에 고정한 랜덤 포레스트다. 원논문과 창·특징·설정이 달라 논문 재현 점수로 부르지 않는다.','',
    '랜덤 포레스트 전체 OOF 혼동행렬(행=실제 실험 조건, 열=예측):','',
    '| 실제 / 예측 | 비스트레스 | 스트레스 |','|---|---:|---:|',f'| 비스트레스 | {cm[0][0]} | {cm[0][1]} |',f'| 스트레스 | {cm[1][0]} | {cm[1][1]} |','',
    '## 개인차와 실패 사례','',
    '| 평가 참가자 | RF Macro-F1 | 스트레스 재현율 | AUROC |','|---|---:|---:|---:|']
    for r in m[m.model=='random_forest'].itertuples():lines.append(f'| {r.subject} | {r.macro_f1:.3f} | {r.stress_recall:.3f} | {r.roc_auc:.3f} |')
    lines+=['','S13과 S14는 AUROC가 높지만 고정 임계값 0.5에서 성능이 떨어진다. 이는 점수의 순위 분리와 결정 임계값의 일반화가 별개임을 보여준다. 테스트 참가자 라벨로 임계값을 사후 조절하지 않았다. 15명뿐인 실험실 데이터이므로 실제 사용자의 일상 스트레스 감지 성능으로 일반화할 수 없다.','',
    '## 실제 설명 생성','',
    f'RF의 TP/TN/FP/FN에서 각 4개, 총 {len(c)}개 OOF 사례를 선정했다. 서로 다른 참가자를 우선했고 seed=42를 사용했다. 설명 품질이 좋아 보이는 사례로 사후 교체하지 않았다. 오류 유형을 균형 표집했으므로 이 16개에서 계산한 정확도는 전체 모델 정확도가 아니다.','',
    '| 사례 | 평가 참가자 | 유형 | 모델 스트레스 점수 | LIME 국소 R² | Anchor precision | coverage | 0.95 충족 |','|---|---|---|---:|---:|---:|---:|---|']
    for x in c:
        a=x['anchor'];lines.append(f'| {x["case_id"]} | {x["subject"]} | {x["bucket"]} | {x["probability"]:.3f} | {x["lime"]["local_r2"]:.3f} | {a["precision"]:.3f} | {a["coverage"]:.3f} | {"예" if a["threshold_met"] else "아니오"} |')
    err=max(x['shap']['additivity_error'] for x in c)
    low=min(x['lime']['local_r2'] for x in c);high=max(x['lime']['local_r2'] for x in c);failed=sum(not x['anchor']['threshold_met'] for x in c)
    lines+=['',f'이번 LIME 국소 R² 범위는 {low:.3f}–{high:.3f}로, 3개 조건을 사용한 근사 충실도가 제한적이다. Anchors는 {failed}/{len(c)}개 사례에서 목표 precision 0.95를 충족하지 못했다. 설명을 생성했다는 사실과 설명이 충분히 충실하다는 주장은 구분해야 한다.','',f'SHAP 기준값 + 전체 기여도 합의 최대 모델 출력 오차는 {err:.3e}이다. 최종 허용 오차는 0.000001(확률 단위; 0.0001%p)로 설정했다. 상위 3개만 화면에 보여주므로 표시된 3개 합만으로 모델 점수가 재구성되는 것은 아니다.','',
    'LIME의 R²는 가상으로 생성한 근방에서 단순 모델이 RF를 근사하는 정도다. 사람별 일반화 성능이 아니다. Anchor precision은 정답률이 아니라 샘플링한 입력에서 모델 예측이 유지되는 비율이다. coverage가 낮으면 규칙이 적용되는 범위가 좁다. 설명이 간단해도 맞는 예측이라는 보장은 없다.','',
    '최종 SHAP은 tree_path_dependent 방식이다. 최초 interventional 방식에서 한 사례의 합 불일치(약 0.0030585)를 확인해 전체 사례의 SHAP을 재계산했다. 훈련 트리 경로 빈도를 배경으로 쓰는 방식이므로 상관 특징 처리 가정도 달라진다. 모델 성능·예측·LIME·Anchors는 변경하지 않았다. 자세한 변경 이유는 PROTOCOL_KO.md에 기록했다.','',
    '각 사례의 전체 설명은 results/explanations.json, 읽기 쉬운 연구자용 설명 시트는 STIMULI_KO.md에 있다. 실제 피험자에게는 해당 조건의 패널 하나만 보여주고 정답/참가자 ID는 숨겨야 한다.','',
    '## HCI 사용자 실험 진행 상태','',
    '- 수행 전: 성인 16명 탐색 파일럿, 4조건 참가자 내 비교.',
    '- 준비 완료: Williams 순서 + 사례 묶음 순환으로 생성한 16명×16건 배정표.',
    '- 응답 양식은 전부 비어 있다. 이해도·신뢰·행동 변화에 대한 실험 결과는 없다.',
    '- 주 지표: 오답 예측을 맞다고 받아들이는 비율. 조건별 이해도·판단 시간은 보조 지표.',
    '- 실제 동의·모집·화면 제시·응답 수집 후 analyze_hci.py로 수용률을 분석할 수 있다.','',
    '## 검증과 발표에서의 주장','',
    '15개 fold의 참가자 분리, 입력에서 메타데이터 제외, 60초 창 비중첩, OOF 예측의 유일성, Macro-F1 독립 재계산, 설명과 해당 fold 예측의 일치, SHAP 합 일치, 배정표 균형을 검사했다. 빈 응답 양식의 분석은 거부하도록 했다. 상세 기록은 results/verification.json.','',
    '**발표 가능한 결론:** “WESAD 손목 센서 17개 특징에 전통적 ML을 적용해 참가자 독립 검증을 수행했고, 동일 모델의 실제 예측을 세 설명 방식으로 제시할 수 있었다. 평균 성능과 달리 일부 참가자에서는 큰 오류가 발생했다.”','',
    '**아직 검증하지 않은 주장:** 설명이 사용자 이해도를 높인다 / 알림이 스트레스를 줄인다 / 앱이 생산성을 높인다.','',
    '출처: Schmidt et al., ICMI 2018, https://doi.org/10.1145/3242969.3242985 ; 공식 배포 https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html . 본 연구는 과학·비상업적 수업 실험이다.']
    (ROOT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    stimuli=['# 연구자용 실제 예측 설명 시트','', '정답·대상자 ID가 포함되어 있다. 이 파일 전체를 피험자에게 보여주지 않는다. A/B/C/D 중 배정된 조건만 제시하고 공통 정보와 센서 특징 요약은 동일하게 유지한다. B와 C의 수치는 의미가 다르다. 한국어 특징 설명은 FEATURE_DICTIONARY_KO.md 참조.','']
    for x in c:
        stimuli += [f'## {x["case_id"]}', '',f'연구자 키: {x["subject"]}, {x["bucket"]}, 실제 label={x["target"]}.','',f'공통 패널: 예측={"스트레스" if x["prediction"] else "비스트레스"}; 모델 스트레스 출력 점수={x["probability"]:.3f}. 개인별 스트레스 확률이나 진단이 아님.','', '공통 특징 요약(모든 조건에 같은 표 사용):','', '| 특징 | 값 |','|---|---:|']
        for k,val in x['features'].items():stimuli.append(f'| {k} | {val:.5g} |')
        stimuli += ['', '### A — 설명 없음','','추가 설명 없음.','', '### B — SHAP','','스트레스 모델 출력에 대한 상위 3개 기여도. 양수는 스트레스 출력 증가, 음수는 감소.','']
        for item in x['shap']['top3']:stimuli.append(f'- {item["feature"]}: {item["value"]:+.4f}')
        stimuli+=['',f'배경 기준 출력={x["shap"]["base"]:.3f}. 나머지 특징 기여도는 생략됨.','', '### C — LIME','','같은 예측 주변에서 모델을 근사한 상위 3개 조건. 가중치의 양수/음수는 근사 모델에서 스트레스 방향/반대 방향.','']
        for condition,weight in x['lime']['top3']:stimuli.append(f'- {condition}: {weight:+.4f}')
        stimuli+=['',f'국소 근사 R²={x["lime"]["local_r2"]:.3f}.','', '### D — Anchors','']
        a=x['anchor']
        if a['threshold_met'] and a['rule']:stimuli.append('규칙: '+' AND '.join(a['rule']))
        else:stimuli.append('목표 precision을 충족하는 충분한 규칙을 찾지 못함. 후보 규칙: '+(' AND '.join(a['rule']) or '(없음)'))
        stimuli += ['',f'샘플링한 입력에서의 모델 예측 유지 비율(precision)={a["precision"]:.3f}; 적용 범위(coverage)={a["coverage"]:.3f}. 실제 정답률이 아님.','']
    (ROOT/'STIMULI_KO.md').write_text('\n'.join(stimuli)+'\n',encoding='utf-8')
    print('Generated RESULTS_KO.md and STIMULI_KO.md from actual outputs.')
if __name__=='__main__':main()
