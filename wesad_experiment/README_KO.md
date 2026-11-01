# WESAD 실험 실행 패키지

먼저 `PROTOCOL_KO.md`와 `RESULTS_KO.md`를 읽으세요. 실제 결과와 아직 수행하지 않은 HCI 사용자 실험을 구분했습니다. 실제 진행문과 질문은 `QUESTIONNAIRE_KO.md`, 사례별 설명 화면의 내용은 `STIMULI_KO.md`에 있습니다.

## 새 환경에서 재실행

Python 3.12 권장. 빈 가상환경을 만들고 아래 명령을 이 폴더에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
curl -L --fail -o data/WESAD.zip https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
python experiment.py extract --zip data/WESAD.zip
python experiment.py train
python experiment.py explain
python prepare_hci.py
```

원본 다운로드는 약 2.25GB입니다. 신뢰할 수 있는 공식 WESAD 파일만 읽으세요. pickle은 임의 코드 실행이 가능한 형식입니다. 데이터 사용 조건은 공식 페이지를 따릅니다. 원본 신호와 설치 라이브러리는 제출 압축 파일에서 제외했습니다.

`requirements.txt`는 이번 실행의 주요 버전이며 `requirements-lock.txt`는 이번에 설치한 추가 의존성 목록입니다. 실제 버전은 results/environment.json, explanation_environment.json에서도 확인할 수 있습니다. 플랫폼에 따라 추가 의존성 조정이 필요할 수 있습니다.

## 결과 파일

- results/features.csv: 60초 구간별 특징과 라벨. 참가자 ID·시간·condition·target은 모델 입력에서 제외.
- results/fold_metrics.csv: 15명의 각 LOSO fold 성능.
- results/oof_predictions.csv: 각 사례를 학습에 사용하지 않은 모델의 예측.
- results/summary.json: 참가자 평균·pooled 지표·참가자 부트스트랩 구간.
- results/explanations.json: 동일 RF 예측에 대한 SHAP·LIME·Anchors 실제 설명.
- results/hci_response_template.csv: 16명 × 16건의 배정표와 빈 응답 칸. 참여자가 응답한 결과가 아님.
- results/hci_researcher_key.csv: 정답 키. 참가자에게 제시하지 않음.
- results/folds/: 재실행 시 생성되는 fold 모델. 배포 패키지에서는 용량을 줄이기 위해 제외. 설명을 재생성하려면 train을 먼저 실행.

HCI는 `PROTOCOL_KO.md`에 따라 사람을 모집하고 수행해야 합니다. 응답 파일에서 judgment는 correct/incorrect/unsure(모델 예측이 맞음/틀림/판단 불가), accept_break_alert는 yes/no로 기록합니다. consent·안내·연습·실제 응답 수집은 이 코드가 대신하지 않습니다.

```bash
python analyze_hci.py collected_responses.csv --out hci_analysis
```

빈 응답 템플릿을 입력하면 분석을 중단합니다. 자동 분석은 오답/정답 수용률에 한정합니다. 이해도 문항·면담은 별도 기록 및 분석이 필요합니다.

## 해석

스트레스 실험 조건 분류와 생산성 향상은 다른 주장입니다. 이 실험은 전자를 평가하고 후자는 검증하지 않습니다. SHAP·LIME은 모델의 특징 활용, Anchors는 모델 출력의 국소적 일관성을 설명합니다. 건강 상태의 원인이나 휴식의 치료 효과를 증명하지 않습니다.
