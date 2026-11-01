# WESAD Explain Lab

웨어러블 스트레스 분류와 설명 가능한 AI를 탐색하는 **HCI 수업 프로젝트**입니다. 숫자를 바꾸면 실제 랜덤 포레스트와 Tree SHAP을 브라우저에서 실행합니다.

**시연 웹:** https://wesad-explain-lab-pxxguin.janghongseo106548.chatgpt.site

## 웹에서 해볼 수 있는 것

- 실제 WESAD 검증 사례 16개: 스트레스·비스트레스·오탐·미탐.
- 17개 센서 특징의 숫자 입력과 주요 6개 특징 슬라이더.
- 기본값 복원, 훈련 표본에 작은 변화를 준 가상 입력 생성.
- 실제 RF의 스트레스 출력 점수, 임계값 0.5, 실제 라벨과의 비교.
- SHAP 기여도, LIME 국소 근사, Anchors 규칙과 품질 지표.
- 현재 입력과 결과 JSON 다운로드, 실제 실험 결과와 해석 가이드.

임의값에는 정답 라벨이 없습니다. 임의 입력과 실제 검증 사례를 화면에 구분해서 표시합니다. 높은 점수는 개인별 스트레스 확률이나 진단이 아닙니다.

## 실행

외부 JavaScript 프레임워크나 패키지 설치가 필요하지 않습니다. Node.js 20+와 Python 3이 있으면 실행할 수 있습니다. **공개 저장소에는 실제 생체 특징값·사례별 설명·모델 자산을 포함하지 않습니다.** 아래 데이터 생성 절차를 먼저 진행하거나 계정 전용 시연 웹을 사용하세요.

```bash
npm test
npm run build
python -m http.server 8000 --directory dist
```

http://localhost:8000 에 접속합니다. `index.html`을 file://로 직접 열면 ES modules, 데이터 fetch, Web Worker가 동작하지 않을 수 있으므로 HTTP 서버를 사용하세요.

## 설명의 출처를 구분하세요

| 입력 | RF 예측 | SHAP | LIME | Anchors |
|---|---|---|---|---|
| 변경하지 않은 16개 원본 사례 | 실제 LOSO RF | 저장된 Python SHAP | 저장된 Python LIME | 저장된 Alibi AnchorTabular |
| 칼럼 값을 변경한 가상 입력 | 동일 RF의 브라우저 추론 | 정확한 path-dependent Tree SHAP | 브라우저용 LIME 근사 | 브라우저용 anchor-style 근사 |

브라우저 RF와 Tree SHAP은 **16개 원본 사례의 모든 예측값 및 각 SHAP 값**을 Python과 비교하는 테스트를 통과했습니다. RF는 float32 입력으로 분기합니다. Tree SHAP 구현은 SHAP의 MIT 라이선스 알고리즘을 JavaScript로 옮겼으며 저작권 고지는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있습니다.

값을 바꾼 경우 LIME은 768개 샘플·분위수 구간 표시·가중 Ridge·상위 3개 특징 재적합을 사용합니다. Python LIME과 동일한 난수 생성·특징 선택 구현이 아닙니다. 규칙 탐색은 후보당 256개 샘플·최대 3개 조건·beam 2를 사용하며, Alibi의 통계적 신뢰한계 기반 탐색이 아닙니다. 웹은 이 차이를 각 설명 패널에 표시합니다. **공식 HCI 비교 실험에는 저장된 Python 결과와 실험 프로토콜을 사용하세요.**

웹은 설명 탐색용이며 HCI 설문의 자동 수집 시스템은 아닙니다. 입력은 브라우저 안에서 계산하고 서버에 저장하지 않습니다. Google Fonts를 제외한 실행 코드·모델·결과는 정적 파일입니다. 인터넷이 없으면 시스템 글꼴로 표시됩니다.

## 실제 오프라인 실험

- WESAD 참가자 15명, 손목 센서, 비중첩 60초 창 526개, 특징 17개.
- 평상·즐거움 → 비스트레스, 스트레스 조건 → 스트레스. 각 라벨 경계 5초 제외.
- 참가자 독립 LOSO 15 folds, 고정 설정, 학습 fold 안에서만 결측 대치·정규화.
- 참가자 평균 Macro-F1: 기준선 0.412, 로지스틱 회귀 0.862, RF 0.858.
- RF의 참가자별 Macro-F1은 0.222–1.000. 평균만으로 개인 성능을 보장할 수 없습니다.

[집계 결과](wesad_experiment/results/summary.json) · [실험 프로토콜](wesad_experiment/PROTOCOL_KO.md) · [질문지](wesad_experiment/QUESTIONNAIRE_KO.md) · [재현 안내](wesad_experiment/README_KO.md)

HCI 참가자 16명 배정표와 빈 응답 양식은 준비되어 있지만 실제 사람 대상 실험은 수행 전입니다. 생산성 개선·휴식 효과·건강 진단을 검증한 프로젝트가 아닙니다.

## 폴더

```text
web/                       브라우저 웹 소스와 실제 RF JSON
  engine.js                RF, Tree SHAP, 브라우저 근사 설명기
  explain-worker.js        입력 편집 시 별도 스레드에서 설명 계산
  data/models/             검증 사례별 참가자 제외 RF 및 훈련 참조값
  data/explanations.json   실제 Python 설명 16개
scripts/export_web_models.py  Python fold 모델 → 브라우저 JSON
scripts/build.mjs          web/ → dist/ 복사
wesad_experiment/          오프라인 코드·문서·검증 결과·배정표
tests/                    브라우저 엔진의 Python 일치 검증
```

공개 저장소에는 코드·프로토콜·빈 질문지·집계 성능 지표만 포함합니다. 원본 신호, 행 단위 생체 특징, 사례별 예측·설명, 웹 모델·훈련 참조값, Python vendor/, 캐시, 쿠키와 인증정보는 제외합니다. 웹 모델과 실제 사례는 계정 전용 시연 웹에서 사용하며, GitHub 체크아웃에서 실행하려면 공식 데이터를 직접 내려받아 생성해야 합니다. 생성된 자산은 .gitignore로 제외됩니다.

## 로컬 데이터 생성과 모델 내보내기

먼저 `wesad_experiment/README_KO.md`의 환경을 설치하고 공식 데이터를 다운로드한 뒤 extract → train → explain 순으로 실행합니다. 이때 results/features.csv, explanations.json, folds/가 생성됩니다. HCI 배정표와 연구자용 시트도 로컬에서 생성할 수 있습니다. 실제 생체 특징을 공개 Git에 추가하지 마세요.

```bash
python scripts/export_web_models.py
npm test
npm run build
```

`--source /path/to/results`로 다른 결과 폴더를 지정할 수 있습니다. 원본 분석 결과와 일치하는 fold 모델을 사용하세요. 데이터 자산이 없을 때는 synthetic unit test만 실행하고 Python 일치 통합 테스트는 skip됩니다. 생성 후에는 16개 실제 사례 일치 테스트도 실행됩니다.

## 데이터 출처와 조건

Schmidt et al. (2018), *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*, ICMI. DOI: https://doi.org/10.1145/3242969.3242985

공식 배포: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

WESAD는 출처를 명시하는 과학·비상업적 이용 조건입니다. 데이터와 파생 특징·모델을 사용할 때 원본 조건을 확인하세요. SHAP의 별도 라이선스는 제3자 고지를 참조하세요.
