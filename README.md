# 1-lead_ECG_to_12-lead

---

# BiLSTM + Attention 모델을 활용한 12-lead ECG 예측

이 프로젝트는 BiLSTM (양방향 LSTM)과 Attention 메커니즘을 활용하여 단일 리드 ECG 데이터를 12-lead ECG로 변환하는 모델을 구현한 것입니다. 이 모델은 주로 의료 데이터를 다루는 연구에서 활용될 수 있습니다.

## 모델 설명

- 모델 구조  
  모델은 BiLSTM을 사용하여 시계열 ECG 데이터를 처리하고, Attention 메커니즘을 통해 중요한 특징을 추출합니다. 마지막으로, Fully Connected 레이어를 통해 12-lead ECG로 예측된 출력을 생성합니다.

- 입력 데이터:  
  단일 리드 ECG 신호 (1D 시계열 데이터)를 입력으로 사용합니다.

- 출력 데이터:  
  모델은 12-lead ECG 데이터를 예측합니다. (2D 시계열 데이터)

## 주요 기능

1. 신호 리샘플링:  
   입력된 ECG 신호는 원하는 길이에 맞게 선형 보간법을 통해 리샘플링됩니다.
   
2. Min-Max 스케일링:  
   리샘플링된 신호는 [0, 1] 범위로 스케일링되어 모델에 입력됩니다.

3. 모델 체크포인트 로드:  
   학습된 모델을 로드하여 예측을 수행할 수 있습니다.

4. 예측 및 저장:  
   모델을 사용하여 단일 리드 ECG 데이터를 12-lead ECG로 변환한 후, 결과를 `.npy` 파일로 저장합니다.

## 파일 구조

- `input_folder/`: 변환할 단일 리드 ECG 데이터가 저장된 폴더 (파일 확장자 `.npy`).
- `output_folder/`: 변환된 12-lead ECG 데이터를 저장할 폴더.
- `checkpoint_1ead_12lead/checkpoint_epoch_10.pth.tar`: 학습된 모델의 체크포인트 파일.

## 설치 방법

### 1. 필수 라이브러리 설치

```bash
pip install torch torchvision pandas scikit-learn neurokit2 tqdm wandb matplotlib
```

### 2. 코드 실행 방법

1. 프로젝트를 로컬에 클론합니다:

```bash
git clone https://github.com/yourusername/bi_lstm_attention_ecg.git
cd bi_lstm_attention_ecg
```

2. 필요한 데이터를 준비합니다:
   - `input_folder`에 `.npy` 형식의 단일 리드 ECG 데이터 파일을 넣습니다.
   - `output_folder`는 변환된 데이터를 저장할 폴더입니다.

3. 모델 학습이 완료된 후, 학습된 체크포인트 파일을 사용하여 모델을 로드합니다.

4. 모델을 사용하여 데이터를 변환하고 결과를 저장합니다:

```python
input_folder = "/path/to/your/ecg/data"
output_folder = "/path/to/save/12lead/data"
checkpoint_path = "/path/to/your/model/checkpoint.pth"

process_and_save_npy_files(input_folder, output_folder, model, device)
```

## 코드 설명

- resample_signal(signal, target_length) 
  입력된 신호를 `target_length` 길이로 리샘플링합니다. (선형 보간법 사용)

- min_max_scale(signal, feature_range=(0, 1)) 
  신호를 [0, 1] 범위로 스케일링합니다.

- process_and_save_npy_files(input_folder, output_folder, model, device)
  입력 폴더에 있는 `.npy` 파일을 처리하고 모델을 사용하여 12-lead ECG를 예측한 후, 결과를 출력 폴더에 저장합니다.

- Attention  
  Attention 메커니즘을 구현한 클래스입니다. LSTM 출력을 가중합하여 중요한 특징을 강조합니다.

- BiLSTMAttentionModel 
  BiLSTM과 Attention을 결합한 모델입니다. 이 모델은 시계열 데이터를 처리하고 12-lead ECG를 예측합니다.

## 모델 학습

이 모델은 이전에 학습된 상태로 제공되며, 별도로 학습을 진행할 필요는 없습니다. 학습을 원하시면, `BiLSTMAttentionModel`을 사용하여 모델을 훈련시키고 체크포인트를 저장할 수 있습니다.

## 참고 사항

- 성능 개선: 이 모델은 특정 조건에서 잘 작동하지만, 데이터의 품질이나 양에 따라 성능 차이가 있을 수 있습니다.
- 사용 가능한 데이터: 모델의 입력으로 사용되는 데이터는 `npy` 형식의 단일 리드 ECG 신호여야 합니다.


---

