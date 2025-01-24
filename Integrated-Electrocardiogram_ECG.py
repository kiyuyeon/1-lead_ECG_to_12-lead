import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from adamp import AdamP
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import neurokit2 as nk
from scipy.interpolate import interp1d
import torchvision.models as models
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import os
import torch
import numpy as np
from scipy.interpolate import interp1d

device = torch.device("cuda" if torch.cuda.is_available()else"cpu")

def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded checkpoint from '{filepath}'")
    return model, optimizer

def resample_signal(signal, target_length):
    if signal.ndim == 1:  # Single-lead (1D)
        current_length = signal.shape[0]
        original_points = np.linspace(0, 1, current_length)
        new_points = np.linspace(0, 1, target_length)
        interpolation_function = interp1d(original_points, signal, kind='linear', fill_value="extrapolate")
        resampled_signal = interpolation_function(new_points)
    elif signal.ndim == 2:  # 12-lead (2D)
        num_leads, current_length = signal.shape
        resampled_signal = np.zeros((num_leads, target_length))
        original_points = np.linspace(0, 1, current_length)
        new_points = np.linspace(0, 1, target_length)
        for i in range(num_leads):
            interpolation_function = interp1d(original_points, signal[i, :], kind='linear', fill_value="extrapolate")
            resampled_signal[i, :] = interpolation_function(new_points)
    else:
        raise ValueError("Signal must be either 1D or 2D.")
    return resampled_signal
    
def min_max_scale(signal, feature_range=(0, 1)):

    min_val = feature_range[0]
    max_val = feature_range[1]
    
    # Scale each lead separately if the signal is 2D.
    if signal.ndim == 2:
        scaled_signal = np.zeros_like(signal, dtype=np.float32)
        for i in range(signal.shape[0]):
            min_val_signal = np.min(signal[i])
            max_val_signal = np.max(signal[i])
            scaled_signal[i] = (signal[i] - min_val_signal) / (max_val_signal - min_val_signal) * (max_val - min_val) + min_val
    else:  # Signal is 1D.
        min_val_signal = np.min(signal)
        max_val_signal = np.max(signal)
        scaled_signal = (signal - min_val_signal) / (max_val_signal - min_val_signal) * (max_val - min_val) + min_val
    
    return scaled_signal

def process_and_save_npy_files(input_folder, output_folder, model, device, target_length=1000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            save_path = os.path.join(output_folder, filename)
            # 파일이 이미 존재하는 경우 스킵
            if os.path.exists(save_path):
                print(f"{filename} is already processed. Skipping.")
                continue

            file_path = os.path.join(input_folder, filename)
            signal = np.load(file_path, allow_pickle=True)

            # signal이 2차원 배열인 경우, 첫 번째 차원을 사용 (예시)
            if signal.ndim > 1:
                print(f"{filename} has shape {signal.shape}, using the first dimension.")
                signal = signal[0, :]

            # 재샘플링
            resampled_signal = resample_signal(signal, target_length)
            scaled_signal = min_max_scale(resampled_signal)
            resampled_signal_tensor = torch.tensor(scaled_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

            # 모델을 사용하여 변환
            predicted_12_lead_ecg = model(resampled_signal_tensor)
            predicted_12_lead_ecg = predicted_12_lead_ecg.squeeze().cpu().detach().numpy()

            # 데이터 전치하여 저장 형식을 [12, 1000]으로 조정
            predicted_12_lead_ecg_transposed = predicted_12_lead_ecg.T

            # 변환된 데이터 저장
            np.save(save_path, predicted_12_lead_ecg_transposed)
            print(f"{filename} 변환 완료 및 저장됨: {save_path}")



class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias
        self.weights = nn.Parameter(torch.Tensor(feature_dim, 1))
        nn.init.kaiming_uniform_(self.weights)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        eij = torch.matmul(x.contiguous().view(-1, self.feature_dim), self.weights)
        if self.bias is not None:
            eij = eij + self.bias
            
        eij = torch.tanh(eij)
        eij = eij.view(-1, self.step_dim)
        a = torch.softmax(eij, dim=1)
        
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_leads=12):
        super(BiLSTMAttentionModel, self).__init__()
        self.input_size = input_size  
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_leads = num_leads
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2, sequence_length)
        self.fc = nn.Linear(hidden_size * 2, sequence_length * num_leads) 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        attention_out = self.attention(lstm_out)
        out = self.fc(attention_out)
        out = out.view(-1, self.sequence_length, self.num_leads)
        
        return out

# Parameters
input_size = 1
hidden_size = 400 
num_layers = 3
output_size = 12
sequence_length = 1000
learning_rate = 1e-3


# 예제 사용
input_folder = "/data/hw/mobile_ecg_rawdata/Jenonnam_Result_ECG2CSV_mobi_care/clean_mobicare"
output_folder = "/data/mkdata/0410_12lead"

# 모델, 옵티마이저, 디바이스 설정 및 체크포인트 불러오기
model = BiLSTMAttentionModel(input_size, hidden_size, num_layers, sequence_length).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
model.to(device)
checkpoint_path = '/data/checkpoint_1ead_12lead/checkpoint_epoch_10.pth.tar'
model, optimizer = load_checkpoint(checkpoint_path, model, optimizer)

# 파일 처리 및 저장
process_and_save_npy_files(input_folder, output_folder, model, device)
