# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:15:19 2022

@author: yling
"""


import os
import glob
import numpy as np
import torch
from torch import nn



def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.shape[0] == 2048 and fea.dtype == np.float32
    fea.astype('<f4').tofile(path)
    return True


def reconstruct_feature(path: str) -> np.ndarray:
    fea = np.fromfile(path, dtype='<f4')
    fea = np.concatenate(
        [fea, np.zeros(2048 - fea.shape[0], dtype='<f4')], axis=0
    )
    return fea


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.quant = torch.quantization.QuantStub() # quansub converts tensor from floating to quantized
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(

            # nn.Linear(16, 32),
            # nn.ReLU(),
            # nn.Linear(32, 64),
            # nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Sigmoid()
        )
        self.dequant = torch.quantization.DeQuantStub() # convert tensor from quantized to floating
        
    def forward(self, x):
        x = self.quant(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = self.dequant(decoded)
        
        return encoded, decoded

def reconstruct(bytes_rate):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(bytes_rate)
    query_fea_dir = 'query_feature'
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)

    compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))
    assert(len(compressed_query_fea_paths) != 0)
    names = []
    X = []
    # feature_len = 0
    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
        query_fea_path = os.path.join(query_fea_dir,query_basename + '.dat')

        
        fea = np.fromfile(query_fea_path, dtype='<f4')
        assert fea.ndim == 1 and fea.dtype == np.float32
        X.append(fea)
        names.append(reconstructed_fea_path)
    
    # Do decompress
    with open('AutoEncoder_' + str(bytes_rate) + '_int8' + '.pkl', 'rb') as f:
        Coder = AutoEncoder()
        Coder.qconfig = torch.quantization.get_default_qconfig('fbgemm')     
        Coder_prepared = torch.quantization.prepare(Coder)
        Coder_int8 = torch.quantization.convert(Coder_prepared)
        
        Coder_int8.load_state_dict(torch.load(f,map_location='cpu'),False)
        Coder_int8.eval()
        
        
        X = np.vstack(X)
        tensor_X = torch.tensor(np.expand_dims(X,axis=1)).float()
        print('tensor_X_dtype:',tensor_X.dtype)
        
        encoded,decoded = Coder_int8(tensor_X)
        
        print('decoded_dtype:',decoded.dtype)
        # Given a quantized Tensor,.int_repr() returns a CPU Tensor with uint8_t as data type that stores the underlying uint8_t values of the given Tensor.
        reconstructed_fea = np.squeeze(decoded.cpu().detach().numpy(),1)
        print('reconstructed_fea_dtype:',reconstructed_fea.dtype)

    # np.savetxt("./reconstructed_data.txt", c, delimiter=',')
    for path, decompressed_feature in zip(names, reconstructed_fea):
        # reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, path + '.dat')
        write_feature_file(decompressed_feature, path)

    print('Reconstruction Done' + bytes_rate)

if __name__ == '__main__':
    reconstruct('64')
    # compress( '128')
    # compress( '256')