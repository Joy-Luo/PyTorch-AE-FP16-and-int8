# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:06:05 2022

@author: yling
"""


import os
import glob
import numpy as np
import torch
from torch import nn
# from AE import AutoEncoder

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    return np.fromfile(path, dtype='<f4')


def mse(a, b):
    return np.sqrt(np.sum((a-b)**2))


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


def compress(bytes_rate):
    if not isinstance(bytes_rate, int):
        bytes_rate = int(bytes_rate)
    # query_fea_dir = 'query_feature'
    query_fea_dir = r'C:\\Users\\yling\\Desktop\\query_feature'
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))
    assert(len(query_fea_paths) != 0)
    X = []
    fea_paths = []
    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path) # 00056451
        fea = read_feature_file(query_fea_path) # (2048, )  float32
        assert fea.ndim == 1 and fea.dtype == np.float32
        X.append(fea)
        compressed_fea_path = os.path.join(compressed_query_fea_dir, query_basename + '.dat')
        fea_paths.append(compressed_fea_path)
    input_feature_size = X[0].size
    print('Feature size is {}'.format(input_feature_size))
    print('Sample feature: {}'.format(X[0]))
    
    print("Start doing AE...")
    with open('AutoEncoder_' + str(bytes_rate) + '_int8' + '.pkl', 'rb') as f:
        Coder = AutoEncoder()
        Coder.qconfig = torch.quantization.get_default_qconfig('fbgemm')     
        Coder_prepared = torch.quantization.prepare(Coder)
        Coder_int8 = torch.quantization.convert(Coder_prepared)
        
        Coder_int8.load_state_dict(torch.load(f,map_location='cpu'),False)
        Coder_int8.eval()
        
        X = np.vstack(X)
        tensor_X = torch.Tensor(np.expand_dims(X,axis = 1)).float()
        print('tensor_dtype:',tensor_X.dtype)
        
        encoded, decoded = Coder_int8(tensor_X)  
        print('decoded_dtype:',decoded.dtype)
        print('encoded_dtype:',encoded.dtype)
        print(encoded)
        
        # Given a quantized Tensor,.int_repr() returns a CPU Tensor with uint8_t as data type that stores the underlying uint8_t values of the given Tensor.
        compressed_X = np.squeeze(encoded.int_repr().detach().numpy(),1) 
        print('compress_x_dtype:',compressed_X.dtype)

        c = np.squeeze(decoded.cpu().detach().numpy(),1)
        print('c_type:',c.dtype)
        loss = mse(X, c)
        # np.savetxt("./reconstructed_data.txt", c, delimiter=',')
        print("The reconstructed loss is {}".format(loss))
        print("Start writing compressed feature")
        for path, compressed_fea in zip(fea_paths, compressed_X):
            with open(path, 'wb') as f:
                f.write(int(input_feature_size).to_bytes(4, byteorder='little', signed=False))
                f.write(compressed_fea.astype('<f2').tostring())
        print('Compression Done for bytes_rate' + str(bytes_rate))


if __name__ == '__main__':
    compress('64')
    # compress( '128')
    # compress( '256')