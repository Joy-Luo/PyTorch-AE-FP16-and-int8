# import glob
import torch
# import zipfile
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import numpy as np
# import tqdm
from torch import nn
import time
from torch.cuda.amp import autocast as autocast


class MyDataset(Dataset):

    def __init__(self, data_path: str, split: str, len_train:float, **kwargs):
        self.data_dir = Path(data_path)
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.dat']) 
        self.imgs = imgs[:int(len(imgs) * len_train)] if split == "train" else imgs[int(len(imgs) * len_train):] 

    def __len__(self):
        return len(self.imgs) 

    def __getitem__(self, idx):
        img = np.fromfile(self.imgs[idx], dtype=np.float32) # (2048, )
        img = np.expand_dims(img, axis=0) # (1,2048)
        img = torch.from_numpy(img) # tensor,torch.Size([1,2048])
        return img, 0.0  # dummy datat to prevent breaking

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
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
            # nn.Linear(64, 128),
            # nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


if __name__ == '__main__':
    starttime = time.time()
    torch.manual_seed(1) 
    EPOCH = 10
    BATCH_SIZE = 64
    LR = 0.0001
    N_TEST_IMG = 5
    NUM_WORKS = 1
    DATA_PATH = r'C:\\Users\\yling\\Desktop\\dataset\\train_feature\\'
    # DATA_PATH = 'C:\\Users\\yling\\Desktop\\test_B\\gallery_feature_B\\'
    # DATA_PATH = 'C:\\Users\\cheng\\Desktop\\datasets\\NAIC2021Reid\\gallery_feature_A\\'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Coder = AutoEncoder().half().to(device)
    Coder = AutoEncoder().to(device)
    print(Coder)

    optimizer = torch.optim.Adam(Coder.parameters(),lr=LR) 
    # optimizer = torch.optim.SGD(Coder.parameters(),lr=LR) 
    scaler = torch.cuda.amp.GradScaler()
    
    loss_func = nn.MSELoss() 
    train_data = MyDataset(DATA_PATH, split='train',len_train=0.85, ) # torch.Size([1,2048])
    loader = DataLoader(dataset=train_data,  
                        batch_size=BATCH_SIZE, 
                        num_workers=NUM_WORKS ,  
                        shuffle=True)
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(loader):
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): # 进入auto上下文后，会自动将tensor类型改成fp16
           
                x,y = x.to(device),y.float().to(device)
                # x = x.half()
                # print('step:',step,'|'+'(x,y)',x.shape) # torch.size([64,1,2048])
                # print('x:',x)
                # print('x_dtype:',x.dtype)
                
                encoded , decoded = Coder(x)
                # encoded , decoded = encoded.float(),decoded.float()
                # print('step:',step,'|' + 'encoded:',encoded.size()) # torch.Size([64, 1, 16])
                # print('encoded_dtype:',encoded.dtype) 
                # print('step:',step,'|' + 'decoded:',decoded.size()) # torch.Size([64, 1, 2048])
                # print('decoded_dtype:',decoded.dtype)
                # print('decoded:',decoded)
                loss = loss_func(decoded,x)
                # print('loss_dtype:',loss.dtype)
            # loss = loss.float()
            scaler.scale(loss).backward()  # 把梯度放大
            scaler.step(optimizer)  # 把梯度unscale回来，梯度不为nan就更新权重
            scaler.update()
            # optimizer.zero_grad()
            # loss.backward()  
            # optimizer.step()  
            # print('epoch{}, loss{:3f}'.format(epoch,loss.item))
            if step%50 == 0:  
                print('Epoch :', epoch,'|','train_loss:%.4f'%loss.data)

        # torch.save(Coder, 'AutoEncoder'+'_epoch'+str(epoch)+'.pkl') 
    torch.save(Coder.cpu().state_dict(),'AutoEncoder_256_fp16.pkl')
    print('________________________________________')
    print('finish training')

    endtime = time.time()
    print('训练耗时：',(endtime - starttime))


    #Coder = AutoEncoder()
    #Coder = torch.load('AutoEncoder.pkl')