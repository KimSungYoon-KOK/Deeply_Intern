import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, s_batch, n_class):
        super(CNN,self).__init__()
        self.s_batch = s_batch
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3),             
            nn.ReLU(),                                                          
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),                               
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),          
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)                                
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(34048, 100),                                              
            nn.ReLU(),
            nn.Linear(100,n_class)         
        )

    def forward(self,x):
        out = self.layer(x)           
        out = out.view(self.s_batch, -1)                                              
                                                              
        out = self.fc_layer(out)
        return out
