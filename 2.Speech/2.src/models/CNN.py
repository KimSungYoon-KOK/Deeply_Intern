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

class CNN_v2(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch3 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 17, 120)
        self.fc2 = nn.Linear(120, self.n_class)

    def forward(self, x):
        # print("input : ", x.shape)                      # input : torch.Size([batch_size, 1, 64, 157])
        layer1 = self.conv1(x)
        layer1 = self.batch1(layer1)
        layer1 = self.pool(self.relu(layer1))
        # print("layer1 : ", layer1.shape)                # layer1 : torch.Size([16, 16, 31, 77])

        layer2 = self.conv2(layer1)
        layer2 = self.batch2(layer2)
        layer2 = self.pool(self.relu(layer2))
        # print("layer2 : ", layer2.shape)                # layer2 : torch.Size([16, 32, 14, 37])

        layer3 = self.conv3(layer2)
        layer3 = self.batch3(layer3)
        layer3 = self.pool(self.relu(layer3))
        # print("layer3 : ", layer3.shape)                # layer3 : torch.Size([16, 64, 6, 17])

        flatten = layer3.view(layer3.size(0), -1)
        # print("flatten : ", flatten.shape)              # flatten : torch.Size([16, 6528])

        fc = self.fc1(flatten)
        output = self.fc2(fc)

        return output
    
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_v2(n_class=2).to(device)
    test_input = torch.rand(16, 1, 64, 157).to(device)
    test_output = model(test_input)
    print(f"Output : {test_output.shape}")