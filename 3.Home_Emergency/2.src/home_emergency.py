import os, timeit
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import Models
import models
    

# ============== Dataset ==============
class AudioDataset(Dataset):
    def __init__(self, dir, transforms, seg=1, train=True):
        self.dir, self.transforms, self.seg = dir, transforms, seg
        self.categories = ['Absence', 'Alarm', 'Cat', 'Dog', 'Kitchen', 'Scream', 'Shatter', 'ShaverToothbrush', 'Slam', 'Speech', 'Water']
        self.num_class = len(self.categories)

        self.transforms = transforms
        class_list = os.listdir(self.dir)
        class_list = [x for x in class_list if 'train' in x] if train else [x for x in class_list if 'test' in x]
        class_list.sort()
        self.class_dict = dict(zip(class_list, torch.arange(len(class_list))))
        
        self.file_list = []
        for class_ in class_list:
            class_path = os.path.join(self.dir, class_, class_.split('_')[0])

            for path in os.listdir(class_path):
                path = os.path.join(class_, class_.split('_')[0], path)
                self.file_list.append(path)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        data_path = os.path.join(self.dir, self.file_list[index])
        audio, sample_rate = torchaudio.load(data_path)
        audio = audio.mean(dim=0)                                       
        if(self.transforms != None):
            audio = self.transforms(audio)   

        audio = audio.view(self.seg, audio.shape[0], int(audio.shape[1]/self.seg)) + 0.001
        audio = audio.log2()

        label = self.class_dict[self.file_list[index].split('/')[0]]

        return audio, label

def print_report(labels_total, y_hat_total, total_loss, target_names):
    conf_mat = confusion_matrix(labels_total, y_hat_total)
    class_rep = classification_report(labels_total, y_hat_total, target_names=target_names)
    total_acc = accuracy_score(labels_total, y_hat_total)
    print(conf_mat)
    print(class_rep)
    print(f'Accuracy : {total_acc}')
    print(f'Test loss: {total_loss:6f}')

    return total_acc

def save_model(model_save_path, model_name, model, epoch):
    if not os.path.exists(model_save_path + model_name):
        os.makedirs(model_save_path + model_name)

    path = model_save_path + model_name + f'/{model_name}_epoch_{epoch+1:03d}.pt'
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model, path)

if __name__ == '__main__':
    # dsp
    Fs, n_fft, n_mels = 16000, 1024, 64

    root_dir = '/home/ksy/3.Home-Emergency/1.dataset/' 
    model_save_path = '/home/ksy/3.Home-Emergency/3.model/'  
     
    # ============== Config Init ==============
    config_defaults = {
        'model_name' : 'ResNet_attention',
        'drop_rate' : 0.1,
        'epochs': 50,
        'h_test' : 5,
        'batch_size': 16,
        'learning_rate': 0.01,
        'h_stepsize' : 2,
        'h_decay' : 0.95,
        'optimizer': 'sgd'
    }

    wandb.init(config=config_defaults, project='home_emergency', entity='ksy')
    wandb.run.name = config_defaults['model_name']

    config = wandb.config

    # ============== Data Load ==============
    transforms = T.MelSpectrogram(sample_rate=Fs, n_fft=n_fft, n_mels=n_mels)

    start_time = timeit.default_timer()
    trainset = AudioDataset(root_dir, transforms, train=True)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, num_workers=8, shuffle=True)
    print(f'trainset_time: {timeit.default_timer()-start_time}s')
    
    start_time = timeit.default_timer()
    testset = AudioDataset(root_dir, transforms, train=False)
    test_loader = DataLoader(testset, batch_size=config.batch_size, num_workers=8, shuffle=True)
    print(f'trainset_time: {timeit.default_timer()-start_time}s')

    for data, labels in train_loader:
        print(f'input size: {data.shape}, labels: {labels.shape}')    # inputs: torch.Size([16, 1, 64, 159]), labels: torch.Size([16])
        break

    # ============== Model Init ==============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        # 'Simple_CNN' : models.CNN(config.num_classes).to(device)
        # 'CNN_v2' : models.CNN_v2(config.num_classes).to(device),
        # 'ResNet18' : models.resnet18().to(device),
        # 'ResNet50' : models.resnet50().to(device),
        # 'DenseNet201' : models.densenet201(drop_rate=float(config.drop_rate), num_classes=int(config.num_classes)).to(device),
        'ResNet_attention' : models.ResidualNet(depth=101, num_classes=trainset.num_class, att_type='CBAM').to(device)      # att_type : CBAM or BAM
    }
    model = model_config[config.model_name]
    model = nn.DataParallel(model)
    loss_fn = nn.CrossEntropyLoss()

   # Define the optimizer
    if config.optimizer=='sgd':
      optim = torch.optim.SGD(model.parameters(),lr=config.learning_rate, momentum=0.9)
    elif config.optimizer=='adam':
      optim = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optim, config.h_stepsize, config.h_decay)

    wandb.watch(model)

    # ============== Train & Test ==============
    best_labels, best_y_hat, best_acc = [], [], 0.
    for epoch in range(config.epochs):
        # ========> Train
        model.train()

        total_loss, running_loss = 0.0,  0.0
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f'{epoch + 1:2d}')):
            optim.zero_grad()
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        
        # logging
        wandb.log({"Train Loss" : total_loss, "global_step" : epoch+1})
        print(f'[{(epoch+1):2d}/{config.epochs:4d}] loss: {total_loss:.6f}')
        
        scheduler.step()

        # ========> Test
        total_loss = 0.0
        if(epoch % config.h_test == config.h_test-1):
            model.eval()
            save_model(model_save_path, config.model_name, model, epoch)
            
            y_hat_total, labels_total, total_loss = [], [], 0.0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels_total += labels.tolist()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()

                    y_hat = torch.argmax(outputs, dim=1)
                    y_hat_total += y_hat.tolist()
                
                acc = print_report(labels_total, y_hat_total, total_loss, testset.categories)
       
                # Best Accuracy Check
                if acc > best_acc:
                    best_acc = acc
                    best_labels = labels_total
                    best_y_hat = y_hat_total
        
                wandb.log({"Test Acc": 100.*acc, 
                    "Test Loss": total_loss,
                    "learning_rate": optim.param_groups[0]['lr'], 
                    "global_step" : epoch+1})
            
    wandb.log({"conf_mat" : wandb.sklearn.plot_confusion_matrix(best_labels, best_y_hat, testset.categories)})         
                
