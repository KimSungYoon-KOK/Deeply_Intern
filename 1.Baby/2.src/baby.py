import os, timeit, random, math
from glob import glob
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import Model
import models


# ============== Dataset Load ==============
class AudioDataset(Dataset):
    def __init__(self, root, transforms, seg=1, train=True, verbose=False):
        self.root, self.transforms, self.seg = root, transforms, seg

        self.file_list = []
        self.categories = ['home_sound', 'laughter', 'babbling', 'cry']

        flag = 'train/' if train else 'test/'
        for class_path in os.listdir(self.root + flag):
            if 'web' in class_path: continue
            temp_list = glob(self.root + flag + class_path + '/*')

            if verbose:
                print(f'{class_path} : {len(temp_list)}')

            self.file_list.extend(temp_list)

        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        audio, sample_rate = torchaudio.load(self.file_list[index])
        audio = audio.mean(dim=0)                                       
        if(self.transforms != None):
            audio = self.transforms(audio)

        audio = audio.view(self.seg, audio.shape[0], int(audio.shape[1]/self.seg)) + 0.001
        audio = audio.log2()

        label = self.file_list[index].split('/')[-2]
        label = torch.tensor(self.categories.index(label))

        return audio, label

def print_report(classes, labels_total, y_hat_total, total_loss):
    conf_mat = confusion_matrix(labels_total, y_hat_total)
    class_rep = classification_report(labels_total, y_hat_total, target_names=classes)
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

    root_dir = '/home/ksy/1.Baby/1.dataset/'
    model_save_path = '/home/ksy/1.Baby/3.model/'

    # ============== Config Init ==============
    config_defaults = {
        'model_name' : 'ResNet_attention',
        'drop_rate' : 0.1,
        'epochs': 50,
        'h_test' : 5,
        'batch_size': 50,
        'learning_rate': 0.01,
        'h_stepsize' : 2,
        'h_decay' : 0.95,
        'optimizer': 'sgd',
        'num_classes':4
    }

    # ============== Data Load ==============
    transforms = T.MelSpectrogram(sample_rate=Fs, n_fft=n_fft, n_mels=n_mels)

    start_time = timeit.default_timer()
    trainset = AudioDataset(root_dir, transforms, train=True, verbose=False)
    print(f'trainset_time: {timeit.default_timer()-start_time}s')

    start_time = timeit.default_timer()
    testset = AudioDataset(root_dir, transforms, train=False, verbose=False)
    print(f'testset_time: {timeit.default_timer()-start_time}s')

    train_loader = DataLoader(trainset, batch_size=config_defaults['batch_size'], num_workers=8, shuffle=True)
    test_loader = DataLoader(testset, batch_size=config_defaults['batch_size'], num_workers=8, shuffle=True)

    # ============== W&B Init ==============
    wandb.init(config=config_defaults, project='baby_cry', entity='ksy')        # resume='1vhbkwh7'
    wandb.run.name = config_defaults['model_name']

    config = wandb.config


    # ============== Model Init ==============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
            # 'Simple_CNN' : models.Simple_CNN(config.num_classes).to(device),
            # 'CNN_v2' : models.CNN_v2(config.num_classes).to(device),
            # 'DenseNet201' : models.densenet201(drop_rate=config.drop_rate, num_classes=config.num_classes).to(device),
            # 'ASTModel': models.ASTModel(label_dim=config.num_classes, input_fdim=n_mels, input_tdim=33, imagenet_pretrain=True, audioset_pretrain=True).to(device)
            'ResNet_attention' : models.ResidualNet(depth=101, num_classes=config.num_classes, att_type='CBAM').to(device)      # att_type : CBAM or BAM
        }

    if wandb.run.resumed:
        # restore model
        model_path = model_save_path + config.model_name + '/'
        model = torch.load(model_path + os.listdir(model_path)[-1]).to(device)
        model = nn.DataParallel(model)
    else:
        # New model
        model = model_config[config.model_name]   
        model = nn.DataParallel(model)

        # Define the optimizer
        if config.optimizer=='sgd':
            optim = torch.optim.SGD(model.parameters(), config.learning_rate, momentum=0.9)
        elif config.optimizer=='adam':
            optim = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=5e-7, betas=(0.95, 0.999))
    

    if 'weight' in config.model_name:
        weigths = torch.FloatTensor([0.254, 0.914, 0.865, 0.967]).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weigths)
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.StepLR(optim, config.h_stepsize, config.h_decay)

    wandb.watch(model)
    best_labels, best_y_hat, best_acc = [], [], 0.
    for epoch in range(config.epochs):
        """
        Train
        """
        start_time = timeit.default_timer()
        model.train()

        total_loss, running_loss = 0.0, 0.0
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f'{epoch + 1:2d}')):
            optim.zero_grad()
            inputs, labels = data[0].to(device), data[1].to(device)
            
            outputs = model(inputs)
            
            if epoch == 0 and batch_idx == 0:
                print(f'inputs: {inputs.shape}, lnabels: {labels.shape}')           
                print(f'outputs: {outputs.shape}, labels: {labels.shape}')

            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        
        wandb.log({"Train Loss" : total_loss, "global_step" : epoch+1})

        print(f"[{(epoch+1):2d}/{config.epochs:4d}] loss: {total_loss:.6f}")
        print(f"epoch_elapsed_time: {timeit.default_timer()-start_time:3f}s")
        
        scheduler.step()

        """
        Test
        """
        if(epoch % config.h_test == config.h_test-1):
            model.eval()
            save_model(model_save_path, config.model_name, model, epoch)
            
            y_hat_total, labels_total, test_loass = [], [], 0.0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels_total += labels.tolist()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()

                    y_hat = torch.argmax(outputs, dim=1)
                    y_hat_total += y_hat.tolist()
                
                acc = print_report(testset.categories, labels_total, y_hat_total, total_loss)
            
            # Best Accuracy Check
            if acc > best_acc:
                best_acc = acc
                best_labels = labels_total
                best_y_hat = y_hat_total
            
            wandb.log({
                "Test Acc": 100.*acc, 
                "Test Loss": total_loss,
                "learning_rate": optim.param_groups[0]['lr'], 
                "global_step" : epoch+1})
    
    wandb.log({"conf_mat" : wandb.sklearn.plot_confusion_matrix(best_labels, best_y_hat, testset.categories)})

