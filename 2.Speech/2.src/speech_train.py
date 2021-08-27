# Import Util
import os, timeit
from glob import glob
from tqdm import tqdm

# Import Pytorch & Metric
import torch, torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Import Model
import models     

# ============== Dataset Load ==============
class AudioDataset(Dataset):
    def __init__(self, root, transforms, seg=1, train=True):
        self.root, self.transforms, self.seg = root, transforms, seg
        
        self.file_list = []
        self.categories = ['speech', 'home_sound']
        self.categories_kor = ['성인 목소리', '기타 생활소음']
        self.num_class = len(self.categories)

        flag = 'train/' if train else 'test/'
        for class_path in os.listdir(self.root + flag):
            self.file_list.extend(glob(self.root + flag + class_path + '/*'))

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

def print_report(labels_total, y_hat_total, target_names):
    class_rep = classification_report(labels_total, y_hat_total, target_names=target_names)
    conf_mat = confusion_matrix(labels_total, y_hat_total)
    print(conf_mat)
    print(class_rep)
    # acc_each_class = conf_mat.diagonal()/conf_mat.sum(axis=1)
    # f1Score = f1_score(labels_total, y_hat_total, average=None)
    # print("\n[ 클래스 별 Accuracy, F1-Score ]")
    # print("\t       Accuracy F1-Score")
    
    # print(f"{target_names[0]:>8} : {acc_each_class[0]:0.4f} \t {f1Score[0]:0.4f}")
    # print(f"{target_names[1]:>7} : {acc_each_class[1]:0.4f} \t {f1Score[1]:0.4f}")
    
    # overall_acc = accuracy_score(labels_total, y_hat_total)
    # overall_f1 = f1_score(labels_total, y_hat_total, average='macro')

    # overall_str = "전체 클래스"
    # print(f'\n{overall_str:>8} : {overall_acc:0.4f} \t {overall_f1:0.4f}\n')

def save_model(model_save_path, model, epoch):
    model_name = "CNN"
    if not os.path.exists(model_save_path + model_name):
        os.makedirs(model_save_path + model_name)

    path = model_save_path + model_name + f'/{model_name}_epoch_{epoch+1:03d}.pt'
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model, path)
    

if __name__ == "__main__":
    # DSP
    Fs, n_fft, n_mels = 16000, 1024, 64

    # learning
    s_batch = 16
    h_lr = 0.01
    h_stepsize = 2
    h_decay = 0.95
    h_epoch = 5
    h_test = 5

    root_dir = '/home/ksy/2.Speech/1.dataset/' 
    model_save_path = '/home/ksy/2.Speech/3.model/'  

    transforms = torchaudio.transforms.MelSpectrogram(sample_rate=Fs, n_fft=n_fft, n_mels=n_mels)

    # Dataset Load
    start_time = timeit.default_timer()
    trainset = AudioDataset(root_dir, transforms, train=True)
    train_loader = DataLoader(trainset, batch_size=s_batch, num_workers=8, shuffle=True)
    print(f'trainset_time: {timeit.default_timer()-start_time}s')

    start_time = timeit.default_timer()
    testset = AudioDataset(root_dir, transforms, train=False)
    test_loader = DataLoader(testset, batch_size=s_batch, num_workers=8)
    print(f'testset_time: {timeit.default_timer()-start_time}s')

    for data, labels in train_loader:
        print(f'input size: {data.shape}, labels: {labels.shape}')    # inputs: torch.Size([batch_size, 1, 64, 157]), labels: torch.Size([16])
        break

    # Model Load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.CNN_v2(n_class=trainset.num_class).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Loss & Optimizer Init
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=h_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, h_stepsize, h_decay)


    # ============== Train & Test ==============
    for epoch in range(h_epoch):
        # Train
        model.train()
        total_loss = 0.0
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f'{epoch + 1:2d}')):
            optim.zero_grad()
            inputs, labels = data[0].to(device), data[1].to(device) 
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print("[{:3d}/{:3d}] loss: {:.6f}".format(epoch+1, h_epoch, total_loss))
        
        scheduler.step()

        # Test
        if(epoch % h_test == h_test-1):
            model.eval()
            save_model(model_save_path, model, epoch)
            
            y_hat_total, labels_total = [], []
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    y_hat = torch.argmax(outputs, dim=1)

                    loss = loss_fn(outputs, labels)
                    total_loss += loss.item()

                    # y_hat = []
                    # for speech, home in outputs.tolist():
                    #     if speech > 1.0 and speech > home:
                    #         y_hat.append(0)
                    #     else:
                    #         y_hat.append(1)

                    labels_total += labels.tolist()
                    y_hat_total += y_hat.tolist()
                
                print_report(labels_total, y_hat_total, testset.categories)


