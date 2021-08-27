# UserWarning Ignore
import warnings
warnings.simplefilter("ignore", UserWarning)

# Import Util
import os, unicodedata
from tqdm import tqdm

# Import Pytorch & Metric
import torch, torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Import Models
import models

class AudioDataset(Dataset):
    def __init__(self, dir, transforms, seg=1, train=True, verbose=True):
        if verbose:
            print("학습 데이터셋 : 무음 11285개, 알람소리  2765개, 고양이 11724개, 강아지  6673개, 부엌소리 12291개, 비명소리  4308개,\n\t\t깨지는 소리  2877개, 칫솔/면도 11231개, 쿵소리  1565개, 사람목소리 30113개, 물소리  6796개 (총 101628개)")
            print("시험 데이터셋 : ", end='')

        self.dir, self.transforms, self.seg = dir, transforms, seg
        self.categories = ['Absence', 'Alarm', 'Cat', 'Dog', 'Kitchen', 'Scream', 'Shatter', 'ShaverToothbrush', 'Slam', 'Speech', 'Water']
        self.categories_kor = ['무음', '알람소리', '고양이', '강아지', '부엌소리', '비명소리', '깨지는 소리', '칫솔/면도', '쿵소리', '사람목소리', '물소리']
        self.num_class = len(self.categories)

        self.transforms = transforms
        class_list = os.listdir(self.dir)
        class_list = [x for x in class_list if 'train' in x] if train else [x for x in class_list if 'test' in x]
        class_list.sort()
        self.class_dict = dict(zip(class_list, torch.arange(len(class_list))))
        
        self.file_list = []
        for idx, class_ in enumerate(class_list):
            class_path = os.path.join(self.dir, class_, class_.split('_')[0])

            for path in os.listdir(class_path):
                path = os.path.join(class_, class_.split('_')[0], path)
                self.file_list.append(path)
            
            if verbose:
                if idx == 6:
                    print(f"\n\t\t{self.categories_kor[idx]} {len(os.listdir(class_path)):>5}개", end='')
                else:
                    print(f"{self.categories_kor[idx]} {len(os.listdir(class_path)):>5}개", end='')
                if idx < 10:
                    print(", ", end='')
                else:
                    print(f" (총 {len(self.file_list):>6}개)")
        
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

def fill_str_with_space(input_s="", max_size=20, fill_char=" "):
    """
    - 길이가 긴 문자는 2칸으로 체크하고, 짧으면 1칸으로 체크함. 
    - 최대 길이(max_size)는 40이며, input_s의 실제 길이가 이보다 짧으면 
    남은 문자를 fill_char로 채운다.
    """
    l = 0 
    for c in input_s:
        if unicodedata.east_asian_width(c) in ['F', 'W']:
            l+=2
        else: 
            l+=1
    return fill_char*(max_size-l)+input_s

def print_report(labels_total, y_hat_total, target_names):
    # class_rep = classification_report(labels_total, y_hat_total, target_names=target_names)
    conf_mat = confusion_matrix(labels_total, y_hat_total)
    acc_each_class = conf_mat.diagonal()/conf_mat.sum(axis=1)
    f1Score = f1_score(labels_total, y_hat_total, average=None)
    print("\n[ 클래스 별 Accuracy, F1-Score ]")
    print("\t     Accuracy   F1-Score")
    for i, acc in enumerate(acc_each_class):
        # print(f"{target_names[i]:>10} : {acc:0.4f} \t {f1Score[i]:0.4f}")
        print(f"{fill_str_with_space(target_names[i], max_size=12)} : {acc:0.4f} \t {f1Score[i]:0.4f}")
    
    overall_acc = accuracy_score(labels_total, y_hat_total)
    overall_f1 = f1_score(labels_total, y_hat_total, average='macro')

    overall_str = "전체 클래스"
    print(f'\n{fill_str_with_space(overall_str, max_size=12)} : {overall_acc:0.4f} \t {overall_f1:0.4f}\n')

if __name__ == '__main__':
    for iter_idx in range(3):
        print(f"\n[ 시행 횟수: {iter_idx + 1} / 3 ]")
        root_dir = '/home/ksy/3.Home-Emergency/1.dataset'

        # DSP
        Fs, n_fft, n_mels = 16000, 1024, 64
        transforms = torchaudio.transforms.MelSpectrogram(sample_rate=Fs, n_fft=n_fft, n_mels=n_mels)

        # Learning
        s_batch = 16

        # Dataset Load
        testset = AudioDataset(dir=root_dir, transforms=transforms, train=False)
        test_loader = DataLoader(testset, batch_size=s_batch, num_workers=8)
            
        # Model Load
        print("모델 로딩 중...\r", end='')
        model_path = '/home/ksy/3.Home-Emergency/3.model/ResNet_attention/ResNet_attention_epoch_025.pt'
        model = models.ResidualNet(depth=101, num_classes=testset.num_class, att_type='CBAM')      # att_type : CBAM or BAM
        model.load_state_dict(torch.load(model_path))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        print("모델 로드 완료")
        

        # Model Test
        model.eval()
        pred_total, labels_total = [], []
        with torch.no_grad():
            for data in tqdm(test_loader, desc="테스트 중"):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                pred = torch.argmax(outputs, dim=1)

                labels_total += labels.tolist()
                pred_total += pred.tolist()

        print_report(labels_total, pred_total, target_names=testset.categories_kor)

        if iter_idx < 2:
            input("테스트를 계속 하시려면 엔터키를 입력하세요.")
        else:
            print("테스트 완료")

