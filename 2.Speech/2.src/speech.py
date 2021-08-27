# UserWarning Ignore
import warnings
warnings.simplefilter("ignore", UserWarning)

# Import Util 
import os, math, random
from glob import glob
from tqdm import tqdm

# Import Pytorch & Metric
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

class AudioDataset(Dataset):
    def __init__(self, root, transforms, seg=1, train=False):
        print("학습 데이터셋 : 성인 목소리  9000개, 기타 생활소음 20000개 (총 29000개)")
        print("시험 데이터셋 : ", end='')
        
        self.root, self.transforms, self.seg = root + '1.dataset/', transforms, seg
        self.file_list = []
        self.categories = ['speech', 'home_sound']
        self.categories_kor = ['성인 목소리', '기타 생활소음']

        flag = 'train/' if train else 'test/'
        for idx, class_path in enumerate(os.listdir(self.root + flag)):
            temp_list = glob(self.root + flag + class_path + '/*')
            self.file_list.extend(temp_list)
            
            print(f"{self.categories_kor[idx]} {len(temp_list):>5}개", end='')
            if idx < 1:
                print(", ", end='')
            else:
                print(f" (총 {len(self.file_list):>5}개)") 
            
        if train:
            self.file_list = self.file_list[:-5]    
        else:
            self.file_list = self.file_list[:-8]  
                
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        audio, sample_rate = torchaudio.load(self.file_list[index])
        audio = audio.mean(dim=0)                                       
        if(self.transforms != None):
            audio = self.transforms(audio).squeeze().transpose(0,1)     

        audio = audio.view(self.seg, audio.shape[1], int(audio.shape[0]/self.seg)) + 0.001
        audio = audio.log2()

        label = self.file_list[index].split('/')[-2]            # home_sound / speech
        label = torch.tensor(0) if label == 'speech' else torch.tensor(1)

        return audio, label
            
def print_report(labels_total, y_hat_total, target_names):
    class_rep = classification_report(labels_total, y_hat_total, target_names=target_names)
    conf_mat = confusion_matrix(labels_total, y_hat_total)
    acc_each_class = conf_mat.diagonal()/conf_mat.sum(axis=1)
    f1Score = f1_score(labels_total, y_hat_total, average=None)
    print("\n[ 클래스 별 Accuracy, F1-Score ]")
    print("\t       Accuracy F1-Score")
    
    print(f"{target_names[0]:>8} : {acc_each_class[0]:0.4f} \t {f1Score[0]:0.4f}")
    print(f"{target_names[1]:>7} : {acc_each_class[1]:0.4f} \t {f1Score[1]:0.4f}")
    
    overall_acc = accuracy_score(labels_total, y_hat_total)
    overall_f1 = f1_score(labels_total, y_hat_total, average='macro')

    overall_str = "전체 클래스"
    print(f'\n{overall_str:>8} : {overall_acc:0.4f} \t {overall_f1:0.4f}\n')


if __name__ == '__main__':
    for iter_idx in range(3):
        print(f"\n[ 시행 횟수: {iter_idx + 1} / 3 ]")  
        root_dir = '/Users/sungyoon-kim/Desktop/2.Speech/'

        # dsp
        Fs, n_fft, n_mels = 16000, 1000, 64
        
        # learning
        s_batch = 16

        # test
        threshold = 5.0

        transforms = T.MelSpectrogram(sample_rate=Fs, n_fft=n_fft, n_mels=n_mels)

        testset = AudioDataset(root_dir, transforms, train=False)
        test_loader = DataLoader(testset, batch_size=s_batch, num_workers=4)
        
        print("모델 로딩 중...\r", end='')
        model_path = root_dir + '3.model/model_epoch_030.pt'
        devcice = "cpu"
        model = torch.load(model_path, map_location=devcice)
        print("모델 로드 완료")

        model.eval()
        y_hat_total, labels_total = [], []
        with torch.no_grad():
            for data in tqdm(test_loader, desc="테스트 중"):
                inputs, labels = data[0].to(devcice), data[1].to(devcice)
                labels_total += labels.tolist()
                outputs = model(inputs)

                y_hat = []
                for speech, home in outputs.tolist():
                    if speech > threshold and speech > home:
                        y_hat.append(0)
                    else:
                        y_hat.append(1)

                y_hat_total += y_hat

        print_report(labels_total, y_hat_total, testset.categories_kor)
        
        if iter_idx < 2:
            input("테스트를 계속 하시려면 엔터키를 입력하세요.")
        else:
            print("테스트 완료")
        
    