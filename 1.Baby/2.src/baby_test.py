# UserWarning Ignore
import warnings
warnings.simplefilter("ignore", UserWarning)

# Import Util
import os
from glob import glob
from tqdm import tqdm

# Import Pytorch & Metric
import torch, torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Import Models
import models

# ============== Dataset ==============
class AudioDataset(Dataset):
    def __init__(self, root, transforms, seg=1, train=True, verbose=True):
        if verbose:
            print("학습 데이터셋 : 기타 소리 10000개, 아기 웃음   300개, 아기 소리  3000개, 아기 울음  3000개 (총 16300개)")
            print("시험 데이터셋 : ", end='')
        self.root, self.transforms, self.seg = root, transforms, seg

        self.file_list = []
        self.categories = ['home_sound', 'laughter', 'babbling', 'cry']
        self.categories_kor = ['기타 소리', '아기 웃음', '아기 소리', '아기 울음']
        self.num_class = len(self.categories)

        flag = 'train/' if train else 'test/'
        for idx, class_path in enumerate(os.listdir(self.root + flag)):
            if 'web' in class_path: continue
            temp_list = glob(self.root + flag + class_path + '/*')
            self.file_list.extend(temp_list)

            if verbose:
                print(f"{self.categories_kor[idx]} {len(temp_list):>5}개", end='')
                if idx < 3:
                    print(", ", end='')
                else:
                    print(f" (총 {len(self.file_list):>5}개)")
        
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
    # class_rep = classification_report(labels_total, y_hat_total, target_names=target_names)
    conf_mat = confusion_matrix(labels_total, y_hat_total)
    acc_each_class = conf_mat.diagonal()/conf_mat.sum(axis=1)
    f1Score = f1_score(labels_total, y_hat_total, average=None)
    print("\n[ 클래스 별 Accuracy, F1-Score ]")
    print("\t     Accuracy   F1-Score")
    for i, acc in enumerate(acc_each_class):
        print(f"{target_names[i]:>7} : {acc:0.4f} \t {f1Score[i]:0.4f}")
    
    overall_acc = accuracy_score(labels_total, y_hat_total)
    overall_f1 = f1_score(labels_total, y_hat_total, average='macro')

    overall_str = "전체 클래스"
    print(f'\n{overall_str:>6} : {overall_acc:0.4f} \t {overall_f1:0.4f}\n')
    


if __name__ == '__main__':
    
    for iter_idx in range(3):
        print(f"\n[ 시행 횟수: {iter_idx + 1} / 3 ]")
        root_dir = '/home/ksy/1.Baby/1.dataset/'

        # DSP
        Fs, n_fft, n_mels = 16000, 1024, 64
        transforms = torchaudio.transforms.MelSpectrogram(sample_rate=Fs, n_fft=n_fft, n_mels=n_mels)

        # Dataset Load
        testset = AudioDataset(root_dir, transforms, train=False, verbose=True)
        test_loader = DataLoader(testset, batch_size=50, num_workers=8)
            
        # Model Init
        print("모델 로딩 중...\r", end='')
        model_name = 'ResNet_attention'
        model_path = '/home/ksy/1.Baby/3.model/ResNet_attention/ResNet_attention_epoch_015.pt'

        model = models.ResidualNet(depth=101, num_classes=4, att_type='CBAM')      # att_type : CBAM or BAM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        print("모델 로드 완료")


        # Model Test
        model.eval()    
        y_hat_total, labels_total = [], []
        with torch.no_grad():
            for data in tqdm(test_loader, desc="테스트 중"):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                y_hat = torch.argmax(outputs, dim=1)

                labels_total += labels.tolist()
                y_hat_total += y_hat.tolist()
            
        print_report(labels_total, y_hat_total, target_names=testset.categories_kor)

        if iter_idx < 2:
            input("테스트를 계속 하시려면 엔터키를 입력하세요.")
        else:
            print("테스트 완료")
