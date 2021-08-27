import os, sys, random, math
import torch
from torch.cuda.amp import autocast
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import models


class AudioDataset(IterableDataset):
    def __init__(self, root, transforms, seg=1, train=True):
        self.root, self.transforms, self.seg = root, transforms, seg

        class_list = os.listdir(self.root)
        class_list = [x for x in class_list if 'train' in x] if train else [x for x in class_list if 'test' in x]
        class_list.sort()

        self.class_dict = dict(zip(class_list, torch.arange(len(class_list))))
        # print(self.class_dict)

        self.file_list = []
        for class_ in class_list:
            list_temp = os.listdir(root_dir + class_ + '/' + class_.split('_')[0])
            # print(f'{class_} : {len(list_temp)}')
            for idx in range(len(list_temp)):
                self.file_list.append(class_ + '/' + class_.split('_')[0] + '/' + list_temp[idx])
        random.shuffle(self.file_list)
        self.transforms = transforms
        
        self.start = 0
        self.end = len(self.file_list)-1
        # print(len(self.file_list))        # Train : 101611 / Test : 9364

    def __iter__(self):
        for file_name in self.file_list[self.start:self.end]:
            audio, sample_rate = torchaudio.load(self.root + file_name)                 
            audio = audio.mean(dim=0)                                       
            if(self.transforms != None):
                audio = self.transforms(audio).squeeze().transpose(0,1)     

            audio = audio.view(self.seg, audio.shape[1], int(audio.shape[0]/self.seg)) + 0.001
            audio = audio.log2()

            label = file_name.split('/')[0]
            label = torch.full((1,), self.class_dict[label], dtype=torch.long).squeeze()
            
            yield {"data":audio, "label":label}             # inputs: torch.Size([16, 1, 64, 162]), labels: torch.Size([16])

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))

    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

def print_report(classes, labels_total, y_hat_total):
    print(confusion_matrix(labels_total, y_hat_total))
    print(classification_report(labels_total, y_hat_total, target_names=classes))
    accuracy = accuracy_score(labels_total, y_hat_total)
    print(f'Accuracy : {accuracy}')

    return accuracy

if __name__=='__main__':
    # dsp
    Fs, n_fft, n_mels = 16000, 1000, 64
    bath_size = 16
    root_dir = '/home/ksy/3.Home-Emergency/1.dataset/' 
    classes = ['Absence', 'Alarm', 'Cat', 'Dog', 'Kitchen', 'Scream', 'Shatter', 'ShaverToothbrush', 'Slam', 'Speech', 'Water']
    
    # ============== Data Load ==============
    transforms = T.MelSpectrogram(sample_rate=Fs, n_fft=n_fft, n_mels=n_mels)

    iterable_testset = AudioDataset(root_dir, transforms, train=False)
    test_loader = DataLoader(iterable_testset, batch_size=bath_size, num_workers=20, worker_init_fn=worker_init_fn)

    # audioset input sequence length is 1024
    pretrained_mdl_path = '/home/ksy/3.Home-Emergency/3.model/pretrained_models/audioset_10_10_0.4593.pth'
    # get the frequency and time stride of the pretrained model from its name
    fstride, tstride = int(pretrained_mdl_path.split('/')[-1].split('_')[1]), int(pretrained_mdl_path.split('/')[-1].split('_')[2].split('.')[0])
    # The input of audioset pretrained model is 1024 frames.
    input_tdim = 162

    # initialize an AST model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(pretrained_mdl_path, map_location=device)
    model = models.ASTModel(label_dim=11, input_fdim=n_mels, input_tdim=input_tdim, fstride=fstride, tstride=tstride).to(device)
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(sd, strict=False)

    y_hat_total, labels_total, total_loss = [], [], 0.0
    for i, data in enumerate(test_loader):
        inputs, labels = data["data"].to(device), data["label"].to(device)
        labels_total += labels.tolist()
    
        # print(f"Input : {inputs.shape} / dtype : {inputs.dtype}")        # torch.Size([16, 1, 162, 128])
        outputs = model(inputs)
        # print(f'outputs: {outputs.shape}, labels: {labels.shape}')       # outputs: torch.Size([16, 11]), labels: torch.Size([16])

        y_hat = torch.argmax(outputs, dim=1)
        y_hat_total += y_hat.tolist()

    accuracy = print_report(classes, labels_total, y_hat_total)
