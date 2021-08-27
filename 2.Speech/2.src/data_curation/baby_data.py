import os
import csv
import json
from pydub import AudioSegment
from tqdm import tqdm
import librosa
import soundfile as sf
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import shutil
import seaborn as sns

def Baby_cry_dataset():
    """ 
    Cry Data
    Z-score threshold : zscore > 0.8 (104)
    Variance threshold : var > 0.1   (101)
    Remove Data : 4_25.wav
    
    Babbling Data
    Z-score threshold : zscore < -0.65 (137)
    Variance threshold : var < 0.0016  (111)
    Remove Data : 5_15, 11_41, 22694-B-20-2, 50665-A-20_3~4, 59579-B-20_3~4, 198411-A-20_2, 198411-G-20_4, 8_16, 80482-A-20_0.wav
    
    
    1.96 / -1.96
    """
    
    path = '/Users/gimseong-yun/Downloads/Baby_cry/'
    dst = '/Users/gimseong-yun/Downloads/TTA_Dataset/1.baby/testset/'
    Fs = 16000
    rmsList, varList, fileList = [], [], []
    
    for file in os.listdir(path):
        audio_temp, Fs = librosa.load(path + file, sr=Fs, mono=True)
        rms = np.sqrt(np.mean(np.power(audio_temp, 2)))
        varList.append(np.var(audio_temp))
        rmsList.append(rms)
        fileList.append(file)
    
    standadized_data = ss.zscore(rmsList)
    
    # print(len(list(filter(lambda x: x < 0.0016, varList))))
    
    cry_file_list, babbling_file_list = [], []
    var1, var2 = [], []
    for i, zscore in enumerate(standadized_data):
        if zscore > 0.8 and varList[i] > 0.1:
            cry_file_list.append(fileList[i])
            # shutil.copy(path + fileList[i], dst + 'cry/')
        elif zscore < -0.65 and varList[i] < 0.0016:
            babbling_file_list.append(fileList[i])
            # shutil.copy(path + fileList[i], dst + 'babbling/')  
        
    print(len(cry_file_list))           # 101
    print(len(babbling_file_list))      # 111


def labels_to_csv(label_path, fileNm, duration, nClass, src_file, take, datasetName):
    if not os.path.exists(label_path):
        with open(label_path, 'w', newline='') as label_file:
            label_file.write('file_name,duration,class,src_file,take,dataset\n')
    
    with open(label_path, 'a', newline='') as label_file:
        wr = csv.writer(label_file)
        wr.writerow([fileNm, duration, nClass, src_file,take, datasetName])
        
def ESC50(save_path, label_path, nClass, Fs):
    datasetName = 'ESC-50'
    data_path = '/Volumes/SSD 2TB #2/Baby_training/ESC-50-master/audio/'
    meta_path = '/Volumes/SSD 2TB #2/Baby_training/ESC-50-master/meta/esc50.csv'
    
    with open(meta_path, 'r') as meta_file:
        reader = csv.reader(meta_file)
        labels_list = list(reader)[1:]
        
    for labels in tqdm(labels_list):
        fName, category, src_file, take = labels[0], labels[3], labels[5], labels[6]
        
        if category == 'crying_baby':
            audio_data, Fs = librosa.load(data_path + fName, Fs)
            duration = librosa.get_duration(audio_data, Fs)
            labels_to_csv(label_path, fName, duration, nClass, src_file, take, datasetName)
            save_path_full = save_path + nClass + '/' + datasetName + '/' + fName
            sf.write(save_path_full, audio_data, Fs, format='wav')

def FSD50K(save_path, label_path, nClass, Fs):
    datasetName = 'FSD50K'
    path = '/Users/gimseong-yun/Desktop/Dataset/FSD50K/'
    data_path = [path + 'FSD50K.dev_audio/',
                 path + 'FSD50K.eval_audio/']
    ground_truth_path = [path + 'FSD50K.ground_truth/dev.csv',
                         path + 'FSD50K.ground_truth/eval.csv']
    meta_path = [path + 'FSD50K.metadata/dev_clips_info_FSD50K.json',
                 path + 'FSD50K.metadata/eval_clips_info_FSD50K.json']
    
    for i in range(2):  # 0: dev, 1: eval
        with open(ground_truth_path[i], 'r') as ground_truth_file:
            reader = csv.reader(ground_truth_file)
            labels_list = list(reader)[1:]
        
        for labels in tqdm(labels_list):
            fName, category = labels[0] + '.wav', labels[1]
            
            if 'Crying_and_sobbing' in category:    
                with open(meta_path[i], 'r') as meta_file:
                    meta_data = json.load(meta_file)
                
                meta_temp = str(meta_data[fName.rstrip('.wav')])
                keywords = ['Baby', 'baby', 'toddler', 'Toddler', 'infant', 'Infant']
                
                if any(keyword in meta_temp for keyword in keywords):
                    audio_data, Fs = librosa.load(data_path[i] + fName, Fs)
                    
                    duration = librosa.get_duration(audio_data, Fs)
                    src_file = fName.rstrip('.wav')
                    labels_to_csv(label_path, fName, duration, nClass, src_file, 'A', datasetName)
                    
                    save_path_full = save_path + nClass + '/' + datasetName + '/' + fName
                    sf.write(save_path_full, audio_data, Fs, format='wav')
                                      
def TUT_DCASE2017(save_path, label_path, nClass, Fs):
    datasetName = 'TUT_DCASE2017'
    data_path = '/Volumes/SSD 2TB #2/Baby_training/TUT/TUT-rare-sound-events-2017-data/'
    
    with open(label_path) as labels_file:
        reader = csv.reader(labels_file)
        labels_list = list(reader)[1:]
        src_file_list = [i[0] for i in labels_list]
        
    
    for file in tqdm(os.listdir(data_path)):
        if file.endswith('.wav'):
            src_file = file.rstrip('.wav')
            if src_file in src_file_list:
                print(src_file)
                continue
            
            audio_data, Fs = librosa.load(data_path + file, Fs)
            duration = librosa.get_duration(audio_data, Fs)
            labels_to_csv(label_path, file, duration, nClass, src_file, 'A', datasetName)
            
            save_path_full = save_path + nClass + '/' + datasetName + '/' + file
            sf.write(save_path_full, audio_data, Fs, format='wav')
    
def Baby_cry_Kaggle(save_path, label_path, nClass, Fs):
    datasetName = 'Baby_cry_Kaggle'
    data_paths = ['/Volumes/SSD 2TB #2/Baby_training/Baby_cry2/train/',
                 '/Volumes/SSD 2TB #2/Baby_training/Baby_cry2/test/']
    
    for data_path in data_paths:
        for file in tqdm(os.listdir(data_path)):
            if file.startswith('.'): continue
                
            audio_data, Fs = librosa.load(data_path + file, Fs)
            duration = librosa.get_duration(audio_data, Fs)
            src_file = file.rstrip('.wav')
            labels_to_csv(label_path, file, duration, nClass, src_file, 'A', datasetName)
            
            save_path_full = save_path + nClass + '/' + datasetName + '/' + file
            sf.write(save_path_full, audio_data, Fs, format='wav')
                
def giulbia_github(save_path, label_path, nClass, Fs):
    datasetName = 'giulbia_github'
    data_path = '/Users/gimseong-yun/Downloads/baby_cry_detection-master/data/crying/'
    
    for file in os.listdir(data_path):
        if file.startswith('.'): continue
        
        audio_data, Fs = librosa.load(data_path + file, Fs)
        duration = librosa.get_duration(audio_data, Fs)
        src_file = file.rstrip('.wav').rsplit('_', 1)[0]
        take = file.rstrip('.wav').rsplit('_', 1)[1]
        labels_to_csv(label_path, file, duration, nClass, src_file, take, datasetName)
        
        save_path_full = save_path + nClass + '/' + datasetName + '/' + file
        sf.write(save_path_full, audio_data, Fs, format='wav')
    
def m4a_to_wav(file_path, save_path, fName):
    nfile = fName.rstrip('.m4a') + '.wav'
    m4aFile = AudioSegment.from_file(os.path.join(file_path,fName), 'm4a')
    m4aFile.export(os.path.join(save_path, nfile), format='wav')
    
            
    
    
if __name__ == '__main__':
    save_path = '/Volumes/SSD 2TB #2/Baby_training/training/'
    label_path = '/Volumes/SSD 2TB #2/Baby_training/training/cry_labels.csv'
    nClass = ['cry', 'babbling', 'laughter']
    Fs = 16000
    
    
    # ============ ESC-50 ============
    # ESC50(save_path, label_path, nClass, Fs)
    
    # ============ FSD50K ============
    # FSD50K_cry(save_path, label_path, nClass[0], Fs)
    
    # ============ TUT_DCASE2017 ============
    # TUT_DCASE2017(save_path, label_path, nClass[0], Fs)
    
    # ============ Baby_cry_Kaggle ============
    # Baby_cry_Kaggle(save_path, label_path, nClass[0], Fs)
    
    # ============ giulbia_github ============
    # giulbia_github(save_path, label_path, nClass[0], Fs)
    
    # ============ Total Dataset Info ============
    # dataSet = ['ESC50', 'FSD50K', 'TUT_DCASE2017', 'Baby_cry_Kaggle', 'giulbia_github']
    
    # with open(label_path) as labels_file:
    #     reader = csv.reader(labels_file)
    #     labels_list = list(reader)[1:]
    
    # duration = 0.0
    # for labels in labels_list:
    #     dataSetName = labels[5]
    #     duration += float(labels[1])
    # print(duration)
    
    input_path = '/Users/gimseong-yun/Downloads/laughter-training-data-master/baby_laughter_clips'
    output_path = '/Users/gimseong-yun/Downloads/data'
    for file in os.listdir(input_path):
        if file.startswith('.'): continue
        
        m4a_to_wav(input_path, output_path, file)
    
    
    """
    labels.csv path = '/Volumes/SSD 2TB #2/Baby_training/training/labels.csv'
    
    Info
        파일 형식 : wav
        Fs : 16000
        음원 파일 개수 : 1308개
        데이터셋 크기 : 761.88 MB
        데이터셋 길이 : 23807s = 396.78분 = 6.613시간
        수집한 데이터셋 : FSD50K, ESC50, TUT_DCASE2017, Baby_cry_Kaggle, giulbia_github
        
    Total Dataset
        음원 파일 개수 : 1306개
        데이터셋 크기 : 757.588 MB
        데이터셋 길이 : 23672초 = 394.53분 = 6.576시간
    """

    
    
    
    
    
        
        
        
        

    
    