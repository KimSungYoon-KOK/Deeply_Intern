import os
from glob import glob
import csv, json
import re
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import librosa.display
import soundfile as sf
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import shutil


class Curator():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.Fs = 16000
        
    # 기존 데이터셋의 메타 데이터 파일 오픈
    def open_meta_data(self, meta_path):
        with open(meta_path) as meta_file:
            reader = csv.reader(meta_file)
            meta_list = list(reader)[1:]
        
        return meta_list

    # 불러온 메타 데이터에서 열 추출
    def extract_column(self, mList, column):
        transList = np.array(mList).T.tolist()
        if len(transList) > column:
            return transList[column]
        else:
            print("ERROR : out of range")

    def labels_to_csv(self, label_path, meta_list):
        if not os.path.exists(label_path):
            with open(label_path, 'w', newline='') as label_file:
                label_file.write('file_name,duration,class,src_file,take,dataset\n')
    
        with open(label_path, 'a', newline='') as label_file:
            wr = csv.writer(label_file)
            for meta in meta_list:
                wr.writerow(meta)

    # wav 파일 도표
    def plot_wav_list(self, path_list):
        fig, ax = plt.subplots(nrows=len(path_list), sharex=True)
        for i, path in enumerate(path_list):
            audio_temp, Fs = librosa.load(path, self.Fs)
            librosa.display.waveshow(audio_temp, sr=Fs, ax=ax[i])
            ax[i].set(title = os.path.basename(path))
            ax[i].label_outer()    

    # Split Cry Data 1s
    def split_cry_data(self, sec):
        data_path = self.root_dir + '0.raw/cry/'
        cry_dst = self.root_dir + '0.raw/cry_1/'
        
        meta_path = self.root_dir + 'meta/train_baby_cry_labels.csv'
        meta_list = self.open_meta_data(meta_path)

        meta_path = self.root_dir + 'meta/train_baby_cry_splitted_labels.csv'
        with open(meta_path, 'a', newline='') as meta_file:
            wr = csv.writer(meta_file)
            
            for meta in tqdm(meta_list):
                fName, duration, className, src_file, take, datasetName = meta
                file_path = data_path + fName

                audio_data, Fs = librosa.load(file_path, self.Fs)
                trimmed_audio, index = librosa.effects.trim(audio_data, top_db=20)

                for i in range(len(trimmed_audio) // Fs):
                    splited_audio = trimmed_audio[i*Fs : (i+sec)*Fs]
                    newName = fName.rstrip('.wav') + '_' + str(i+1) + '.wav'
                    sf.write((cry_dst + newName), splited_audio, Fs, format='wav')
                    wr.writerow([newName, 1.0, className, src_file,take, datasetName])

    # Split laughter Data 1s
    def split_laughter_data(self, sec, trim=True):
        data_path = self.root_dir + '0.raw/laughter_test_1/'
        dst = self.root_dir + '1.dataset/test/laughter/'

        meta_path = self.root_dir + '1.dataset/meta/test_laughter_splitted_labels.csv'
        with open(meta_path, 'a', newline='') as meta_file:
            wr = csv.writer(meta_file)
            
            for fName in tqdm(os.listdir(data_path)):
                if fName.startswith('.'): continue

                file_path = data_path + fName
                audio_data, Fs = librosa.load(file_path, self.Fs)
                # /home/ksy/1.Baby/0.raw/laughter_test_1/laugh_2.m4a_19.wav
                if trim:
                    trimmed_audio, index = librosa.effects.trim(audio_data, top_db=20)
                else:
                    trimmed_audio = audio_data

                for i in range(len(trimmed_audio) // Fs):
                    splitted_audio = trimmed_audio[i*Fs : (i+sec)*Fs]
                    newName = fName.rstrip('.wav') + '_' + str(i+1) + '.wav'
                    duration = librosa.get_duration(splitted_audio, Fs)
                    meta = [newName, duration, 'laughter', fName, str(i+1), 'AudioSet']

                    sf.write((dst + newName), splitted_audio, Fs, format='wav')
                    wr.writerow(meta)

    # Validate Data            
    def validate_data(self):
        data_path = self.root_dir + '1.dataset/train/laughter/'
        meta_path = self.root_dir + '1.dataset/meta/train_laughter_splitted_labels.csv'
        meta_list = self.open_meta_data(meta_path)
        
        fName_list = self.extract_column(meta_list, 0)[0:5]
        path_list = [data_path + fName for fName in fName_list]
        self.plot_wav_list(path_list)

        for file in path_list:
            audio_data, Fs = librosa.load(file, self.Fs)
            rms = np.sqrt(np.mean(np.power(audio_data, 2)))
            variance = np.var(audio_data)
            print(os.path.basename(file), rms, variance)

    # Cry & Babbling Rule_based_Classifier
    def rule_based_classifier(self):
        """
        ==== Test Set ====
            Cry Data
                Z-score threshold : zscore > 0.8 (104)
                Variance threshold : var > 0.1   (101)
                Remove Data : 4_25.wav     
            Babbling Data
                Z-score threshold : zscore < -0.65 (137)
                Variance threshold : var < 0.0016  (111)       
                Remove Data : 5_15, 11_41, 22694-B-20-2, 50665-A-20_3~4, 59579-B-20_3~4, 198411-A-20_2, 198411-G-20_4, 8_16, 80482-A-20_0.wav

        ==== Train Set ====
            Cry Data
                Z-score threshold : zscore >= 1.0   (2117)      
            Babbling Data
                Z-score threshold : zscore <= 0     
                Variance threshold : var < 0.0016   (8568)
        """

        # data_path = self.root_dir + '0.raw/cry_1/'
        data_path = '/home/ksy/0.OriginalData/Baby_cry/'
        dst = self.root_dir + '1.dataset/test/'
        rmsList, varList, fileList = [], [], []
        
        for file_path in tqdm(glob(data_path + '*')):
            fName = os.path.basename(file_path)
            if not fName.endswith('.wav'): continue

            audio_temp, Fs = librosa.load(file_path, self.Fs)
            rms = np.sqrt(np.mean(np.power(audio_temp, 2)))
            varList.append(np.var(audio_temp))
            rmsList.append(rms)
            fileList.append(fName)
        
        standadized_data = ss.zscore(rmsList)
        
        # print(len(list(filter(lambda x: x < 0.0016, varList))))
        
        cry_file_list, babbling_file_list = [], []
        for i, zscore in enumerate(tqdm(standadized_data)):
            if zscore >= 0.8:
                cry_file_list.append(fileList[i])
                # shutil.copy(data_path + fileList[i], dst + 'cry/')
            elif zscore <= -0.65 and varList[i] < 0.0016:
                babbling_file_list.append(fileList[i])
                shutil.copy(data_path + fileList[i], dst + 'babbling/')  
            
        print(len(cry_file_list))           
        print(len(babbling_file_list))

    # Baby_web Data labeling
    def labeling_baby_web(self, durationThreshold):
        meta_path = self.root_dir + 'Baby_web/label/'
        categories = ['silent', 'cry', 'babbling', 'home_sound']
        
        newMeta_path = '/home/ksy/1.Baby/1.dataset/meta/train_baby_web_labels.csv'
        newMeta_list, name_dict = [], {}
        cry_cnt, cry_du, babbling_cnt, babbling_du = 0, 0, 0.0, 0.0
        for path in glob(meta_path + '*'):
            src_file = os.path.basename(path)
            if src_file.startswith('.') or src_file[-1] == '~': continue

            src_file = src_file.rstrip('.txt') + '.wav'
            
            with open(path, 'r') as meta_file:
                reader = meta_file.read().splitlines()

                for line in tqdm(reader):
                    line = re.split('\t| ', line)
                    try:
                        start, end, category = float(line[0]), float(line[1]), categories[int(line[-1])]
                        duration = end - start
                        
                        if category == 'cry':
                            cry_cnt += 1
                            cry_du += duration
                        elif category == 'babbling':
                            babbling_cnt += 1
                            babbling_du += duration
                        
                        if duration >= durationThreshold:
                            file_name = '_'.join([category, src_file.split('.')[0]])
                            
                            if file_name in name_dict:
                                name_dict[file_name] += 1
                            else:
                                name_dict[file_name] = 1

                            file_name = '_'.join([file_name, str(name_dict[file_name])])
                            newMeta_list.append([file_name, duration, category, src_file, '1', 'Baby_web', start, end])
                    
                    except ValueError as e:
                        print(e)
                        print(src_file)
                        print(line)
                        break
        print(cry_cnt, cry_du)
        print(babbling_cnt, babbling_du)
        self.labels_to_csv(newMeta_path, newMeta_list)

    # Extract Cry, Babbling Data of Baby_web
    def extract_baby_web(self, durationThreshold):
        data_path = self.root_dir + 'Baby_web/audio/'
        save_path = '/home/ksy/1.Baby/0.raw/baby_web/original/'

        meta_path = '/home/ksy/1.Baby/0.raw/baby_web/baby_web_labels.csv'
        meta_list = self.open_meta_data(meta_path)

        audio_data, prev_file = [], ''
        category_cnt = {'cry': 0, 'babbling': 0}
        for meta in tqdm(meta_list):
            file_name, duration, category, src_file, take, dataset,  start, end = meta

            if category not in ['cry', 'babbling']: continue
            
            # 직전에 한 번 load 했던 파일이면 다시 load 안 할수 있게
            if src_file != prev_file:
                wav_file = data_path + src_file
                mp3_file = data_path + src_file.rstrip('.wav') + '.mp3'

                if os.path.isfile(wav_file):
                    audio_data, Fs = librosa.load(wav_file, self.Fs)
                
                elif os.path.isfile(mp3_file):
                    sound = AudioSegment.from_mp3(data_path + mp3_file)
                    sound.export(data_path + src_file, format="wav")
                    audio_data, Fs = librosa.load(data_path + src_file, self.Fs)
                    
                else:
                    print('No such File : ', wav_file)
                    continue

            # 데이터에서 분리해서 각각 wav 파일로 저장
            start_idx = int(float(start) * self.Fs)
            end_idx = int(float(end) * self.Fs)
            splitted_data = audio_data[start_idx : end_idx]
            
            save_file = save_path + category + '/' + file_name + '.wav'
            sf.write(save_file, splitted_data, self.Fs, format='wav')
            
            category_cnt[category] += 1
            prev_file = src_file

    def split_baby_web(self, durationThreshold=1.0):
        data_path = ['/home/ksy/1.Baby/0.raw/baby_web/original/babbling', '/home/ksy/1.Baby/0.raw/baby_web/original/cry']
        save_path = '/home/ksy/1.Baby/0.raw/baby_web/splitted_data/'

        for class_path in data_path:
            category = os.path.basename(class_path)
            print(category)
            
            for file_path in tqdm(glob(class_path + '/*')):
                raw_file_name = os.path.basename(file_path).rstrip('.wav')
                audio_data, Fs = librosa.load(file_path, self.Fs)
                raw_duration = librosa.get_duration(audio_data, self.Fs)
                
                for i in range(int(raw_duration)):
                    start_idx, end_idx = i*self.Fs, (i+1)*self.Fs
                    splitted_data = audio_data[start_idx : end_idx]

                    new_file_name = '_'.join([raw_file_name, str(i+1)]) + '.wav'
                    if librosa.get_duration(splitted_data, self.Fs) == durationThreshold:
                        save_file = save_path + category + '/' + new_file_name
                        sf.write(save_file, splitted_data, self.Fs, format='wav')
                    
                    else:
                        print("Duration Error : ", new_file_name)
                        continue




class BabyCryDataset():

    def labels_to_csv(self, label_path, fileNm, duration, nClass, src_file, take, datasetName):
        if not os.path.exists(label_path):
            with open(label_path, 'w', newline='') as label_file:
                label_file.write('file_name,duration,class,src_file,take,dataset\n')
        
        with open(label_path, 'a', newline='') as label_file:
            wr = csv.writer(label_file)
            wr.writerow([fileNm, duration, nClass, src_file,take, datasetName])
            
    def ESC50(self, save_path, label_path, nClass, Fs):
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
                self.labels_to_csv(label_path, fName, duration, nClass, src_file, take, datasetName)
                save_path_full = save_path + nClass + '/' + datasetName + '/' + fName
                sf.write(save_path_full, audio_data, Fs, format='wav')

    def FSD50K(self, save_path, label_path, nClass, Fs):
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
                        self.labels_to_csv(label_path, fName, duration, nClass, src_file, 'A', datasetName)
                        
                        save_path_full = save_path + nClass + '/' + datasetName + '/' + fName
                        sf.write(save_path_full, audio_data, Fs, format='wav')
                                        
    def TUT_DCASE2017(self, save_path, label_path, nClass, Fs):
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
                self.labels_to_csv(label_path, file, duration, nClass, src_file, 'A', datasetName)
                
                save_path_full = save_path + nClass + '/' + datasetName + '/' + file
                sf.write(save_path_full, audio_data, Fs, format='wav')
        
    def Baby_cry_Kaggle(self, save_path, label_path, nClass, Fs):
        datasetName = 'Baby_cry_Kaggle'
        data_paths = ['/Volumes/SSD 2TB #2/Baby_training/Baby_cry2/train/',
                    '/Volumes/SSD 2TB #2/Baby_training/Baby_cry2/test/']
        
        for data_path in data_paths:
            for file in tqdm(os.listdir(data_path)):
                if file.startswith('.'): continue
                    
                audio_data, Fs = librosa.load(data_path + file, Fs)
                duration = librosa.get_duration(audio_data, Fs)
                src_file = file.rstrip('.wav')
                self.labels_to_csv(label_path, file, duration, nClass, src_file, 'A', datasetName)
                
                save_path_full = save_path + nClass + '/' + datasetName + '/' + file
                sf.write(save_path_full, audio_data, Fs, format='wav')
                    
    def giulbia_github(self, save_path, label_path, nClass, Fs):
        datasetName = 'giulbia_github'
        data_path = '/Users/gimseong-yun/Downloads/baby_cry_detection-master/data/crying/'
        
        for file in os.listdir(data_path):
            if file.startswith('.'): continue
            
            audio_data, Fs = librosa.load(data_path + file, Fs)
            duration = librosa.get_duration(audio_data, Fs)
            src_file = file.rstrip('.wav').rsplit('_', 1)[0]
            take = file.rstrip('.wav').rsplit('_', 1)[1]
            self.labels_to_csv(label_path, file, duration, nClass, src_file, take, datasetName)
            
            save_path_full = save_path + nClass + '/' + datasetName + '/' + file
            sf.write(save_path_full, audio_data, Fs, format='wav')
        
    def m4a_to_wav(self, file_path, save_path, fName):
        nfile = fName.rstrip('.m4a') + '.wav'
        m4aFile = AudioSegment.from_file(os.path.join(file_path,fName), 'm4a')
        m4aFile.export(os.path.join(save_path, nfile), format='wav')
        

if __name__ == '__main__':
    # ============ Baby Cry Dataset Collect ============
    def collect_baby_cry_dataset():
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
    # collect_baby_cry_dataset()

    """
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

    root_dir = '/home/ksy/0.OriginalData/'
    curator = Curator(root_dir)
    # curator.split_cry_data(sec=1)
    # curator.validate_data()
    # curator.rule_based_classifier()
    # curator.split_laughter_data(sec=1, trim=False)
    # curator.labeling_baby_web(durationThreshold=0.0)
    # curator.extract_baby_web(durationThreshold=0.0)
    # curator.merge_audio()
    curator.split_baby_web()


