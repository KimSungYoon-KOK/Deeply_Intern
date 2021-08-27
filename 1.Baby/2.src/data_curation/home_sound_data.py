import os
from glob import glob
import csv
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
import shutil
from random import randrange

class Curator():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.Fs = 16000
        
    def labels_to_csv(self, labels_path, meta):
        if not os.path.exists(labels_path):
            with open(labels_path, 'w', newline='') as labels_file:
                labels_file.write('file_name,duration,category,src_file,take,dataset\n')
        
        with open(labels_path, 'a', newline='') as labels_file:
            wr = csv.writer(labels_file)
            fileNm,duration,category,src_file,take,datasetName = meta
            wr.writerow([fileNm, duration, category, src_file,take, datasetName])

    # Freesound.org 파일 ID 중복 체크
    def check_duplicates_src_file(self, src_dict, src_file, datasetName):
        if src_dict != None:        
            if (src_file in src_dict) and (src_dict[src_file] != datasetName):
                return False
        
        return True

    # 새로 저장할 라벨 파일 오픈
    def open_labels(self, labels_path):
        src_dict = {}

        if not os.path.exists(labels_path):
            with open(labels_path, 'w', newline='') as labels_file:
                labels_file.write('file_name,duration,category,src_file,take,dataset\n')
            
        else:
            with open(labels_path) as labels_file:
                reader = csv.reader(labels_file)
                temp_list = list(reader)[1:]

            if len(temp_list) > 0:
                for i in temp_list:
                    src_dict[i[3]] = i[5]
            
        return src_dict

    # 기존 데이터셋의 메타 데이터 파일 오픈
    def open_meta_data(self, meta_path):
        with open(meta_path) as meta_file:
            reader = csv.reader(meta_file)
            meta_list = list(reader)[1:]
        
        return meta_list

    def total_duration(self,labels_path):
        with open(labels_path) as label_file:
            reader = csv.reader(label_file)
            labels_list = list(reader)[1:]
            
        duration = 0.0
        for labels in labels_list:
            duration += float(labels[1])
        
        return duration

    def cutoff_data(self, data_path, save_path, Fs, sec, durationThreshold):
            
        for file in tqdm(os.listdir(data_path)):
            if not file.endswith('.wav'): continue

            file_path = os.path.join(data_path, file)
            audio_data, Fs = librosa.load(file_path, Fs)
            cutoff_data = audio_data[:(sec*Fs)]

            if librosa.get_duration(cutoff_data, Fs) == durationThreshold:
                save_file = os.path.join(save_path, file)
                sf.write(save_file, cutoff_data, Fs, format='wav')

    # 불러온 메타 데이터에서 열 추출
    def extract_column(self, mList, column):
        transList = np.array(mList).T.tolist()
        if len(transList) > column:
            return transList[column]
        else:
            print("ERROR : out of range")

    def split_data(self, sec, trim=True):
        data_path = self.root_dir + '0.raw/laughter_test_1/'
        dst = self.root_dir + '1.dataset/train/home_sound/'

        meta_path = '/home/ksy/1.Baby/1.dataset/meta/train_home_sound_labels.csv'
        meta_list = self.open_meta_data(meta_path)

        meta_path = self.root_dir + '1.dataset/meta/train_home_sound_splitted_labels.csv'
        with open(meta_path, 'a', newline='') as meta_file:
            wr = csv.writer(meta_file)
            
            for meta in tqdm(meta_list):
                fName, duration, category, src_file, take, dataset = meta

                if dataset == 'ESC-50':
                    data_path = '/home/ksy/0.OriginalData/ESC50/audio/'
                elif dataset == 'FSD50K':
                    data_path = '/home/ksy/0.OriginalData/FSD50K/audio/'
                elif dataset == 'UrbanSound8K':
                    data_path = '/home/ksy/0.OriginalData/UrbanSound8K/audio/'

                file_path = data_path + fName
                audio_data, Fs = librosa.load(file_path, self.Fs)

                if trim:
                    trimmed_audio, index = librosa.effects.trim(audio_data, top_db=20)
                else:
                    trimmed_audio = audio_data

                for i in range(len(trimmed_audio) // Fs):
                    splitted_audio = trimmed_audio[i*Fs : (i+sec)*Fs]
                    newName = fName.rstrip('.wav') + '_' + str(i+1) + '.wav'
                    split_duration = librosa.get_duration(splitted_audio, Fs)
                    split_meta = [newName, split_duration, category, src_file, take + '_'+ str(i+1), dataset]

                    sf.write((dst + newName), splitted_audio, Fs, format='wav')
                    wr.writerow(split_meta)



if __name__ == '__main__':
    # ============== Test Set Meta Data ==============
    def testset_to_metadata():
        curator = Curator()
        root_dir = '/home/ksy/1.Baby/1.dataset/test/home_sound/'
        labels_path = '/home/ksy/1.Baby/1.dataset/meta/test_home_sound_labels.csv'

        urban_meta_path = '/home/ksy/0.OriginalData/UrbanSound8K/metadata/UrbanSound8K.csv'
        urban_meta_list = curator.open_meta_data(urban_meta_path)
        urban_fName_list = curator.extract_column(urban_meta_list, 0)

        meta_path = '/home/ksy/0.OriginalData/Home_Sound/meta/home_sound_labels.csv'
        meta_list = curator.open_meta_data(meta_path)
        fName_list = curator.extract_column(meta_list, 0)

        for fName in os.listdir(root_dir):
            try:
                fName_idx = fName_list.index(fName)
                fileNm, duration, category, src_file, take, datasetName = meta_list[fName_idx]
                curator.labels_to_csv(labels_path, fileNm, 1.0, category, src_file, take, datasetName)

            except ValueError:

                try:
                    fName_idx = urban_fName_list.index(fName)
                    slice_file_name, fsID, start, end, salience, fold, classID, nClass = urban_meta_list[fName_idx]
                    take = slice_file_name.rstrip('.wav').rsplit('-',1)[1]
                    curator.labels_to_csv(labels_path, slice_file_name, 1.0, nClass, fsID, take, 'UrbanSound8K')
                
                except ValueError:
                    src_file = fName.rstrip('.wav')
                    take = 'A'
                    nClass = 'home_sound'
                    datasetName = 'FSD50K' if len(fName.split('-')) == 1 else 'ESC-50'
                    curator.labels_to_csv(labels_path, fName, 1.0, nClass, src_file, take, datasetName)
    # testset_to_metadata()

    # ============== Train Set to Meta Data ==============
    def trainset_to_metadata():
        durationThreshold = 1.0
        Fs = 16000

        curator = Curator()
        train_meta_path = '/home/ksy/1.Baby/1.dataset/meta/train_home_sound_labels.csv'

        temp_list = curator.open_meta_data('/home/ksy/1.Baby/1.dataset/meta/test_home_sound_labels.csv')
        fsID_set = set(curator.extract_column(temp_list, 3))

        # ESC-50
        # filename,fold,target,category,esc10,src_file,take
        esc_meta_path = '/home/ksy/0.OriginalData/ESC50/meta/esc50.csv'
        esc_meta_list = curator.open_meta_data(esc_meta_path)
        esc_class = ['dog', 'rain', 'door_wood_knock', 'sneezing', 'mouse_click', 'clapping', 'keyboard_typing', 'siren', 'door_wood_creaks', 'car_horn', 'chirping_birds', 
                    'coughing', 'can_opening', 'engine', 'cat', 'water_drops', 'footsteps', 'washing_machine', 'train', 'laughing', 'vacuum_cleaner', 'pouring_water', 
                    'brushing_teeth', 'clock_alarm', 'airplane', 'toilet_flush', 'snoring', 'clock_tick', 'thunderstorm', 'drinking_sipping', 'glass_breaking']

        # FSD50K
        # fname,labels,mids,split
        # fname,labels,mids
        fsd_meta_path = ['/home/ksy/0.OriginalData/FSD50K/FSD50K.ground_truth/dev.csv', '/home/ksy/0.OriginalData/FSD50K/FSD50K.ground_truth/eval.csv']
        fsd_meta_list_dev = curator.open_meta_data(fsd_meta_path[0])
        fsd_meta_list_eval = curator.open_meta_data(fsd_meta_path[1])
        fsd_class = set(['Domestic_sounds_and_home_sounds', 'Domestic_animals_and_pets'])
        fsd_data_path = ['/home/ksy/0.OriginalData/FSD50K/FSD50K.dev_audio', '/home/ksy/0.OriginalData/FSD50K/FSD50K.eval_audio']

        # UrbanSound8K
        urban_meta_path = '/home/ksy/0.OriginalData/UrbanSound8K/metadata/UrbanSound8K.csv'
        urban_meta_list = curator.open_meta_data(urban_meta_path)
        urban_class = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'engine_idling', 'siren']

        with open(train_meta_path, 'a', newline='') as train_meta_file:
            wr = csv.writer(train_meta_file)

            # ESC-50
            for meta in tqdm(esc_meta_list):
                filename, fold, target, category, esc10, src_file, take = meta
                duration = 5.0
                if (category in esc_class) and (src_file not in fsID_set):
                    wr.writerow([filename, duration, category, src_file, take, 'ESC-50'])
            
            # FSD50K
            for i, fsd_meta_list in enumerate([fsd_meta_list_dev, fsd_meta_list_eval]):
                for meta in tqdm(fsd_meta_list):
                    fName = meta[0] + '.wav'
                    src_file = meta[0]
                    categories = set(meta[1].split(','))

                    if (len(categories.intersection(fsd_class)) >= 1) and (src_file not in fsID_set):
                        audio_data, Fs = librosa.load(os.path.join(fsd_data_path[i], fName), Fs)
                        duration = librosa.get_duration(audio_data, Fs)
                        if duration > durationThreshold:
                            wr.writerow([fName, duration, str(categories), src_file, 'A', 'FSD50K'])
                        
                    

            # UrbanSound8K
            for meta in tqdm(urban_meta_list):
                slice_file_name, fsID, start, end, salience, fold, classID, className = meta
                duration = float(end) - float(start)
                take = slice_file_name.rstrip('.wav').rsplit('-',1)[1]

                if (className in urban_class) and (fsID not in fsID_set) and (duration > 1.0):
                    wr.writerow([slice_file_name, duration, className, fsID, take, 'UrbanSound8K'])
    # trainset_to_metadata()

    root_dir = '/home/ksy/1.Baby/'
    curator = Curator(root_dir)
    # curator.split_data(sec=1)

    def testset():
        save_meta_path = '/home/ksy/1.Baby/1.dataset/meta/test_home_sound_splitted_labels.csv'
        save_data_path = '/home/ksy/1.Baby/1.dataset/test/home_sound/'
        data_path = '/home/ksy/1.Baby/1.dataset/train/home_sound/'
        meta_path = '/home/ksy/1.Baby/1.dataset/meta/home_sound_splitted_labels.csv'
        meta_list = curator.open_meta_data(meta_path)
        srcfile_list = curator.extract_column(meta_list, 3)

        test_list = []
        rand_idx_list = []
        while len(test_list) < len(meta_list)*0.15:
            rand_idx = randrange(len(meta_list))
            if rand_idx in rand_idx_list: continue      # 중복 제거
            rand_idx_list.append(rand_idx)
            file_name, duration, category, src_file, take, dataset = meta_list[rand_idx]

            idx_list = np.where(np.array(srcfile_list) == src_file)[0].tolist()
            for idx in idx_list:
                file_name,duration,category,src_file,take,dataset = meta_list[idx]
                test_list.append(meta_list[idx])
                
                path = data_path + file_name
                if os.path.isfile(path):
                    shutil.move(path, save_data_path + file_name)
                    curator.labels_to_csv(save_meta_path, meta_list[idx])
        
        print(len(test_list))

    
        
        
            
            

        


