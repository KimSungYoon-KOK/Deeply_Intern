import os
import csv
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np

"""
[ FSD50K ]
Domestic_sounds_and_home_sounds
    Toilet_flush                    o
    Sink_(filling_or_washing)       o
    Microwave_oven                  o
    Door                            o
    Cutlery_and_silverware          o
    Dishes_and_pots_and_pans        o
    Typing                          o
    Frying                          o
    
[ UrbanSound8K ]
air_conditioner     o
car_horn            o
children_playing    o
dog_bark            o
engine_idling       o
siren               o

[ ESC-50 ]
Interior/domestic sounds
    Mouse click         o
    Clock alarm         o
    Vacuum cleaner      o
Exteriror/urban noises
    Train               o
    Airplane            o
    Washing machine     o
"""
class TestSet():    
    def durationCheck(data_path, fname):
        sr, wav_data = sio.wavfile.read(data_path+fname)
        duration = len(wav_data)/float(sr)
        if duration >= 1:
            return True
        
    def labelCheck_FSD(labels):
        if 'Toilet_flush' in labels and 'Domestic_sounds_and_home_sounds' in labels:
            return 0, 'toilet_flush'
        elif 'Sink_(filling_or_washing)' in labels and 'Domestic_sounds_and_home_sounds' in labels:
            return 1, 'sink_(filling_or_washing)'
        elif 'Microwave_oven' in labels and 'Domestic_sounds_and_home_sounds' in labels:
            return 2, 'microwave_oven'
        elif 'Door' in labels and 'Domestic_sounds_and_home_sounds' in labels:
            return 3, 'door'
        elif 'Cutlery_and_silverware' in labels and 'Domestic_sounds_and_home_sounds' in labels:
            return 4, 'cutlery_and_silverware'
        elif 'Dishes_and_pots_and_pans' in labels and 'Domestic_sounds_and_home_sounds' in labels:
            return 5, 'dishes_and_pots_and_pans'
        elif 'Typing' in labels and 'Domestic_sounds_and_home_sounds' in labels:
            return 6, 'typing'
        elif 'Frying' in labels and 'Domestic_sounds_and_home_sounds' in labels:
            return 7, 'frying'
        else:
            return -1, 'None'

    def labelCheck_Urban(label):
        if 'air_conditioner' == label:
            return 0, label
        elif 'car_horn' == label:
            return 1, label
        elif 'children_playing' == label:
            return 2, label
        elif 'dog_bark' == label:
            return 3, label
        elif 'engine_idling' == label:
            return 4, label
        elif 'siren' == label:
            return 5, label
        else:
            return -1, 'None'

    def labelCheck_ESC(label):
        if 'mouse_click' == label:
            return 0, label
        elif 'clock_alarm' == label:
            return 1, label
        elif 'vacuum_cleaner' == label:
            return 2, label
        elif 'train' == label:
            return 3, label
        elif 'airplane' == label:
            return 4, label
        elif 'washing_machine' == label:
            return 5, label
        else:
            return -1, 'None'
        
    def label_classifier(data_path, dst, meta_path, dataset_name, copy=False):
        path = data_path
        meta_file = pd.read_csv(meta_path)
        df = pd.DataFrame(meta_file)
        
        if dataset_name == 'FSD50K':
            labelCheck = labelCheck_FSD
            col = ['fname', 'labels']
        elif dataset_name == 'UrbanSound8K':
            labelCheck = labelCheck_Urban
            col = ['slice_file_name', 'salience', 'fold', 'class']
        elif dataset_name == 'ESC-50':
            labelCheck = labelCheck_ESC
            col = ['filename', 'category']
        else:
            return 0
        
        noise_list = [[],[],[],[],[],[],[],[]]
        pfid = ''
        
        for data in df[col].values:
            
            if dataset_name == 'FSD50K':
                fname, labels = str(data[0]) + '.wav', data[1]
            elif dataset_name == 'UrbanSound8K':
                fname, salience, fold, labels = data[0], data[1], 'fold' + str(data[2]) + '/', data[3]
                if salience != 1: continue
                path += fold
            elif dataset_name == 'ESC-50':
                fname, labels = data[0], data[1]
            else:
                return 0
            
            index, category = labelCheck(labels)
            
            if index != -1 and durationCheck(path, fname):
                noise_list[index].append(fname)
                
                # Remove Duplicate File ID
                # fid = fname.split('-')[0]
                # if pfid != fid:
                #     pfid = fid
                #     noise_list[index].append(fname)
                    
                if copy:
                    shutil.copy(path + fname, dst + category)
            
            path = data_path
                    
        print(dataset_name)            
        for i in noise_list:
            if len(i) != 0:
                print(len(i))
            
        return noise_list

    def data_split(data_path, dst):
        Fs = 16000
        sec = 1
        
        for category in os.listdir(data_path):
            if category.startswith('.'): continue    
            
            for file in os.listdir(data_path + category):
                if file.startswith('.'): continue  
                file_path = data_path + category + '/' + file
                save_path = dst + category + '_slice'
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                audio_data, Fs = librosa.load(file_path, sr=Fs)
                trimmed_data, index = librosa.effects.trim(audio_data)
                
                # duration = librosa.get_duration(trimmed_data, Fs)
                # if duration >= 2:
                #     cutoff_data = trimmed_data[(sec*Fs): (2*Fs)]
                # else:
                #     cutoff_data = trimmed_data[:(sec*Fs)]
                cutoff_data = trimmed_data[:(sec*Fs)]
                
                save_file = save_path + '/' + file
                sf.write(save_file, cutoff_data, Fs, format='wav')

    def label_to_csv(data_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = open(save_path + 'labels.csv', "w")
        f.write('fname,category,label\n')

        for category in os.listdir(data_path):
            if category.startswith('.'): continue    
            
            for file in os.listdir(data_path + category):
                if file.startswith('.'): continue  
                
                f.write(file + ',' + 'noise' + ',' + category + '\n')
                
        f.close()


class TrainingSet():
    def labels_to_csv(self, labels_path, fileNm, duration, category, src_file, take, datasetName):
        if not os.path.exists(labels_path):
            with open(labels_path, 'w', newline='') as labels_file:
                labels_file.write('file_name,duration,category,src_file,take,dataset\n')
        
        with open(labels_path, 'a', newline='') as labels_file:
            wr = csv.writer(labels_file)
            wr.writerow([fileNm, duration, category, src_file,take, datasetName])

    # Freesound.org 파일 ID 중복 체크
    def check_duplicates_src_file(self, src_dict, src_file, datasetName):
        if len(src_dict) > 0:        
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
            labels_list = list(reader)[1:]
        
        return labels_list

    def total_duration(self,labels_path):
        with open(labels_path) as label_file:
            reader = csv.reader(label_file)
            labels_list = list(reader)[1:]
            
        duration = 0.0
        for labels in labels_list:
            duration += float(labels[1])
        
        return duration
        
    def UrbanSound8K(self, labels_path, root_dir, save_path, Fs, durationThreshold):
        datasetName = 'UrbanSound8K'
        data_path = os.path.join(root_dir, 'UrbanSound8K/audio')
        meta_path = os.path.join(root_dir, '/UrbanSound8K/metadata/UrbanSound8K.csv')
        
        src_list, dataset_list = self.open_labels(labels_path)
        labels_list = self.open_meta_data(meta_path)
        
            
        for labels in tqdm(labels_list):
            slice_file_name, fsID, start, end, salience, fold, classID, className = labels
            duration = float(end) - float(start)
            
            if className in category_list:
                if self.check_duplicates_src_file(src_list, dataset_list, fsID, datasetName) and duration > durationThreshold:
                    file_path = os.path.join(data_path, slice_file_name)
                    audio_data = librosa.load(file_path, Fs)
                    
                    temp_name = slice_file_name.rstrip('.wav').rsplit('-',2)
                    take = str(temp_name[1]) + '_' + str(temp_name[2])
                    self.labels_to_csv(labels_path, slice_file_name, duration, className, fsID, take, datasetName)
                    
                    sf.write(os.path.join(save_path, slice_file_name) , audio_data, Fs, format='wav')
            
    def FSD50K(self, labels_path, root_dir, save_path, Fs, durationThreshold):
        datasetName = 'FSD50K'
        category_list = ['Domestic_sounds_and_home_sounds', 'Domestic_animals_and_pets']
        data_path = os.path.join(root_dir, 'FSD50K/FSD50K.dev_audio')
        meta_path = os.path.join(root_dir, 'FSD50K/FSD50K.ground_truth/dev.csv')
        
        src_dict = self.open_labels(labels_path)
        labels_list = self.open_meta_data(meta_path)
                
        for labels in tqdm(labels_list):
            fName, categories = labels[0] + '.wav', labels[1].split(',')
            
            for category in category_list:
                src_file = fName.rstrip('.wav')
                if (category in categories) and self.check_duplicates_src_file(src_dict, src_file, datasetName):
                    audio_data, Fs = librosa.load(os.path.join(data_path, fName), Fs)
                    
                    duration = librosa.get_duration(audio_data, Fs)
                    if duration < durationThreshold: break
                    
                    self.labels_to_csv(labels_path, fName, duration, category, src_file, 'A', datasetName)
                    
                    sf.write(os.path.join(save_path, fName) , audio_data, Fs, format='wav')
                    break

    def ESC50(self, labels_path, root_dir, save_path, Fs):
        datasetName = 'ESC50'
        category_list = ['dog', 'rain', 'door_wood_knock', 'sneezing', 'mouse_click', 'clapping', 'keyboard_typing', 'siren', 'door_wood_creaks', 'car_horn', 'chirping_birds', 
                        'coughing', 'can_opening', 'engine', 'cat', 'water_drops', 'footsteps', 'washing_machine', 'train', 'laughing', 'vacuum_cleaner', 'pouring_water', 
                        'brushing_teeth', 'clock_alarm', 'airplane', 'toilet_flush', 'snoring', 'clock_tick', 'thunderstorm', 'drinking_sipping', 'glass_breaking']
        data_path = os.path.join(root_dir, 'ESC50/audio')
        meta_path = os.path.join(root_dir, 'ESC50/meta/esc50.csv')
        duration = 5.0
        
        src_list, dataset_list = self.open_labels(labels_path)
        labels_list = self.open_meta_data(meta_path)
        
        for labels in tqdm(labels_list):
            fName, category, src_file, take = labels[0], labels[3], labels[5], labels[6]
            
            if (category in category_list) and self.check_duplicates_src_file(src_list, dataset_list, src_file, datasetName):
                audio_data, Fs = librosa.load(os.path.join(data_path,fName), Fs)
                self.labels_to_csv(labels_path, fName, duration, category, src_file, take, datasetName)
                
                sf.write(os.path.join(save_path, fName) , audio_data, Fs, format='wav')

    def cutoff_data(self, data_path, save_path, Fs, sec, durationThreshold):
        
        for file in tqdm(os.listdir(data_path)):
            if not file.endswith('.wav'): continue

            file_path = os.path.join(data_path, file)
            audio_data, Fs = librosa.load(file_path, Fs)
            cutoff_data = audio_data[:(sec*Fs)]

            if librosa.get_duration(cutoff_data, Fs) == durationThreshold:
                save_file = os.path.join(save_path, file)
                sf.write(save_file, cutoff_data, Fs, format='wav')



if __name__ == '__main__':
    # =============== TTA Test Set ===============
    
    # =============== FSD50K ===============
    # data_path = '/Users/gimseong-yun/Downloads/FSD50K/FSD50K.dev_audio/'
    # dst = '/Users/gimseong-yun/Downloads/TTA_Dataset/2.speech/testset/noise/'
    # meta_path = '/Users/gimseong-yun/Downloads/FSD50K/FSD50K.ground_truth/dev.csv'
    # fname_list = label_classifier(data_path, dst, meta_path, 'FSD50K', copy=False)
    
    # =============== UrbanSound8K ===============
    # data_path = '/Users/gimseong-yun/Downloads/UrbanSound8K/audio/'
    # dst = '/Users/gimseong-yun/Downloads/TTA_Dataset/2.speech/testset/noise2/'
    # meta_path = '/Users/gimseong-yun/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv'
    # fname_list = label_classifier(data_path, dst, meta_path, 'UrbanSound8K', copy=False)
    
    # =============== ESC-50 ===============
    # data_path = '/Users/gimseong-yun/Downloads/ESC-50-master/audio/'
    # dst = '/Users/gimseong-yun/Downloads/TTA_Dataset/2.speech/testset/noise/'
    # meta_path = '/Users/gimseong-yun/Downloads/ESC-50-master/meta/esc50.csv'
    # fname_list = label_classifier(data_path, dst, meta_path, 'ESC-50', copy=False)



    # =============== Training Set ===============
    root_dir = '/home/ksy/OriginalData'
    category_list = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'engine_idling', 'siren']
    labels_path = '/home/ksy/Speech_Training/2.src/home_sound_labels.csv' 
    save_path = '/home/ksy/Speech_Training/0.raw/trainingset/home_sound'
    Fs = 16000
    durationThreshold = 5.0
    sec = 5
    
    Curator = TrainingSet()
    # Curator.UrbanSound8K(labels_path, root_dir, save_path, Fs, durationThreshold)
    # Curator.ESC50(labels_path, root_dir, save_path, Fs)
    # Curator.FSD50K(labels_path, root_dir, save_path, Fs, durationThreshold)
    
    # print(Curator.total_duration(labels_path))      # 12시간


    data_path = save_path
    save_path = '/home/ksy/Speech_Training/1.feature/home_sound'
    Curator.cutoff_data(data_path, save_path, Fs, sec, durationThreshold)


