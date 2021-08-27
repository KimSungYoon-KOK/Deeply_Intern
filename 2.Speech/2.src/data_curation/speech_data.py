import os
import json
import csv
from numpy import lib
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
from glob import glob
import shutil
import random


def speech_FSD50K():
    
    """
    Female_speech_and_woman_speaking
    Male_speech_and_man_speaking
        
    Male
    33474, 117127,121295, 124810,124812,124829,124831,124834,124838,124855,
    124857,124858,124873,124878,161267,170641,181316,243827,249170,330564,
    343613,343614,368174,371554,371582,400726,400742
    
    Female
    65687, 67331, 86385, 91076, 91081, 93703, 93704, 97346, 97349, 97350, 97351
    109845, 236493, 236494, 249893, 328699, 328712, 339268, 377738, 393504, 415343
    """
    
    # ===== eval dataset =====
    # data_path = '/Users/gimseong-yun/Downloads/FSD50K/FSD50K.eval_audio/'
    # dst = '/Users/gimseong-yun/Downloads/FSD50K/Test/'
    # meta_path = '/Users/gimseong-yun/Downloads/FSD50K/FSD50K.metadata/eval_clips_info_FSD50K.json'
    # ground_truth_file = pd.read_csv('/Users/gimseong-yun/Downloads/FSD50K/FSD50K.ground_truth/eval.csv')
    
    # ===== dev dataset =====
    data_path = '/Users/gimseong-yun/Downloads/FSD50K/FSD50K.dev_audio/'
    meta_path = '/Users/gimseong-yun/Downloads/FSD50K/FSD50K.metadata/dev_clips_info_FSD50K.json'
    dst = '/Users/gimseong-yun/Downloads/TTA_Dataset/2.speech/testset/speech'
    ground_truth_file = pd.read_csv('/Users/gimseong-yun/Downloads/FSD50K/FSD50K.ground_truth/dev.csv')
    
    df = pd.DataFrame(ground_truth_file)
    Fs = 16000
    
    # with open(meta_path) as json_file:
    #     meta_data = json.load(json_file)
    
    male_speech_list, female_speech_list = [], []
    for data in df[['fname', 'labels']].values:
        fname, labels = str(data[0]) + '.wav', data[1]
        
        if 'Male_speech_and_man_speaking,Speech,Human_voice' == labels:
            # audio_temp, Fs = librosa.load(path + fname, sr=Fs, mono=True)
            sr, wav_data = sio.wavfile.read(data_path+fname)
            duration = len(wav_data)/float(sr)
            # metaStr = str(meta_data[str(data[0])])
            if duration >= 1:
                male_speech_list.append(fname)
                # shutil.copy(data_path + fname, dst + 'Male')
            
        elif 'Female_speech_and_woman_speaking,Speech,Human_voice' == labels:
            # audio_temp, Fs = librosa.load(path + fname, sr=Fs, mono=True)
            sr, wav_data = sio.wavfile.read(data_path+fname)
            duration = len(wav_data)/float(sr)
            # metaStr = str(meta_data[str(data[0])])
            if duration >= 1:
                female_speech_list.append(fname)
                # shutil.copy(data_path + fname, dst + 'Female')
    
    print(len(male_speech_list))    # 58 + 189 = 247
    print(len(female_speech_list))  # 37 + 255 = 292
    
    
    data_path = '/Users/gimseong-yun/Downloads/TTA_Dataset/2.speech/testset/speech/Male/'
    save_path = '/Users/gimseong-yun/Downloads/TTA_Dataset/2.speech/testset/speech/Temp/'
    sec = 2
    
    f_list = [33474,117127,124810,124812,124829,124831,124834,124838,
              124855,124857,124858,124873,249170,330564,343613,343614]
    
    for file in os.listdir(data_path):
        if file.startswith('.'): continue  
        file_path = data_path + file
        
        audio_data, Fs = librosa.load(file_path, sr=Fs)
        trimmed_data, index = librosa.effects.trim(audio_data)
        
        duration = librosa.get_duration(trimmed_data, Fs)
        if duration >= (sec+1):
            cutoff_data = trimmed_data[(sec*Fs):((sec+1)*Fs)]
            sf.write(save_path + file, cutoff_data, Fs, format='wav')
    

class Kor_speech_Training():
    
    """
    Meta Data List
    [대화 정보]
        cityCode 지역 : "수도권"
        recrdEnvrn 녹음환경 : "실내"
    [녹음자 정보]
        recorderId 녹음자ID
        gender 성별 : 남 or 여 
        age 나이 : 17 ~ 60
    [발화 정보]
        fileNm 파일명 : "sample-1.wav"
        recrdTime 녹음시간 : "5.080"
        recrdQuality 녹음품질 : "16K"
        scriptSetNo 스크립트셋 번호 : 
    """
    
    def open_meta_data(self, meta_path):
        with open(meta_path) as meta_file:
            reader = csv.reader(meta_file)
            labels_list = list(reader)[1:]
    
        return labels_list

    def meta_data_check(self, meta_data, duration):
        if (meta_data['대화정보']['cityCode'] == '수도권' and meta_data['대화정보']['recrdEnvrn'] == '실내' and 
            17 < meta_data['녹음자정보']['age'] < 60 and
            meta_data['발화정보']['recrdQuality'] == '16K' and float(meta_data['발화정보']['recrdTime']) > duration):
            return True
        else:
            return False

    def labels_to_csv(self, labels_path, fileNm, category, recrdTime, gender, recorderId, folder):
        if not os.path.exists(labels_path):
            with open(labels_path, 'w', newline='') as labels_file:
                labels_file.write('fileNm,category,recrdTime,gender,recorderId,folder\n')
        
        with open(labels_path, 'a', newline='') as labels_file:
            wr = csv.writer(labels_file)
            wr.writerow([fileNm, category, recrdTime, gender, recorderId, folder])
    
    def json_to_csv(self, root_dir, labels_path, category_list, duration):
        meta_path = os.path.join(root_dir, '0.meta')
        labels_count = {'남':0, '여':0}

        for category in category_list:
            print(category)
            temp = os.path.join(meta_path, category)
            for folder in tqdm(os.listdir(temp)):
                if folder.startswith('.'): continue
                temp2 = os.path.join(temp, folder)

                for meta_file in os.listdir(temp2):
                    if meta_file.startswith('.'): continue

                    path = os.path.join(temp2, meta_file)
                    with open(path) as json_file:
                        meta_data = json.load(json_file)

                        if self.meta_data_check(meta_data, duration):
                            self.labels_to_csv(labels_path, 
                                            meta_data['발화정보']['fileNm'], category, meta_data['발화정보']['recrdTime'],
                                            meta_data['녹음자정보']['gender'], meta_data['녹음자정보']['recorderId'], folder)
                        
                            labels_count[meta_data['녹음자정보']['gender']] += 1
                     
        print(labels_count)     # {'남': 11389, '여': 45082}
        
    def cutoff_data(self, root_dir, labels_path, save_path, Fs, sec):
        gender_cnt = {'남' : 0, '여' : 0}
        labels_list = kor.open_meta_data(labels_path)
        new_labels_path = '/home/ksy/Speech_Training/0.raw/meta/kor_speech_labels1.csv'
        for labels in tqdm(labels_list):
            fName, category, recrdTime, gender, recorderId, folder = labels
            
            if gender_cnt[gender] < 11389:
                file_path = root_dir + '/' + category + '/' + folder + '/' + fName
                audio_data, Fs = librosa.load(file_path, Fs)
                cutoff_data = audio_data[:(sec*Fs)]

                save_file = os.path.join(save_path, fName)
                sf.write(save_file, cutoff_data, Fs, format='wav')
                self.labels_to_csv(new_labels_path, fName, category, recrdTime, gender, recorderId, folder)
                gender_cnt[gender] += 1
        
        print(gender_cnt)

if __name__ == '__main__':
    # ============= MetaData(.json) to CSV File =============
    # metaHanler = MetaDataHandler()
    # meta_path = '/Volumes/SSD 2TB #2/Speech_data_korean/자유대화 음성(일반남녀)/Training/label/'
    # save_path = '/Users/gimseong-yun/Desktop/training_labels.csv'
    # metaHanler.json_to_csv(meta_path, save_path)
    

    # === Training Set ===
    # root_dir = '/home/ksy/OriginalData/Kor_speech/Training'
    # labels_path = '/home/ksy/Speech_Training/0.raw/meta/kor_speech_labels.csv'
    # category_list = ['1.AI_chatbot', '2.voice_collection_tool', '3.studio']
    # duration = 5.0
    # kor = Kor_speech_Training()
    # kor.json_to_csv(root_dir, labels_path, category_list, duration)

    # save_path = '/home/ksy/Speech_Training/0.raw/speech/'
    # Fs = 16000
    # sec = 5
    # kor.cutoff_data(root_dir, labels_path, save_path, Fs, sec)



    # ============== 섞은 다음에 training set, test set 분리 ==============
    root_dir = '/home/ksy/Speech_Training/0.raw/'

    meta_path = root_dir + 'meta/kor_speech_labels.csv'
    test_meta_path = root_dir + 'meta/kor_speech_test_labels.csv'
    train_meta_path = root_dir + 'meta/kor_speech_train_labels.csv'

    # test_labels_list = []
    # train_labels_list = []
    # with open(meta_path) as meta_file:
    #     reader = csv.reader(meta_file)
    #     labels_list = list(reader)[1:]
    #     random.Random(123).shuffle(labels_list)
    #     labels_list = labels_list[:10000]
        
    # for idx, labels in enumerate(tqdm(labels_list)):
    #     recorderId = labels[4]
        
    #     if idx == 0:
    #         test_labels_list.append(labels)
    #         continue
    #     elif idx == 1:
    #         train_labels_list.append(labels)
    #         continue
        
    #     test_list = np.array(test_labels_list).T[4]
    #     train_list = np.array(train_labels_list).T[4]

    #     if recorderId not in train_list:
    #         if len(test_labels_list) < 500:
    #             test_labels_list.append(labels)
    #         else:
    #             if recorderId not in test_list:
    #                 train_labels_list.append(labels)
    #     else:
    #         train_labels_list.append(labels)

    
    # print(len(test_labels_list))
    # print(len(train_labels_list))
    
    # test_list = np.array(test_labels_list).T[4]
    # train_list = np.array(train_labels_list).T[4]

    # duplicated_list = list(set(test_list).intersection(train_list))
    # print(len(duplicated_list))

    # with open(train_meta_path, 'a', newline='') as labels_file:
    #     wr = csv.writer(labels_file)
        
    #     for labels in tqdm(train_labels_list):
    #         wr.writerow(labels)

    # with open(test_meta_path, 'a', newline='') as labels_file:
    #     wr = csv.writer(labels_file)
        
    #     for labels in tqdm(test_labels_list):
    #         wr.writerow(labels)



    # train_labels_list = []

    # with open(meta_path) as meta_file:
    #     reader = csv.reader(meta_file)
    #     labels_list = list(reader)[1:]
    #     random.Random(123).shuffle(labels_list)
    #     labels_list = labels_list[10000:]
    
    # with open(test_meta_path) as test_meta_file:
    #     reader = csv.reader(test_meta_file)
    #     test_labels_list = list(reader)[1:]
    #     test_list = np.array(test_labels_list).T[4]
            
    # for idx, labels in enumerate(tqdm(labels_list)):
    #     recorderId = labels[4]
        
    #     if recorderId not in test_list:
    #         train_labels_list.append(labels)
    
    # print(len(test_labels_list))
    # print(len(train_labels_list))

    # train_list = np.array(train_labels_list).T[4]

    # duplicated_list = list(set(train_list).intersection(test_list))
    # print(len(duplicated_list))
    

    # with open(train_meta_path, 'a', newline='') as labels_file:
    #     wr = csv.writer(labels_file)
        
    #     for labels in tqdm(train_labels_list):
    #         wr.writerow(labels)


    # ================ test, train labels 에 맞춰서 데이터 복사 ================
    # root_dir = '/home/ksy/Speech_Training/0.raw/'
    # category = 'home_sound/'
    # test_meta_path = root_dir + 'meta/home_sound_test_labels.csv'
    # train_meta_path = root_dir + 'meta/home_sound_train_labels.csv'
    # save_path = '/home/ksy/Speech_Training/1.feature/'
    # flag = ['test/', 'train/']


    # for idx, meta_path in enumerate([test_meta_path, train_meta_path]):

    #     with open(meta_path) as meta_file:
    #         reader = csv.reader(meta_file)
    #         labels_list = list(reader)[1:]
    #         if idx == 1:
    #             random.Random(123).shuffle(labels_list)
    #             labels_list = labels_list[:2160]


    #     for labels in tqdm(labels_list):
    #         fName = labels[0]
    #         load_file_path = root_dir + category + fName
    #         save_file_path = save_path + flag[idx] + category + fName
            
    #         shutil.copy(load_file_path, save_file_path)

    root_dir = '/home/ksy/Speech_Training/1.feature'
    cnt = 0
    for flag in glob(root_dir + '/*'):
        for category in glob(flag + '/*'):
            for path in glob(category + '/*'):
                audio_temp, Fs = librosa.load(path, 16000)
                if len(audio_temp) != Fs*5:
                    print(path)
                cnt += 1
    
    print(cnt)


