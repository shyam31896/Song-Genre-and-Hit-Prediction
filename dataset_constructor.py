### Dataset creation  -  Genre Prediction Dataset ###

''' 

To create the dataset for this project, a custom code is written to extract audio from Youtube using the tool `youtube-dl` 
and data collected for the 7 specific music genres from the `AudioSet` released by Google.

source: https://github.com/HareeshBahuleyan/music-genre-classification/blob/master/1_audio_retrieval.ipynb

'''

import os
import re
import youtube_dl
from tqdm import tqdm
import numpy as np
import pandas as pd
from audio_utils import pre_emphasis, MFCC, Zero_crossing_rate, Spectral_centroid, Spectral_rolloff, Chroma_feat

WAV_DIR = 'wav_files/'
genre_dict = {'/m/064t9': 'Pop_music',
		     '/m/0glt670': 'Hip_hop_music',
                     '/m/06by7': 'Rock_music',
                     '/m/06j6l': 'Rhythm_blues',
                     '/m/06cqb': 'Reggae', 
                     '/m/0y4f8': 'Vocal',
                     '/m/07gxw': 'Techno'}

genre_set = set(genre_dict.keys())
temp_str = []
os.system('tar -xvf data-info.tar.gz | grep data-files')
with open('data-files/unbalanced_train_segments.csv', 'r') as f:
    temp_str = f.readlines()
data = np.ones(shape=(1,4)) 

print('Downloading audio files:')

for line in tqdm(temp_str):
    line = re.sub('\s?"', '', line.strip())
    elements = line.split(',')
    common_elements = list(genre_set.intersection(elements[3:]))
    if  common_elements != []:
        data = np.vstack([data, np.array(elements[:3] + [genre_dict[common_elements[0]]]).reshape(1, 4)])

df = pd.DataFrame(data[1:], columns=['url', 'start_time', 'end_time', 'class_label'])

# Remove 10k Techno audio clips - to make the data more balanced

np.random.seed(10)
drop_indices = np.random.choice(df[df['class_label'] == 'Techno'].index, size=10000, replace=False)
df.drop(labels=drop_indices, axis=0, inplace=True)
df.reset_index(drop=True, inplace=False)
df['start_time'] = df['start_time'].map(lambda x: np.int32(np.float(x)))
df['end_time'] = df['end_time'].map(lambda x: np.int32(np.float(x)))

for i, row in tqdm(df.iterrows()):
    url = "'https://www.youtube.com/embed/" + row['url'] + "'"
    file_name = str(i)+"_"+row['class_label']
    try:
        command_1 = "ffmpeg -ss "+str(row['start_time'])+" -i $(youtube-dl -f 140 --get-url "+url+") -t 10 -c:v copy -c:a copy "+file_name+".mp4"
        command_2 = "ffmpeg -i "+file_name+".mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 "+WAV_DIR+file_name+".wav"
        command_3 = 'rm '+file_name+'.mp4' 
        os.system(command_1 + ';' + command_2 + ';' + command_3 + ';')
    except:
        print(i, url)
        pass

print('Download complete')
### Feature extraction and building dataset from downloaded audio files ###

cols = ['file_name'] + ['signal_mean'] + ['signal_std'] +\
       ['mfcc_' + str(i+1) + '_mean' for i in range(20)] + ['mfcc_' + str(i+1) + '_std' for i in range(20)] + \
       ['zero_crossing_mean','zero_crossing_std','spec_centroid_mean','spec_centroid_std', \
        'spec_rolloff_mean','spec_rolloff_std'] + \
       ['chroma_' + str(i+1) + '_mean' for i in range(12)] + ['chroma_' + str(i+1) + '_std' for i in range(12)] +\
       ['label']
labels = {'Hip':0,'Pop':1,'Vocal':2,'Rhythm':3,'Reggae':4,'Rock':5,'Techno':6}

print('Feature extraction started')
dataset = pd.DataFrame(columns=cols)
for file in tqdm(os.listdir('wav_files')):
    signal, sample_rate = librosa.load('wav_files/'+file, sr = 22050)
    pre_emphasized_signal = pre_emphasis(signal)
    signal_mean = np.mean(abs(pre_emphasized_signal))
    signal_std = np.std(pre_emphasized_signal)
    mel_scaled_out = MFCC(pre_emphasized_signal)
    zero_crossing = Zero_crossing_rate(pre_emphasized_signal)
    spec_centroid = Spectral_centroid(pre_emphasized_signal)
    spec_rolloff = Spectral_rolloff(pre_emphasized_signal)
    chroma = Chroma_feat(pre_emphasized_signal)
    res_list = []
    res_list.append(file)
    res_list.append(signal_mean)
    res_list.append(signal_std)
    res_list.extend(np.mean(mel_scaled_out, axis = 1))
    res_list.extend(np.std(mel_scaled_out, axis = 1))
    res_list.extend((np.mean(zero_crossing), np.std(zero_crossing), np.mean(spec_centroid), np.std(spec_centroid)))
    res_list.extend((np.mean(spec_rolloff), np.std(spec_rolloff)))
    res_list.extend(np.mean(chroma, axis = 1))
    res_list.extend(np.std(chroma, axis = 1))
    res_list.extend(str(labels.get(file.replace('.','_').split('_')[1])))
    dataset = dataset.append(pd.DataFrame(res_list, index = cols).T, ignore_index = True)
dataset.to_csv("dataset_genre_pred.csv", index = False)

dataset_genre = pd.read_csv('dataset_genre_pred.csv')
dataset_genre['label'] = pd.to_numeric(dataset_genre['label'])
data_train_genre, data_sec_genre = train_test_split(dataset_genre.drop('file_name', axis = 1), test_size = 0.2, random_state=5)
data_val_genre, data_test_genre = train_test_split(data_sec_genre, test_size = 0.4, random_state=5)
scaler = MinMaxScaler()
data_train_genre[data_train_genre.columns[1:len(data_train_genre.columns)-1]] = scaler.fit_transform(data_train_genre[data_train_genre.columns[1:len(data_train_genre.columns)-1]])
data_val_genre[data_val_genre.columns[1:len(data_val_genre.columns)-1]] = scaler.transform(data_val_genre[data_val_genre.columns[1:len(data_val_genre.columns)-1]])
data_test_genre[data_test_genre.columns[1:len(data_test_genre.columns)-1]] = scaler.transform(data_test_genre[data_test_genre.columns[1:len(data_test_genre.columns)-1]])
os.system('rm -rf data/')
os.system('mkdir data/')
data_train_genre.to_csv('data/data_genre_training.csv', index = False)
data_val_genre.to_csv('data/data_genre_validation.csv', index = False)
data_test_genre.to_csv('data/data_genre_test.csv', index = False)
print('Genre Dataset successfully constructed')

####### Construct dataset for Song Hit Prediction #######

import billboard
import datetime 
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import warnings
warnings.filterwarnings("ignore")

### Collect Songs from Billboard for the given time range of 20 years ###

'''

To collect songs from using the Billboard API, query data for a maximum duration of two years as the API stops responding 
if the number of requests made by the function is too large

'''
print('Billboard audio extraction starts')
num_years = 2
for year in ["%.2d" % i for i in range(19-num_years+1, 19)]:
    prev_date_list = []
    date1 = '20'+year+'-01-01'
    date2 = '20'+str(int(year)+1)+'-11-30'
    start = datetime.datetime.strptime(date1, '%Y-%m-%d')
    end = datetime.datetime.strptime(date2, '%Y-%m-%d')
    step = datetime.timedelta(days=60)
    while start <= end:
        prev_date_list.append(str(start.date()))
        start += step
    print(prev_date_list)

    cols = ['Artist','Track','Label']
    billboard_df = pd.DataFrame()
    chart = billboard.ChartData('hot-100', prev_date_list[0])
    for i in range(1, len(prev_date_list)):
        for ind in range(1, len(chart))[:30]:
            song = chart[ind]
            if i != 1 and song.title in billboard_df[1]:
                pass
            else:
                entry = []
                entry.extend((song.artist, song.title, str(1)))
                billboard_df = billboard_df.append(pd.DataFrame(entry).T)
        chart = billboard.ChartData('hot-100', prev_date_list[i])
        time.sleep(1)
billboard_df.to_csv("billboard_data.csv", index=False)
print('Billboard audio extraction complete')

### Collect Songs which did not make it to the Billboard for the given time range of 20 years ###

print('Non-Billboard audio extraction starts')
billboard_df = pd.read_csv("billboard_data.csv", names=cols).iloc[1:,:]
chart = billboard.ChartData('radio-songs', prev_date_list[0])
for i in range(1, len(prev_date_list)):
    for ind in range(1, len(chart))[:30]:
        song = chart[ind]
        if i != 1 and song.title in billboard_df.iloc[:,1]:
            pass
        else:
            entry = []
            entry.extend((song.artist, song.title, str(0)))
            billboard_df = billboard_df.append(pd.DataFrame(entry, index=cols).T)
    chart = billboard.ChartData('radio-songs', prev_date_list[i])
    time.sleep(2)
billboard_df.to_csv("billboard_data.csv", index=False)
print('Non-Billboard audio extraction complete')

### Extract Song features from Spotify and construct the dataset ###

billboard_df = pd.read_csv("billboard_data.csv")
client_credentials_manager = SpotifyClientCredentials(client_id="769ef3519e8444238fde9c8981c6371c",\
                                                      client_secret="b17e4a7ca0b4426f9962645ba5c74a63")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
features_df = pd.DataFrame()
time_df = pd.DataFrame()
release_feat = ['Year','Month']
spotify_feat = ['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']

for ind in range(len(billboard_df.iloc[:,0:2])):
    artist, track = billboard_df.iloc[ind,0:2]
    songs=sp.search(q='track:'+track+' '+'artist:'+artist+'*' , type='track')
    items = songs['tracks']['items']
    features_to_df = []

    if len(items) == 0:
        features_df = features_df.append(pd.Series(['None']*18), ignore_index = True)
        time_df = time_df.append(pd.Series(['None']*2), ignore_index = True)

    else:
        track = items[0]
        song_id = str(track["id"])
        track_features=sp.audio_features(song_id)
        if int(track['album']['release_date'].split('-')[0]) < 2000: 
            y = 'None'
            m = 'None'
        else:
            y = track['album']['release_date'].split('-')[0]
            m = track['album']['release_date'].split('-')[1]
        rel = [y,m]
        time_df = time_df.append(pd.DataFrame(rel).T)
        features_to_df = [val for val in (track_features)[0].values()]
        features_df = features_df.append(pd.DataFrame(features_to_df).T)

features_df = features_df.drop([11, 12, 13, 14, 15, 16, 17], axis=1)
features_df.columns = spotify_feat
time_df.columns = release_feat
output = pd.concat([billboard_df.iloc[:-1,:],features_df.iloc[:-1,:],time_df.iloc[:-1,:]],axis=1)
output.to_csv("billboard_data_with_spotify.csv", index = False)
dataset = pd.read_csv('billboard_data_with_spotify.csv').drop(['Artist','Track'], axis = 1)
colnames = list(dataset.columns)
dataset_no_label = dataset.drop(['Label'], axis = 1)
dataset['Label'] = pd.to_numeric(dataset['Label'])
data_train, data_sec = train_test_split(dataset, test_size = 0.1, random_state=5)
data_val, data_test = train_test_split(data_sec, test_size = 0.5, random_state=5)
scaler = MinMaxScaler()
data_train[data_train.columns[:2]] = scaler.fit_transform(data_train[data_train.columns[:2]])
data_val[data_val.columns[:2]] = scaler.transform(data_val[data_val.columns[:2]])
data_test[data_test.columns[:2]] = scaler.transform(data_test[data_test.columns[:2]])
data_train[data_train.columns[3:4]] = scaler.fit_transform(data_train[data_train.columns[3:4]])
data_val[data_val.columns[3:4]] = scaler.transform(data_val[data_val.columns[3:4]])
data_test[data_test.columns[3:4]] = scaler.transform(data_test[data_test.columns[3:4]])
data_train[data_train.columns[5:len(data_train.columns)-3]] = scaler.fit_transform(data_train[data_train.columns[5:len(data_train.columns)-3]])
data_val[data_val.columns[5:len(data_val.columns)-3]] = scaler.transform(data_val[data_val.columns[5:len(data_val.columns)-3]])
data_test[data_test.columns[5:len(data_test.columns)-3]] = scaler.transform(data_test[data_test.columns[5:len(data_test.columns)-3]])
data_train.to_csv('data/data_hit_training.csv', index = False)
data_val.to_csv('data/data_hit_validation.csv', index = False)
data_test.to_csv('data/data_hit_test.csv', index = False)

### Visualizing the Datapoints - Hit Predictor ###

color = []
for i in dataset['Label']:
    if i == 1:
        color.append('blue')
    else:
        color.append('red')
# Used Pandas to plot the scatter plot of the independent variables
pd.plotting.scatter_matrix(dataset_no_label,figsize=(15,15),marker='.',c=color,alpha=0.5,s=50)
plt.subplots_adjust(top=0.95)
plt.suptitle('Fig.1: Scatterplot of Independent Variables', fontsize=16)
# Extra commands to display legend
c0, = plt.plot([1,1],'r.')
c1, = plt.plot([1,1],'b.')
plt.legend((c0, c1),('Normal Song', 'Hit Song'),loc=(-0.5,13.1))
c0.set_visible(False)
c1.set_visible(False)
plt.savefig('results/conf_matrices/hit_pred_scatter_matrix.jpg')

print('Audio features extracted and Hit Prediction Dataset Construction complete')
