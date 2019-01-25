# This file is used to analyze over/underperform songs and understand their characteristics

# In[]
import pickle
import pandas as pd
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from unidecode import unidecode

# In[]
with open('./data/million_songs_data.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    millionData = pickle.load(f)      
    f.close() 

with open('./data/Youtube_all_data_df.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    Youtube_all_data_df = pickle.load(f)      
    f.close() 

Youtube_all_data_df = Youtube_all_data_df[Youtube_all_data_df['views']>0]
# In[]
extra_features = ['tempo', 'duration', 'artist_hotttnesss', 'song_hotttnesss', 'loudness', 'genre']

# In[]
# load over and under perform songs
with open('./data/over_under_perform_songs.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    over_under_perform_songs = pickle.load(f)      
    f.close()       

# In[]
millionData = millionData.fillna(0)
Youtube_all_data = Youtube_all_data_df.drop(columns='scraped', axis=1)

# In[]
# All Youtube songs

youtube_metadata= []

for i in range(0, Youtube_all_data.shape[0]):
    songData = Youtube_all_data.iloc[i]
    artist = songData.artist
    song = songData.title
    
    try:
        records = millionData[(millionData['artist_name']==artist.encode('ascii'))
            & (millionData['title']==song.encode('ascii'))]
        print (records[['duration', 'analysis_sample_rate', 'genre']])
        print('='*5)
        if records.shape[0] > 0:
            records = records[extra_features]
            #print (records)
            #print (records[extra_features].mean().values)
            temp = list(records.iloc[:,:records.shape[1]].mean().values)
            temp.append(records.iloc[0]['genre'])
            temp[5] = temp[5].decode('utf-8')
            
            youtube_metadata.append(temp)
        else:
            df = [0,0,0,0,0,'']
            #print (df.values[0])
            youtube_metadata.append(df)
    except:
        continue

# In[]
    
youtube_metadata_DF = pd.DataFrame(youtube_metadata, columns=extra_features)

# In[]
# write youtube_metadata_DF to file, then read it back later
pickle.dump( youtube_metadata_DF, open( "./data/youtube_metadata_DF.pickle", "wb" ) )

# In[]
# read back
with open('./data/youtube_metadata_DF.pickle', 'rb') as file:
    youtube_metadata_DF = pickle.load(file)
    file.close()
    
# In[]
# All Youtube songs with its popularity based on views
adjusted_coef = np.reshape([-0.05003947, -0.16015677], (1,2))
adjusted_intercept = 21.399992 - 0.3

def plot_all_youtube(year_range, peakPos_range, option=0):
    #year_range = [1970, 1975]
    #peakPos_range = [1,100]
    
    songs_in_rank = Youtube_all_data_df.copy()
    selected_samples = (songs_in_rank['peakPos'] >= peakPos_range[0]) \
                & (songs_in_rank['peakPos'] <= peakPos_range[1]) \
                & (songs_in_rank['year'] >= year_range[0]) \
                & (songs_in_rank['year'] <= year_range[1])
    
    peakPos = songs_in_rank['peakPos'][selected_samples]   
    years = 2018 - songs_in_rank['year'][selected_samples]     
    popularity = songs_in_rank['views'][selected_samples] 
     
    peakPos_numpy = np.reshape(peakPos.values,(-1,1))
    year_numpy = np.reshape(years.values,(-1,1))    
    popularity_numpy = np.reshape( popularity.values,(-1,1) ) 
        
    X_val = np.concatenate( (peakPos_numpy, year_numpy), axis = 1 )
    Y_val = np.log(popularity_numpy  + 0.00001) 
    Y_prediction = np.matmul(X_val, np.transpose(adjusted_coef)) + adjusted_intercept
    
    ratio_over = 3  # Defining threshold for over-perform songs
    ratio_under = 3  # Defining threshold for udnder-perform songs
    
    over_perform_condition = (Y_val>(Y_prediction*(1 + 1/ratio_over)))
    under_perform_condition = (Y_val<(Y_prediction*(1 - 1/ratio_under)))
    Y_over_perform = Y_val[over_perform_condition]
    Y_under_perform = Y_val[under_perform_condition]
    
    if option == 0:
        fig = plt.figure(figsize=(10,10))
        plt.scatter(peakPos_numpy,(Y_val),color='chartreuse',s=20, alpha=1, edgecolor='black', label="True Popularity")  
        plt.scatter(peakPos_numpy[over_perform_condition],(Y_over_perform),color='red',s=20, alpha=1, label="Overperform")  
        plt.scatter(peakPos_numpy[under_perform_condition],(Y_under_perform),color='blue',s=20, alpha=1, label="Underperform")  
        #plt.scatter(peakPos_numpy, (Y_prediction), color='red',s=20, alpha=0.5, label="Predicted Popularity")  
        plt.xlabel('peakPos')
        plt.ylabel('log popularity')
        
        plt.legend()
        plt.ylim([0,25])
    else:
        fig = plt.figure(figsize=(10,10))
        plt.scatter(2018 - year_numpy, (Y_val),color='chartreuse',s=20, alpha=1,edgecolor='black', label="True Popularity")  
        plt.scatter(2018 - year_numpy[over_perform_condition], (Y_over_perform),color='red',s=20, alpha=1, label="Overperform")  
        plt.scatter(2018 - year_numpy[under_perform_condition], (Y_under_perform),color='blue',s=20, alpha=1, label="Underperform")  
        plt.xlabel('year')
        plt.ylabel('log popularity')
        
        plt.legend()
        plt.ylim([0,25])


# plot
plot_all_youtube(year_range=[1970, 1975], peakPos_range=[1, 100], option=0)
plot_all_youtube(year_range=[1960, 2018], peakPos_range=[30, 50], option=1)

# In[]
 # inspext metadata of over and under performed songs
def plot_withMetadata(year_range, peakPos_range, option=0, feature_name='song_hotttnesss'):
    Youtube_all = Youtube_all_data.iloc[:Youtube_all_data.shape[0]-2,:].copy()
    songs_in_rank = Youtube_all.copy()
    selected_samples = (songs_in_rank['peakPos'] >= peakPos_range[0]) \
                & (songs_in_rank['peakPos'] <= peakPos_range[1]) \
                & (songs_in_rank['year'] >= year_range[0]) \
                & (songs_in_rank['year'] <= year_range[1])
    
    peakPos = songs_in_rank['peakPos'][selected_samples]   
    years = 2018 - songs_in_rank['year'][selected_samples]  
    popularity = songs_in_rank['views'][selected_samples] 
    peakPos_numpy = np.reshape(peakPos.values,(-1,1))
    year_numpy = np.reshape(years.values,(-1,1)) 
    popularity_numpy = np.reshape( popularity.values,(-1,1) )
    
    selected_samples = np.reshape(selected_samples.values, (-1,1))
    
    feature = np.reshape(youtube_metadata_DF[feature_name].values, (-1,1))[selected_samples]
    
    X_val = np.concatenate( (peakPos_numpy, year_numpy), axis = 1 )
    Y_val = np.log(popularity_numpy  + 0.00001) 
    Y_prediction = np.matmul(X_val, np.transpose(adjusted_coef)) + adjusted_intercept
    
    ratio_over = 3  # Defining threshold for over-perform songs
    ratio_under = 3  # Defining threshold for udnder-perform songs
    
    over_perform_condition = (Y_val>(Y_prediction*(1 + 1/ratio_over)))
    under_perform_condition = (Y_val<(Y_prediction*(1 - 1/ratio_under)))
    
    feature_over_perform = np.reshape(feature, (-1,1))[over_perform_condition]
    feature_under_perform = np.reshape(feature, (-1,1))[under_perform_condition]
    
    if option == 0:
        if (feature_name != 'loudness'):
            fig = plt.figure(figsize=(10,10))
            plt.scatter(peakPos_numpy, np.log(feature), color='chartreuse', s=20, edgecolor='black', label='Normal')  
            plt.scatter(peakPos_numpy[over_perform_condition],
                        np.log(feature_over_perform), color='red', 
                        s=30, alpha=1, label="Overperform")  
            plt.scatter(peakPos_numpy[under_perform_condition], 
                        np.log(feature_under_perform), color='blue',
                        s=30, alpha=1, label="Underperform")  
            plt.xlabel('peakPos', fontsize=15)
            plt.ylabel(feature_name, fontsize=15)
            plt.title(feature_name, fontsize=15)
            plt.legend()
        else:
            fig = plt.figure(figsize=(10,10))
            plt.scatter(peakPos_numpy, (feature), color='chartreuse', s=20, edgecolor='black', label='Normal')  
            plt.scatter(peakPos_numpy[over_perform_condition],
                        (feature_over_perform), color='red', 
                        s=30, alpha=1, label="Overperform")  
            plt.scatter(peakPos_numpy[under_perform_condition], 
                        (feature_under_perform), color='blue',
                        s=30, alpha=1, label="Underperform")  
            plt.xlabel('peakPos', fontsize=15)
            plt.ylabel(feature_name, fontsize=15)
            plt.title(feature_name, fontsize=15)
            plt.legend()
    else:
        if (feature_name != 'loudness'):
            fig = plt.figure(figsize=(10,10))
            plt.scatter(2018 - year_numpy, np.log(feature), color='chartreuse', s=20, edgecolor='black', label='Normal')  
            plt.scatter(2018 - year_numpy[over_perform_condition],
                        np.log(feature_over_perform), color='red', 
                        s=30, alpha=1, label="Overperform")  
            plt.scatter(2018 - year_numpy[under_perform_condition], 
                        np.log(feature_under_perform), color='blue',
                        s=30, alpha=1, label="Underperform")  
            plt.xlabel('year', fontsize=15)
            plt.ylabel(feature_name, fontsize=15)
            plt.title(feature_name, fontsize=15)
            plt.legend()
        else:
            fig = plt.figure(figsize=(10,10))
            plt.scatter(2018 - year_numpy, (feature), color='chartreuse', s=20, edgecolor='black', label='Normal')  
            plt.scatter(2018 - year_numpy[over_perform_condition],
                        (feature_over_perform), color='red', 
                        s=30, alpha=1, label="Overperform")  
            plt.scatter(2018 - year_numpy[under_perform_condition], 
                        (feature_under_perform), color='blue',
                        s=30, alpha=1, label="Underperform")  
            plt.xlabel('year', fontsize=15)
            plt.ylabel(feature_name, fontsize=15)
            plt.title(feature_name, fontsize=15)
            plt.legend()

for feature in extra_features:
    if (feature != 'genre'):
        plot_withMetadata(year_range=[1970, 1975], peakPos_range=[1, 100], option=0, feature_name=feature)
        plot_withMetadata(year_range=[1960, 2018], peakPos_range=[30, 50], option=1, feature_name=feature)
        
        
# In[]
        
def plot_withMetadata_two_features(year_range, peakPos_range, feature1_name='song_hotttnesss', feature2_name='artist_hotttnesss'):
    Youtube_all = Youtube_all_data.iloc[:Youtube_all_data.shape[0]-2,:].copy()
    songs_in_rank = Youtube_all.copy()
    selected_samples = (songs_in_rank['peakPos'] >= peakPos_range[0]) \
                & (songs_in_rank['peakPos'] <= peakPos_range[1]) \
                & (songs_in_rank['year'] >= year_range[0]) \
                & (songs_in_rank['year'] <= year_range[1])
    
    peakPos = songs_in_rank['peakPos'][selected_samples]   
    years = 2018 - songs_in_rank['year'][selected_samples]  
    popularity = songs_in_rank['views'][selected_samples] 
    peakPos_numpy = np.reshape(peakPos.values,(-1,1))
    year_numpy = np.reshape(years.values,(-1,1)) 
    popularity_numpy = np.reshape( popularity.values,(-1,1) )
    
    selected_samples = np.reshape(selected_samples.values, (-1,1))
    
    feature1 = np.reshape(youtube_metadata_DF[feature1_name].values, (-1,1))[selected_samples]
    feature2 = np.reshape(youtube_metadata_DF[feature2_name].values, (-1,1))[selected_samples]
    
    X_val = np.concatenate( (peakPos_numpy, year_numpy), axis = 1 )
    Y_val = np.log(popularity_numpy  + 0.00001) 
    Y_prediction = np.matmul(X_val, np.transpose(adjusted_coef)) + adjusted_intercept
    
    ratio_over = 3 # Defining threshold for over-perform songs
    ratio_under = 3  # Defining threshold for udnder-perform songs
    
    over_perform_condition = (Y_val>(Y_prediction*(1 + 1/ratio_over)))
    under_perform_condition = (Y_val<(Y_prediction*(1 - 1/ratio_under)))
    
    feature1_over_perform = np.reshape(feature1, (-1,1))[over_perform_condition]
    feature2_over_perform = np.reshape(feature2, (-1,1))[over_perform_condition]
    feature1_under_perform = np.reshape(feature1, (-1,1))[under_perform_condition]
    feature2_under_perform = np.reshape(feature2, (-1,1))[under_perform_condition]
    
    fig = plt.figure(figsize=(10,10))
    plt.scatter(np.log(feature1), np.log(feature2), color='chartreuse', s=20, edgecolor='black', label='Regular')  
    plt.scatter(np.log(feature1_over_perform),
                np.log(feature2_over_perform), color='red', 
                s=30, alpha=1, label="Overperform")  
    plt.scatter(np.log(feature1_under_perform), 
                np.log(feature2_under_perform), color='blue',
                s=30, alpha=1, label="Underperform")  
    plt.xlabel(feature1_name, fontsize=15)
    plt.ylabel(feature2_name, fontsize=15)
    plt.title(feature1_name + ' vs ' + feature2_name, fontsize=15)
    plt.legend()


plot_withMetadata_two_features(year_range=[1960, 2018], peakPos_range=[30, 50], feature1_name='song_hotttnesss', feature2_name='tempo')    
plot_withMetadata_two_features(year_range=[1960, 2018], peakPos_range=[30, 50], feature1_name='song_hotttnesss', feature2_name='duration')    
plot_withMetadata_two_features(year_range=[1960, 2018], peakPos_range=[30, 50], feature1_name='artist_hotttnesss', feature2_name='tempo')    
plot_withMetadata_two_features(year_range=[1960, 2018], peakPos_range=[30, 50], feature1_name='artist_hotttnesss', feature2_name='duration') 
plot_withMetadata_two_features(year_range=[1960, 2018], peakPos_range=[30, 50], feature1_name='duration', feature2_name='tempo')  
plot_withMetadata_two_features(year_range=[1960, 1970], peakPos_range=[1, 100], feature1_name='song_hotttnesss', feature2_name='artist_hotttnesss')  
   

        
# In[]
fig = plt.figure(figsize=(10,10))
duration = np.reshape(youtube_metadata_DF['duration'].values, (-1,1))
artist_hotttnesss = np.reshape(youtube_metadata_DF['artist_hotttnesss'].values, (-1,1))
plt.scatter(np.log(duration), np.log(artist_hotttnesss), color='blue', s=20, alpha=0.2)  
#sns.scatterplot(x=np.log(youtube_metadata_DF['duration']), y=np.log(youtube_metadata_DF['artist_hotttnesss']), hue="genre",\
#                alpha=0.5, data=youtube_metadata_DF)
plt.xlabel('duration')
plt.ylabel('artist_hotttnesss')

fig = plt.figure(figsize=(10,10))
duration = np.reshape(youtube_metadata_DF['duration'].values, (-1,1))
song_hotttnesss = np.reshape(youtube_metadata_DF['song_hotttnesss'].values, (-1,1))
plt.scatter(np.log(duration), np.log(song_hotttnesss), color='blue', s=20, alpha=0.2)  
plt.xlabel('duration')
plt.ylabel('song_hotttnesss')

fig = plt.figure(figsize=(10,10))
tempo = np.reshape(youtube_metadata_DF['tempo'].values, (-1,1))
song_hotttnesss = np.reshape(youtube_metadata_DF['song_hotttnesss'].values, (-1,1))
plt.scatter(np.log(tempo), np.log(song_hotttnesss), color='blue', s=20, alpha=0.2)  
plt.xlabel('tempo')
plt.ylabel('song_hotttnesss')

fig = plt.figure(figsize=(10,10))
tempo = np.reshape(youtube_metadata_DF['tempo'].values, (-1,1))
artist_hotttnesss = np.reshape(youtube_metadata_DF['artist_hotttnesss'].values, (-1,1))
plt.scatter(np.log(tempo), np.log(artist_hotttnesss), color='blue', s=20, alpha=0.2)  
plt.xlabel('tempo')
plt.ylabel('artist_hotttnesss')

fig = plt.figure(figsize=(10,10))
song_hotttnesss = np.reshape(youtube_metadata_DF['song_hotttnesss'].values, (-1,1))
artist_hotttnesss = np.reshape(youtube_metadata_DF['artist_hotttnesss'].values, (-1,1))
plt.scatter(np.log(song_hotttnesss), np.log(artist_hotttnesss), color='blue', s=20, alpha=0.2)  
plt.xlabel('song_hotttnesss')
plt.ylabel('artist_hotttnesss')


# In[]

plt.figure(figsize=(10,20))
song_hot_genre = youtube_metadata_DF.groupby('genre')['song_hotttnesss'].mean()
song_hot_genre = song_hot_genre.to_frame().reset_index()
song_hot_genre.sort_values(by=['song_hotttnesss'], ascending=False, inplace=True)
sns.barplot(x=song_hot_genre['song_hotttnesss'], y=song_hot_genre['genre'], data=song_hot_genre, orient="h")


plt.figure(figsize=(10,20))
artist_hot_genre = youtube_metadata_DF.groupby('genre')['artist_hotttnesss'].mean()
artist_hot_genre = artist_hot_genre.to_frame().reset_index()
artist_hot_genre.sort_values(by=['artist_hotttnesss'], ascending=False, inplace=True)
sns.barplot(x=artist_hot_genre['artist_hotttnesss'], y=artist_hot_genre['genre'], data=artist_hot_genre, orient="h")

# In[]
over_songs = over_under_perform_songs[0]
under_songs = over_under_perform_songs[1]

# In[]

def getMetadata_under_over(_data, _type='over'):
    over_under_songs_metadata= []
    
    for i in range(0, _data.shape[0]):
        songData = _data.iloc[i]
        artist = songData.artist
        song = songData.title
        
        try:
            records = millionData[(millionData['artist_name']==artist.encode('ascii'))
                & (millionData['title']==song.encode('ascii'))]
            
            if records.shape[0] > 0:
                records = records[extra_features]
                #print (records)
                #print (records[extra_features].mean().values)
                temp = list(records.iloc[:,:records.shape[1]].mean().values)
                temp.append(records.iloc[0]['genre'])
                temp[5] = temp[5].decode('utf-8')
                if (temp[5] != ''):
                    print(temp[5])
                    print (temp)
                over_under_songs_metadata.append(temp)
            else:
                data = [0,0,0,0,0,'']
                #print (df.values[0])
                over_under_songs_metadata.append(data)
        except:
            continue
        
    over_under_songs_metadata = pd.DataFrame(over_under_songs_metadata, columns=extra_features)
    if _type == 'over':
        pickle.dump( over_under_songs_metadata, open( "./data/over_songs_metadata.pickle", "wb" ) )
    else:
        pickle.dump( over_under_songs_metadata, open( "./data/under_songs_metadata.pickle", "wb" ) )

        
getMetadata_under_over(over_songs, _type='over')
getMetadata_under_over(under_songs, _type='under')


# In[]
with open('./data/over_songs_metadata.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    over_songs_metadata = pickle.load(f)      
    f.close() 
    
with open('./data/under_songs_metadata.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    under_songs_metadata = pickle.load(f)      
    f.close() 

# In[]
def million_song(filepath):
    f = h5py.File(filepath, 'r')
    
    dataConcat = []
    a_group_key = list(f.keys())
    print (a_group_key)
    for key in a_group_key:
        print ('key: ', key)
        data = f[key]['songs'].value
        df = pd.DataFrame(data)
        dataConcat.append(df)
        print('='*40)
    
    return pd.concat(dataConcat, axis=1)

millionData = million_song('./data/msd.h5')
    
# In[]

fig = plt.figure(figsize=(10,10))
song_hotttnesss = np.reshape(over_songs_metadata['song_hotttnesss'].values, (-1,1))
artist_hotttnesss = np.reshape(over_songs_metadata['artist_hotttnesss'].values, (-1,1))
#plt.scatter(np.log(song_hotttnesss), np.log(artist_hotttnesss), color='green', s=20)  
sns.scatterplot(x=np.log(over_songs_metadata['song_hotttnesss']), 
                y=np.log(over_songs_metadata['artist_hotttnesss']), 
                hue='genre', data=over_songs_metadata)
plt.xlabel('song_hotttnesss')
plt.ylabel('artist_hotttnesss')

fig = plt.figure(figsize=(10,10))
song_hotttnesss = np.reshape(under_songs_metadata['song_hotttnesss'].values, (-1,1))
artist_hotttnesss = np.reshape(under_songs_metadata['artist_hotttnesss'].values, (-1,1))
sns.scatterplot(x=np.log(under_songs_metadata['song_hotttnesss']), 
                y=np.log(under_songs_metadata['artist_hotttnesss']), 
                hue='genre', data=under_songs_metadata)
plt.xlabel('song_hotttnesss')
plt.ylabel('artist_hotttnesss')

# In[]

# top over performed songs
top_over_songs = over_songs.sort_values(by='predicted_true_ratio', ascending=False).head(50)

youtube_metadata= []

for i in range(0, top_over_songs.shape[0]):
    songData = top_over_songs.iloc[i]
    artist = songData.artist
    song = songData.title
    print ('song: ', song)
    print ('artist: ', artist)
    try:
        records = millionData[(millionData['artist_name']==artist.encode('ascii'))
            & (millionData['title']==song.encode('ascii'))]
        
        print (records[[ 'genre', 'release', 'year']])
        
        if records.shape[0] > 0:
            records = records[extra_features]
            #print (records)
            #print (records[extra_features].mean().values)
            temp = list(records.iloc[:,:records.shape[1]].mean().values)
            temp.append(records.iloc[0]['genre'])
            temp[5] = temp[5].decode('utf-8')
            
            youtube_metadata.append(temp)
        print ('=' * 10)
    except:
        continue   

# In[]
# top under performed songs
top_under_songs = under_songs.sort_values(by='predicted_true_ratio', ascending=True).head(200)

youtube_metadata= []

for i in range(0, top_under_songs.shape[0]):
    songData = top_under_songs.iloc[i]
    artist = songData.artist
    song = songData.title
    
    try:
        records = millionData[(millionData['artist_name']==artist.encode('ascii'))
            & (millionData['title']==song.encode('ascii'))]
        #print (records[['duration', 'analysis_sample_rate', 'genre']])
        #print('='*5)
        if records.shape[0] > 0:
            records = records[extra_features]
            #print (records)
            #print (records[extra_features].mean().values)
            temp = list(records.iloc[:,:records.shape[1]].mean().values)
            temp.append(records.iloc[0]['genre'])
            temp[5] = temp[5].decode('utf-8')
            
            youtube_metadata.append(temp)
    except:
        continue
