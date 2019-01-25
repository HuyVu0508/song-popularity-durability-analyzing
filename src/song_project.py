# This is used to learn about the project, testing ideas

import song_functions
import pickle
import matplotlib.pyplot as plt
import numpy as np
from popularity_functions import popularity_function_guessing

###################
# Get list of all top songs over time
all_top_songs_list = song_functions.BillBoard_chart_all_time('./data/chart_history.pickle')
all_top_songs_list_continuos = song_functions.BillBoard_chart_all_time('./data/chart_history.pickle')
with open('./data/all_top_songs_list.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(all_top_songs_list_continuos, f)      
    f.close()  
    

    
###################    
# Explore songs'popularity at the same released time
start_year = 1960
end_year = 1960
for i in range(len(all_top_songs_list)):
    song_info = all_top_songs_list[i]
    # Only considering songs released in time interval
    if (song_info['year'] < start_year) | (song_info['year'] > end_year):
        continue
    
    # Getting song infor in Blackboard and Youtube
    print(song_info)
    song_stats = song_functions.lookup_Youtube(song_info['title'], song_info['artist'])   
    print(song_stats["view_count"])
    
    # Plots
    # Not implemented yet



###################
# Explore artist
artist_name = "Katy Perry"
for i in range(len(all_top_songs_list)):
    song_info = all_top_songs_list[i]
    # Only considering songs released in time interval
    if (song_info['artist'] != artist_name):
        continue
    
    # Getting song infor in Blackboard and Youtube
    print(song_info)    
    song_stats = song_functions.lookup_Youtube(song_info['title'], song_info['artist'])   
    print(song_stats["view_count"])


    # Plots
    # Not implemented yet


###################
# Testing functions
lnk = "https://www.youtube.com/watch?v=JF8BRvqGCNs"    
video_data = song_functions.get_Youtube_video_data(lnk)    

song_functions.lookup_Youtube("Shape Of You", "Ed Sheeran")
BillBoard_dict = song_functions.lookup_BillBoard("Shape Of You", "Ed Sheeran")
BillBoard_Youtube_dict = song_functions.lookup_BillBoard_Youtube("Shape Of You", "Ed Sheeran")
chart_all_time = song_functions.BillBoard_chart_all_time('./data/chart_history.pickle')


###################
# Scraping views (popularity now of songs) for all top songs
# Calculate Youtube views of all songs from all_time_top_list
with open('./data/all_top_songs_list.pickle','rb') as f: 
    all_top_songs_list = pickle.load(f)    
    
 
Youtube_view_all_songs = []
Error_songs = []
#for i in range(len(all_top_songs_list)):
import time   
t0 = time.clock()
start = 0
end = 51
for i in range(start, end):    
    if i%50==0:
        print(i)
        print(time.clock()-t0)
        t0 = time.clock()
    try:
        song_info = all_top_songs_list[i]
        song_stats = song_functions.lookup_Youtube(song_info['title'], song_info['artist'])   
        song_info_stats = {}
        song_info_stats.update(song_info)
        song_info_stats.update(song_stats)
        Youtube_view_all_songs.append(song_info_stats)
    except:
        Error_songs.append(i)
        pass
    
    if (i%1000==0) & (i>0):
        print("Saving song number:" + str(i) )
        with open('./data/Youtube Data/Youtube_view_all_songs_'+ str(i) +'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(Youtube_view_all_songs, f)      
            f.close()        
    
with open('./data/Youtube Data/Youtube_view_all_songs_'+ str(start) +'_'+ str(end-1) +'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Youtube_view_all_songs, f)      
    f.close()      




###################
# Get metadata information of songs
# Metadata
import h5py
filename = 'Raw Dataset/msd_summary_file.h5'
f = h5py.File(filename, 'r')
#Get the HDF5 group
group = f['analysis']
group = f['metadata']
group = f['musicbrainz']
#Checkout what keys are inside that group.
for key in group.keys():
    print(key)
data = group['songs'].value
data

song_stats = song_functions.lookup_Youtube('The Village Of St. Bernadette', 'Andy Williams')   


####################
# Get list of all top songs over time
with open('./data/all_top_songs_list.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)      


####################
# Song popularity on Lastfm
with open('./data/Lastfm Data/Lastfm_all_data.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    Lastfm_all_data = pickle.load(f)      
    f.close()          


####################
# Incorporating popularity to songs infor data      
for i in range(len(all_top_songs_list)):    
    
    if i%1000==0:
        print(i)
    
    song_infor = all_top_songs_list[i]
    for j in range(len(Lastfm_all_data)):
        song_view = Lastfm_all_data[j]
        if( (song_view['song'] == song_infor['title']) & (song_view['artist'] in song_infor['artist']) ):
            scrobbles_number = song_view['scrobbles']
            views_number = song_view['listeners']
            all_top_songs_list[i].update({'scrobbles': scrobbles_number, 'views': views_number})
            break            
        else:
            all_top_songs_list[i].update({'scrobbles':0 , 'views': 0})
all_top_songs_list_Lastfm = all_top_songs_list

with open('./data/all_top_songs_list_Lastfm.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(all_top_songs_list_Lastfm, f)      
    f.close()  


##############
with open('./data/all_top_songs_list_Lastfm.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    all_top_songs_list_Lastfm = pickle.load(f)      
    f.close()   
    
# Turn to dataframe and attention on songs with more than 0 view    
import pandas as pd
# All song Lastfm
all_top_songs_list_Lastfm_df = pd.DataFrame(all_top_songs_list_Lastfm)
all_top_songs_list_Lastfm_df = all_top_songs_list_Lastfm_df[all_top_songs_list_Lastfm_df['views']>0]
# All hot song
all_top_songs_list_df = pd.DataFrame(all_top_songs_list)



##############    
# Filters out songs with collaborations, and with mark
# Keywords: feat., featuring, &, X, $, +, x 
collabo_songs_index = all_top_songs_list_Lastfm_df['artist'].str.lower().str.contains(" feat")
collabo_songs_index = collabo_songs_index | (all_top_songs_list_Lastfm_df['artist'].str.lower().str.contains(" & "))
collabo_songs_index = collabo_songs_index | (all_top_songs_list_Lastfm_df['artist'].str.lower().str.contains(" x "))
strange_symbol_songs_index = all_top_songs_list_Lastfm_df['artist'].str.lower().str.contains("\$")
year_index = all_top_songs_list_Lastfm_df['year']>=2008
deleted_index = ((collabo_songs_index & year_index) | strange_symbol_songs_index) 
all_top_songs_list_Lastfm_df = all_top_songs_list_Lastfm_df[~(deleted_index)]


##############
# Scatter plot as suggested by Prof. Skiena
rank_range = [1,100]
songs_in_rank = all_top_songs_list_Lastfm_df[(all_top_songs_list_Lastfm_df['peakPos']>=rank_range[0])
 & (all_top_songs_list_Lastfm_df['peakPos']<=rank_range[1])]
years = songs_in_rank['year']   
popularity = songs_in_rank['views']
peakPos = songs_in_rank['peakPos']
fig = plt.figure(figsize=(20,20))
plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100)
plt.colorbar()
plt.ylim([0,2500000])
plt.title('Ranks Peak: '+ str(rank_range[0]) + ":" + str(rank_range[1]))
    

fig = plt.figure(figsize=(20,20))   
for i in range(5):
    rank_range = [20*i,20*(i+1)]
    songs_in_rank = all_top_songs_list_Lastfm_df[(all_top_songs_list_Lastfm_df['peakPos']>=rank_range[0])
     & (all_top_songs_list_Lastfm_df['peakPos']<=rank_range[1])]
    years = songs_in_rank['year']   
    popularity = songs_in_rank['views']
    peakPos = songs_in_rank['peakPos']    
    ax = fig.add_subplot(3,2,i+1)
    ax.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100)
    ax.set_ylim([0, 2500000])
    ax.set_title('Ranks Peak: '+ str(rank_range[0]) + ":" + str(rank_range[1]))
plt.colorbar()




#######################
# Build function with 2 variables (years and peakpos)
# Specify time
start_year = 1960
end_year = 2015
range_year =  (all_top_songs_list_Lastfm_df['year']>=start_year) & (all_top_songs_list_Lastfm_df['year']<=(end_year))
data_regression_year_range = all_top_songs_list_Lastfm_df[range_year]
### First varialble - time_variable = e^(-year)
time_variable = np.array(end_year - data_regression_year_range['year'])
time_variable = np.exp(-1 * time_variable)
### Second variable - rank_variable = peakPos
rank_variable = np.array(data_regression_year_range['peakPos'])
### Invented variable
invented_variable = popularity_functions(rank_variable, time_variable)
### Output (Popularity) labels values
popularity_output = np.array(data_regression_year_range['views'])


### Linear Regression
# Divind data into train and test
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = invented_variable
Y = popularity_output
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
X_train = np.reshape(X_train,[-1,1])
X_val = np.reshape(X_val,[-1,1])
Y_train = np.reshape(Y_train,[-1,1])
Y_val = np.reshape(Y_val,[-1,1])
# Building model:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
model.coef_
model.intercept_
# Validating:
def rmse(y1, y2):
    return np.sqrt(np.power((y1 - y2),2).mean())
Y_predicted = model.predict(X_val)
Loss_val = rmse(Y_val,Y_predicted)
Loss_val
















































