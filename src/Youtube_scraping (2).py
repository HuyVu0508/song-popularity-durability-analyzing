# This file is used to scraping Youtube data

import numpy as np
import pickle

####################################    
##### Lastfm Data    


# Lastfm dataset
partion = np.array(range(0,27001,1000))
for i in partion:
    with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
        all_top_songs_list = pickle.load(f)    
        print('Partion '+ str(i) + "/" + str(len(all_top_songs_list)))

i = 9999
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list)))
i = 11999
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list)))        
        
       
Lastfm_all_data = []    
i = 9999    
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 
    Lastfm_all_data += all_top_songs_list
    print('Lastfm_all_data length: ' + str(len(Lastfm_all_data)))
i = 11999    
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 
    Lastfm_all_data += all_top_songs_list
    print('Lastfm_all_data length: ' + str(len(Lastfm_all_data)))
i = 15000    
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 
    Lastfm_all_data += all_top_songs_list
    print('Lastfm_all_data length: ' + str(len(Lastfm_all_data)))
i = 20000    
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 
    Lastfm_all_data += all_top_songs_list
    print('Lastfm_all_data length: ' + str(len(Lastfm_all_data)))
i = 23000    
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 
    Lastfm_all_data += all_top_songs_list
    print('Lastfm_all_data length: ' + str(len(Lastfm_all_data)))
i = 24000    
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 
    Lastfm_all_data += all_top_songs_list
    print('Lastfm_all_data length: ' + str(len(Lastfm_all_data)))
i = 25000    
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 
    Lastfm_all_data += all_top_songs_list
    print('Lastfm_all_data length: ' + str(len(Lastfm_all_data)))
i = 27000    
with open('./data/Lastfm Data/all_hot_song_lastfm_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 
    Lastfm_all_data += all_top_songs_list
    print('Lastfm_all_data length: ' + str(len(Lastfm_all_data)))    
    
# Saving list
with open('./data/Lastfm Data/Lastfm_all_data.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Lastfm_all_data, f)      
    f.close()      
    
    
    
    
####################################    
##### Youtube Data    
# Lastfm dataset
partion = np.array(range(5000,27001,1000))
for i in partion:
    with open('./data/Youtube Data/Old Data/Youtube_view_all_songs_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
        all_top_songs_list = pickle.load(f)    
        print('Partion '+ str(i) + "/" + str(len(all_top_songs_list)))    



# Loop through all Youtube data    
i = 9000    
with open('./data/Youtube Data/Youtube_view_all_songs_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 

with open('./data/Youtube Data/Old Data/Youtube_view_all_songs_15000.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print(str(len(all_top_songs_list))) 
    all_top_songs_list[(len(all_top_songs_list)-3):(len(all_top_songs_list))]
    all_top_songs_list[1:3]

    
    
    
# Reusable
Youtube_view_all_songs_0_99 / 100
Youtube_view_all_songs_100_199 / 100
Youtube_view_all_songs_400_999 / 600    
Youtube_view_all_songs_3000_3999 / 999
Youtube_view_all_songs_4000_14999 / 1023
Youtube_view_all_songs_5000 / 544    
Youtube_view_all_songs_10000 / 1000
Youtube_view_all_songs_12000 / 940
Youtube_view_all_songs_13000 / 1023
Youtube_view_all_songs_17000 / 3000   
Youtube_view_all_songs_22000 / 1999    
Youtube_view_all_songs_23000 / 1000
Youtube_view_all_songs_25000 / 1999
Youtube_view_all_songs_26000 / 999
Youtube_view_all_songs_27000 / 1023    
Youtube_view_all_songs_25001_27004 / 1023  

    

# Using all scraped data  
# Old scrape data  
data_files = ["Youtube_view_all_songs_0_99", 
              "Youtube_view_all_songs_100_199", 
              "Youtube_view_all_songs_400_999",
              "Youtube_view_all_songs_3000_3999",
              "Youtube_view_all_songs_4000_14999",
              "Youtube_view_all_songs_5000",
              "Youtube_view_all_songs_10000",
              "Youtube_view_all_songs_12000",
              "Youtube_view_all_songs_13000",
              "Youtube_view_all_songs_17000",
              "Youtube_view_all_songs_22000",
              "Youtube_view_all_songs_23000",
              "Youtube_view_all_songs_25000",
              "Youtube_view_all_songs_26000",
              "Youtube_view_all_songs_27000",
              "Youtube_view_all_songs_25001_27004"
              ] 
Youtube_all_data = []        
for i in range(len(data_files)):
    with open('./data/Youtube Data/Old Data/'+data_files[i]+'.pickle','rb') as f:  
        all_top_songs_list = pickle.load(f)    
        Youtube_all_data += all_top_songs_list
        print("Loading " + data_files[i] + " / "+ str(len(all_top_songs_list)))

# New scrape data  
data_files = ["Youtube_view_all_songs_1000", 
              "Youtube_view_all_songs_2000", 
              "Youtube_view_all_songs_3000",
              "Youtube_view_all_songs_4000",
              "Youtube_view_all_songs_5000",
              "Youtube_view_all_songs_6000",
              "Youtube_view_all_songs_7000",
              "Youtube_view_all_songs_8000",
              "Youtube_view_all_songs_9000",
              "Youtube_view_all_songs_10000",
              "Youtube_view_all_songs_11000",
              "Youtube_view_all_songs_12000",
              "Youtube_view_all_songs_13000",
              "Youtube_view_all_songs_unscraped_12001_13800",
              "Youtube_view_all_songs_1000_scrape_again",
              "Youtube_view_all_songs_unscraped_0_1423_scrape_again"
              
              ]       
#Youtube_all_data = []        
for i in range(len(data_files)):
    with open('./data/Youtube Data/New Data/'+data_files[i]+'.pickle','rb') as f:  
        all_top_songs_list = pickle.load(f)    
        Youtube_all_data += all_top_songs_list
        print("Loading " + data_files[i] + " / "+ str(len(all_top_songs_list)))
    
    
# Turn into dataframe
import pandas as pd
Youtube_all_data_df = pd.DataFrame(Youtube_all_data)
Youtube_all_data_df['title & artist'] = Youtube_all_data_df['title'] + Youtube_all_data_df['artist']
Youtube_all_data_df = Youtube_all_data_df.drop_duplicates()    
# Get all_top_songs_list and turn into dataframe
with open('./data/all_top_songs_list.pickle','rb') as f: 
    all_top_songs_list = pickle.load(f)    
all_top_songs_list_df = pd.DataFrame(all_top_songs_list)    
all_top_songs_list_df['title & artist'] = all_top_songs_list_df['title'] + all_top_songs_list_df['artist']
# Loop through all_top_songs_list and update
songs_scraped_index = all_top_songs_list_df['title & artist'].isin(Youtube_all_data_df['title & artist'])    
all_top_songs_list_df['scraped'] =  songs_scraped_index  
# Select unscraped songs to put them into a list of dictionary
all_top_songs_list_unscraped_df = all_top_songs_list_df[~songs_scraped_index]    
all_top_songs_list_unscraped = list(all_top_songs_list_unscraped_df.T.to_dict().values())
# Save into pickle files
# Saving list
with open('./data/all_top_songs_list_unscraped.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(all_top_songs_list_unscraped, f)      
    f.close()     

    
with open('./data/Youtube_all_data_df.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Youtube_all_data_df, f)      
    f.close()    





######### Rap data year continuous
for i in range(len(Youtube_all_data_df)):
    if i%1000==0:
        print(i)
    Youtube_song_artist = Youtube_all_data_df.iloc[i]['title & artist']
    Youtube_all_data_df.iat[i,1] = all_top_songs_list_df[all_top_songs_list_df['title & artist'] == Youtube_song_artist]['peakPos']
    Youtube_all_data_df.iat[i,8] = all_top_songs_list_df[all_top_songs_list_df['title & artist'] == Youtube_song_artist]['year']
Youtube_all_data_df = Youtube_all_data_df.rename(index=str, columns={"view_count": "views"})    
    
    



######################################
##### Try plotting something interesting
with open('./data/Youtube_all_data_df.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    Youtube_all_data_df = pickle.load(f)      
    f.close()       
Youtube_all_data_df = Youtube_all_data_df.rename(index=str, columns={"view_count": "views"})
Youtube_all_data_df = Youtube_all_data_df[Youtube_all_data_df['views']>0]    
Youtube_all_data_df['year'].groupby("year").count()
    



# Scatter plot as suggested by Prof. Skiena
import matplotlib.pyplot as plt   
rank_range = [1,100]
songs_in_rank = Youtube_all_data_df[(Youtube_all_data_df['peakPos']>=rank_range[0])
 & (Youtube_all_data_df['peakPos']<=rank_range[1])]
years = songs_in_rank['year']   
popularity = songs_in_rank['views']
peakPos = songs_in_rank['peakPos']
fig = plt.figure(figsize=(20,20))
plt.scatter(years,(popularity),edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100)
plt.colorbar()
plt.title('Ranks Peak: '+ str(rank_range[0]) + ":" + str(rank_range[1]))
    

fig = plt.figure(figsize=(20,20))   
for i in range(5):
    rank_range = [20*i,20*(i+1)]
    songs_in_rank = Youtube_all_data_df[(Youtube_all_data_df['peakPos']>=rank_range[0])
     & (Youtube_all_data_df['peakPos']<=rank_range[1])]
    years = songs_in_rank['year']   
    popularity = songs_in_rank['views']
    peakPos = songs_in_rank['peakPos']    
    ax = fig.add_subplot(3,2,i+1)
    ax.set_ylim([2.5, 22.5])
    ax.scatter(years,np.log(popularity),edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100)
    ax.set_title('Ranks Peak: '+ str(rank_range[0]) + ":" + str(rank_range[1]))
plt.colorbar()




    
    
### Testing by visualization
rank_range = [1,100]
songs_in_rank = Youtube_all_data_df[(Youtube_all_data_df['peakPos']>=rank_range[0])
 & (Youtube_all_data_df['peakPos']<=rank_range[1])]
songs_in_rank = songs_in_rank[(songs_in_rank["year"] >= 1960) & (songs_in_rank["year"] < 1980)]
years = songs_in_rank['year']   
popularity = songs_in_rank['views']
peakPos = songs_in_rank['peakPos']    
fig = plt.figure(figsize=(10,10))
plt.scatter(peakPos,np.log(popularity),edgecolors='none',s=20, label="Songs")
#plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100, label="Songs")
plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=rank_range[0], vmax=rank_range[1], label="Songs")
#plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100, cmap=plt.get_cmap('rainbow'))
plt.colorbar().set_label("Peak rank", fontsize=20)
plt.title('Ranks Peak '+ str(rank_range[0]) + " - " + str(rank_range[1]), fontsize=25)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Listeners", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(labelsize=14)


from popularity_functions import popularity_function_guessing
from matplotlib.pyplot import cm
color=cm.rainbow(np.linspace(0,1,rank_range[1]+1-rank_range[0]))
for i in range(rank_range[0],rank_range[1]+1):
    time_variable_val = np.arange(1960, 2015, 0.1)
    rank_variable_val = [i]*len(time_variable_val)
    
    popularity_predicted = popularity_function_guessing(rank_variable_val, time_variable_val)
    plt.plot(time_variable_val, popularity_predicted, color = color[i - rank_range[0]], linestyle = '-', label="Rank "+str(i))    
    if((rank_range[1]-rank_range[0]) < 10):
        plt.legend()
    
    
    
############ Linear Regression
X_train = np.concatenate( (np.reshape(peakPos.values,(-1,1)), np.reshape(years.values,(-1,1))), axis = 1 )
Y_train = np.reshape( np.log(popularity.values),(-1,1) )        
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
model.coef_
model.intercept_        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    