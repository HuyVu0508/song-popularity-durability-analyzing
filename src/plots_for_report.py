import song_functions
import pickle
import matplotlib.pyplot as plt
import numpy as np
from popularity_functions import popularity_function_guessing

# Load data
with open('./data/all_top_songs_list_Lastfm.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    all_top_songs_list_Lastfm = pickle.load(f)      
    f.close()   


# Turn to dataframe and attention on songs with more than 0 view    
import pandas as pd
all_top_songs_list_Lastfm_df = pd.DataFrame(all_top_songs_list_Lastfm)
all_top_songs_list_Lastfm_df = all_top_songs_list_Lastfm_df[all_top_songs_list_Lastfm_df['views']>0]
  
    
# Set up range year
start_year = 1960
end_year = 2015
range_year =  (all_top_songs_list_Lastfm_df['year']>=start_year) & (all_top_songs_list_Lastfm_df['year']<=(end_year))
all_top_songs_list_Lastfm_df = all_top_songs_list_Lastfm_df[range_year]    
    
    
# Filters out songs with collaborations, and with mark
# Keywords: feat., featuring, &, X, $, +, x 
collabo_songs_index = all_top_songs_list_Lastfm_df['artist'].str.lower().str.contains(" feat")
collabo_songs_index = collabo_songs_index | (all_top_songs_list_Lastfm_df['artist'].str.lower().str.contains(" & "))
collabo_songs_index = collabo_songs_index | (all_top_songs_list_Lastfm_df['artist'].str.lower().str.contains(" x "))
strange_symbol_songs_index = all_top_songs_list_Lastfm_df['artist'].str.lower().str.contains("\$")
year_index = all_top_songs_list_Lastfm_df['year']>=2008
deleted_index = ((collabo_songs_index & year_index) | strange_symbol_songs_index) 
all_top_songs_list_Lastfm_df = all_top_songs_list_Lastfm_df[~(deleted_index)]    


# Power Law Visualization
year_range = [1960,2016]
year_observe = (all_top_songs_list_Lastfm_df['year']>=year_range[0]) & (all_top_songs_list_Lastfm_df['year']<=year_range[1])
all_views = np.array(all_top_songs_list_Lastfm_df[year_observe]['views'])
sorted_all_views = np.sort(all_views)[::-1]
plt.figure(figsize=(20,20))
#plt.bar(range(len(sorted_all_views),0,-1), (sorted_all_views))
plt.bar(range(len(sorted_all_views)), (sorted_all_views))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([0, 2500000])
plt.xlim([0, 2000])
plt.xlabel("Order (from most to least)", fontsize=30)
plt.ylabel("Listeners", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Sorted songs number of listeners from 1960 - 2015', pad = 30 , fontsize=40, fontweight='bold' )



# Prof. Skiena suggestions
rank_range = [1,100]
songs_in_rank = all_top_songs_list_Lastfm_df[(all_top_songs_list_Lastfm_df['peakPos']>=rank_range[0])
 & (all_top_songs_list_Lastfm_df['peakPos']<=rank_range[1])]
years = songs_in_rank['year']   
popularity = songs_in_rank['views']
peakPos = songs_in_rank['peakPos']    
fig = plt.figure(figsize=(20,20))
#plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100, label="Songs")
plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=rank_range[0], vmax=rank_range[1], label="Songs")
#plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100, cmap=plt.get_cmap('rainbow'))
plt.colorbar().set_label("Peak rank", fontsize=25)
plt.ylim([0,2000000])
plt.title('Ranks Peak '+ str(rank_range[0]) + " - " + str(rank_range[1]), fontsize=25)
plt.xlabel("Year", fontsize=25)
plt.ylabel("Listeners", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tick_params(labelsize=20)
plt.title('Number of listeners of songs from [1960-2015] in 2018', pad = 30, fontsize=30, fontweight='bold' )

## Subplot
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
    ax.set_ylim([0,2000000])
    ax.set_xlabel("Year", fontsize=25)
    ax.set_ylabel("Listeners", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_title('Ranks '+ str(rank_range[0]) + " - " + str(rank_range[1]), fontsize=25)
    plt.tight_layout()
fig.colorbar()    
plt.colorbar(fig)



# Plot prediction line
from matplotlib.pyplot import cm
color=cm.rainbow(np.linspace(0,1,rank_range[1]+1-rank_range[0]))
for i in range(rank_range[0],rank_range[1]+1):
    time_variable_val = np.arange(1960, 2015, 0.1)
    rank_variable_val = [i]*len(time_variable_val)
    
    popularity_predicted = popularity_function_guessing(rank_variable_val, time_variable_val)
    plt.plot(time_variable_val, popularity_predicted, color = color[i - rank_range[0]], linestyle = '-', label="Rank "+str(i))    
    if((rank_range[1]-rank_range[0]) < 10):
        plt.legend()
    

testing_ranks = range(4)
fig = plt.figure(figsize=(20,20))   
for i in testing_ranks:
    testing_rank = i+1
    songs_in_rank = all_top_songs_list_Lastfm_df[all_top_songs_list_Lastfm_df['peakPos'] == testing_rank]
    # Plot scatter
    years = songs_in_rank['year']   
    popularity = songs_in_rank['views']
    ax = fig.add_subplot(2,2,i+1)
    ax.scatter(years,popularity,edgecolors='none',s=20, label="True popularity")
    ax.set_ylim([0,2000000])
    ax.set_xlabel("Year", fontsize=20)
    ax.set_ylabel("Listeners", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_title('Number of listeners through time of Rank '+ str(testing_rank), fontweight='bold', fontsize=20, pad = 20)    
    # Plot lines
    time_variable_val = np.arange(1960, 2015, 0.1)
    rank_variable_val = [testing_rank]*len(time_variable_val)    
    popularity_predicted = popularity_function_guessing(rank_variable_val, time_variable_val)
    plt.plot(time_variable_val, popularity_predicted, color = "r", linestyle = '-', label="Predicted popularity")    
    plt.legend(fontsize=20)    
    plt.tight_layout()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    