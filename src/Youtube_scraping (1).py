# This file is used to scraping Youtube data - version 2

import song_functions
import pickle
# Scrape the rest of Youtube songs which have not been scraped
with open('./data/all_top_songs_list_unscraped_2.pickle','rb') as f: 
    all_top_songs_list_unscraped = pickle.load(f)    
    
 
Youtube_view_all_songs_unscraped = []
Error_songs = []
#for i in range(len(all_top_songs_list)):
import time   
t0 = time.clock()
start = 0
end = 1424
for i in range(start, end):    
    if i%50==0:
        print(i)
        print(time.clock()-t0)
        t0 = time.clock()
    try:
        song_info = all_top_songs_list_unscraped[i]
        song_stats = song_functions.lookup_Youtube(song_info['title'], song_info['artist'])   
        song_info_stats = {}
        song_info_stats.update(song_info)
        song_info_stats.update(song_stats)
        Youtube_view_all_songs_unscraped.append(song_info_stats)
    except:
        Error_songs.append(i)
        pass
    
    if (i%1000==0) & (i>0):
        print("Saving song number:" + str(i) )
        with open('./data/Youtube Data/New Data/Youtube_view_all_songs_'+ str(i) +'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(Youtube_view_all_songs_unscraped, f)      
            f.close()        
    
with open('./data/Youtube Data/New Data/Youtube_view_all_songs_unscraped_'+ str(start) +'_'+ str(end-1) +'.pickle', 'wb') as f: 
    pickle.dump(Youtube_view_all_songs_unscraped, f)      
    f.close()      