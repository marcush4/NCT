import pickle


"""" 
TO INITIALIZE A PRELOAD DICT FOR A NEW RUN ::
file_path = '/home/marcush/Data/TsaoLabData/split/degraded/preloaded/preloadDict.pickle'
"""
file_path = '/home/marcush/Data/TsaoLabData/split/degraded/preloaded/'
filename = 'preloadDict.pickle'

preloadDict = {}
with open(file_path+filename, 'wb') as file:
    pickle.dump(preloadDict, file)