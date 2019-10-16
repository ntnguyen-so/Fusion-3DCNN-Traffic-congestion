## Importing the libraries
import numpy as np
import numpy.core.defchararray as np_f
import os
#from DataReader import *

class DataParser():
    def __init__(self):
        pass

    def removeDelimiters(self, data, col, removed_str):
        data_col = data[:, col]
        for removed_s in removed_str:
            data_col = [str(s).replace(removed_s, '') for s in data_col]

        data_col = np.array(data_col)
        data[:, col] = data_col

        return data

    def convertDatetime(self, data, col):
        data_col = data[:, col]
        data_col = data_col.astype(np.int)

        data[:, col] = data_col

        return data
        
    def convertLocation(self, data, col, delimiter):
        data_col = data[:, col]
        
        location = np.zeros((len(data_col), 2))
        for i in range(len(data_col)):
            row = data_col[i]
            longitude, latitude = row.split(delimiter)
            
            location[i, 0] = latitude
            location[i, 1] = longitude
        
        data = data[:, :-1]
        data = np.hstack((data, location))
        return data            
    
    def convertList(self, data, col, bound, sep):
        data_col = data[:, col]
        data_col[data_col == 0] = bound
        data_col = [s.count(sep) + 1 for s in data_col]
        data[:, col] = data_col

        return data

    def convertInt(self, data, col):
        data_col = data[:, col]
        data_col = data_col.astype(int)
        data[:, col] = data_col

        return data

    def convertTextType(self, data, col, alias):
        data_col = data[:, col]
        for i in range(len(data_col)):
            counter = 1
            for label in alias:                     
                if data_col[i] == label:
                    data_col[i] = counter
                    break
                counter += 1
                
        data[:, col] = data_col
        return data

    def countElements(self, data, col, delimeter):
        data_col = data[:, col]
        for i in range(len(data_col)):
            if data_col[i] == 0:
                data_col[i] = 0
            else:
                data_col[i] = data_col[i].count(delimeter) + 1
        
        data[:, col] = data_col
        return data

    def convertBoWType(self, data, col, BoW):
        data_col = data[:, col]

        for i in range(data_col.shape[0]):
            if data_col[i] != 0:
                message = data_col[i]
                data_col[i] = 0
                counter = 0
                for word_type in BoW:
                    words = word_type.split(',')
                    for w in words:
                        if w in message:
                            data_col[i] += 2 ** counter
                            break
                        
                    counter += 1

        data_col = data_col.astype(int)
        data[:, col] = data_col
        return data

    def detectOccurrence(self, data, col, BoW):
        data_col = data[:, col]

        for i in range(data_col.shape[0]):
            if data_col[i] != 0:
                message = data_col[i]
                data_col[i] = 0
                words = BoW.split(',')
                for w in words:
                    if w in message:
                        data_col[i] = 1
                        break                

        data_col = data_col.astype(int)
        data[:, col] = data_col
        return data


    def getUniqueWords(self, data, col):
        data_col = data[:, col]
        posts = list(np.unique(data_col))
        unique_words = set()
        for post in posts:
            if ',' in post:
                words = post.split(',')
            else:
                words = post
            for word in words:
                unique_words.add(word)
        return unique_words
            
