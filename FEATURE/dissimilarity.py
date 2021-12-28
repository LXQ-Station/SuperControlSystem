import os
import numpy
import logging
import sys
import time
from loguru import logger



def Dissimilarity (user: numpy.ndarray, db: numpy.ndarray,):
    """The function to execute in FHE, on the untrusted server"""
    d = 1 - numpy.dot(user, db)
    #print("Encrypted dissimilarity = ", d)
    return d


if __name__ == "__main__":
    
    person_pics = os.listdir("./square_Color_features") # 某一类人脸路径下的全部人脸数据文件
    for name1 in person_pics:
        for name2 in person_pics:
           
            X = numpy.load("./square_Color_features/"+ name1)
            X = numpy.squeeze(X)
            
            Y = numpy.load("./square_Color_features/"+ name2)
            Y = numpy.squeeze(Y)
    
            print("{} VS {} dissmilarity = ".format(name1, name2), Dissimilarity(X,Y))
    '''
    X=numpy.load("feature_16489.jpg.npy")
    Y=numpy.load("feature_31251.jpg.npy")
    print("{} VS {} dissmilarity = ".format("",""), Dissimilarity(X,Y))
    '''
