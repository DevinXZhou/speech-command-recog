'''
Created on Sep 22, 2017

@author: original by Doctor Roch
        Modified by Xin Zhou
'''

from .pca import PCA
from .multifileaudioframes import MultiFileAudioFrames
from .dftstream import DFTStream
from .features import get_features

import os.path
from datetime import datetime
import numpy as np

import hashlib  # hash functions




def pca_analysis_of_spectra(files, adv_ms, len_ms, offset_s): 
    """"pca_analysis_of_spectra(files, advs_ms, len_ms, offset_s)
    Conduct PCA analysis on spectra of the given files
    using the given framing parameters.  Only retain
    central -/+ offset_s of spectra
    """

    md5 = hashlib.md5()
    string = "".join(files)
    md5.update(string.encode('utf-8'))
    hashkey = md5.hexdigest()
    
    filename = "VarCovar-" + hashkey + ".pcl"
    try:
        pca = PCA.load(filename)

    except FileNotFoundError:
        example_list = []
        for f in files:
            print("getting features from file: "+f)
            example = get_features(f, adv_ms, len_ms, 
                                   offset_s=offset_s, flatten=False)
            example_list.append(example)
            
        # row oriented examples
        spectra = np.vstack(example_list)
    
        # principal components analysis
        pca = PCA(spectra)

        # Save it for next time
        pca.save(filename)
        
    return pca

def extract_features_from_corpus(files, adv_ms, len_ms, offset_s, pca, 
                                 components):
    """extract_features_from_corpus(files, adv_ms, len_ms, offset_s,
        pca, components)
        
    Return a 2d array of features.  Each row contains a feature vector
    corresponding to one of the filenames passed in files
    
    Spectral features are extracted based on framing parameters advs_ms, len_ms
    and the center +/- offset_s are retained.
    
    These spectra are projected into a PCA space of the specified number
    of components using the PCA space contained in object pca which is of
    type dsp.pca.PCA.
    
    This method will attempt to read from cached data as opposed to
    computing the features.  If the cache does not exist, it will be
    created.  Note the the cache files are not portable across machine
    architectures.
    """

    # This part takes a bit of time, use cache based on files and parameters
    md5 = hashlib.md5()
    string = "".join(files)
    md5.update(string.encode('utf-8'))
    hashkey = md5.hexdigest()
    
    filename = "features-adv_ms{}-len_ms{}-offset_ms{}-hash{}.np".format(
        adv_ms, len_ms, offset_s*1000, hashkey)
    try:
        features = np.load(filename) 
    except IOError:
        example_list = []
        for f in files:
            example = get_features(f, adv_ms, len_ms, pca, components, offset_s)
            print('extracting features from '+f)
            example_list.append(example)
        features = np.stack(example_list, axis=0)
        
        # Cache on secondary storage for quicker computation next time
        features.tofile(filename)

    return features

       
def get_corpus(dir, filetype=".wav"):
    """get_corpus(dir, filetype=".wav"
    Traverse a directory's subtree picking up all files of correct type
    """
    
    files = []
    
    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(filetype)]:
            files.append(os.path.join(dirpath, filename))
                         
    return files
 
def get_labels_path(files):
    sample_path = files[0]

    #deleted 'silence': 0, 
    classmap = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6,
                'off':7, 'stop':8, 'go':9, 'bed':10, 'bird':11, 'cat':12, 'dog':13,
                'eight':14, 'five':15, 'four':16, 'happy':17, 'house':18, 'marvin':19,
                'nine':20, 'one':21, 'seven':22, 'sheila':23, 'six':24, 'three':25,
                'tree':26, 'two':27, 'wow':28, 'zero':29}
    labels = []
    paths = []
    for f in files:
        name = f.split('/')[-2]
        if name in classmap:
            labels.append(classmap[name])
            paths.append(f)

    return [np.array(labels), np.array(paths)]

# def get_class(files):
#     """get_class(files)
#     Given a list of files, extract numeric class labels from the filenames
#     """
    
#     # TIDIGITS single digit file specific
    
#     classmap = {'z': 0, '1': 1, '2': 2, '3': 3, '4': 4,
#                 '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'o': 10}

#     # Class name is given by first character of filename    
#     classes = []
#     for f in files:        
#         dir, fname = os.path.split(f) # Access filename without path
#         classes.append(classmap[fname[0]])
        
#     return classes
    
class Timer:
    """Class for timing execution
    Usage:
        t = Timer()
        ... do stuff ...
        print(t.elapsed())  # Time elapsed since timer started        
    """
    def __init__(self):
        "timer() - start timing elapsed wall clock time"
        self.start = datetime.now()
        
    def reset(self):
        "reset() - reset clock"
        self.start = datetime.now()
        
    def elapsed(self):
        "elapsed() - return time elapsed since start or last reset"
        return datetime.now() - self.start
    
