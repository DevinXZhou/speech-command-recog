'''
Created on Oct 3, 2017

@author: original by Doctor Roch
        Modified by Xin Zhou
'''

from .audioframes import AudioFrames
from .dftstream import DFTStream

import numpy as np
import hashlib  # hash functions


def get_features(file, adv_ms, len_ms, pca=None, components=0, offset_s=.25, flatten=True):
    """get_features(file, adv_ms, len_ms, pca, components, offset_s, flatten=True)
    
    Given a file path (file), compute a spectrogram with
    framing parameters of adv_ms, len_ms.  Retain only the central
    +/- offset_s of features.

    If a pca object is given, reduce the dimensionality of the spectra to the
    specified number of components using a PCA analysis (dsp.PCA object in
    variable pca).
    
    If flatten is True, convert to 1D feature vector    
    """
    
    framestream = AudioFrames(file, adv_ms, len_ms)
    dftstream = DFTStream(framestream)
    
    spectra = []
    for s in dftstream:
        spectra.append(s)
    # Row oriented spectra
    spectra = np.asarray(spectra)

    # Take center of spectra +/- offset_s    
    offset_frames = int(offset_s * 1.0 / (adv_ms/1000))

    # Take center .5 s
    frames = spectra.shape[0]  # Number of spectral frames

    center = int(frames / 2.0)
    left = max(0, center - offset_frames)
    right = min(frames, center + offset_frames)
    
    if frames < 2 * offset_frames + 1:
        raise RuntimeError("File {} too short".format(file))
    
    # Retain center -/+ offset_s
    features = spectra[left:right,:]
    
    # Convert spectra to PCA space
    if not pca is None:
        features = pca.transform(features, components)
        
    # Convert matrix to vector for input
    if flatten:
        features = features.flatten()

    return features