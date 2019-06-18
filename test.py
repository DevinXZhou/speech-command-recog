from dsp.utils import get_corpus
from dsp.audioframes import AudioFrames
from dsp.rmsstream import RMSStream
import os
import numpy as np
adv_ms = 10
len_ms = 20

dir0 = os.getcwd()+"/testAudio"
list0 = get_corpus(dir0)
for name in list0:
	stream = AudioFrames(name, adv_ms, len_ms)
	rms = RMSStream(stream)
	array = []
	for v in rms:
		array.append(v)
	print(name)
	print(np.std(array))