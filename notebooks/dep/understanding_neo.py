## creating my own version of neuralynx extractor
from neo.rawio.neuralynxrawio import NeuralynxRawIO
from neo.io import get_io
from neo.io import NeuralynxIO
import matplotlib.pyplot as plt

dirname = '/Users/js0403/local data/Fear/2022-06-02_13-37-18 18eB R1 C2 Ext'

# read data
reader = NeuralynxIO(filename = dirname+'/'+'TT1_filtered.ntt')
blks = reader.read(lazy=False)
for blk in blks:
    for seg in blk.segments:
        print(seg)
        for asig in seg.analogsignals: # LFP
            print(asig)
        for st in seg.spiketrains: # spikes
            print(st)

plt.plot(st[0:32000])







# initializing the object
reader = NeuralynxRawIO(dirname = dirname)

# this extracts and organizes information
reader.parse_header()

reader = get_io(dirname)
data = reader.read()

global_metadata = {
    "session_start_time": data[0].rec_datetime,
    "identifier": data[0].file_origin,
    "session_id": "180116_0005",
    "institution": "University of Pavia",
    "lab": "D'Angelo Lab",
    "related_publications": "https://doi.org/10.1038/s42003-020-0953-x"
}




