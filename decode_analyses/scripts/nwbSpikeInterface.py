# call in some packages
# remember to select interpreter! Command+shift+p

# This package is for reading nwb data easily
from decode_converters.ephys.nlx2nwb import interfaceNLX

# read or write?
dataCommand = 'read'

# lets get our path with data
folder_name = '/Users/js0403/Sample-data'

# pull in data
if dataCommand == 'read':
    print("reading nwb data")
    data_nwb = interfaceNLX.readNWB(folder_name,'data_nwb')
elif dataCommand == 'write':
    # write data
    print("Writing data...")
    interfaceNLX.writeNWB(fileName='data_nwb')
    print("Reading data...")
    data_nwb = interfaceNLX.readNWB(folder_name,'data_nwb')

# The spikeinterface module by itself imports only the spikeinterface.core submodule
# which is not useful for end user
import matplotlib.pyplot as plt
from pprint import pprint
import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from warnings import simplefilter
simplefilter("ignore")

# working with this code: 
# https://github.com/NeurodataWithoutBorders/nwb_hackathons/blob/main/Cosyne_2023/tutorials/spikeinterface_tutorial.ipynb

# Before getting started, we can set some global arguments for parallel processing. For this example, let's use 4 jobs and time chunks of 1s:
global_job_kwargs = dict(n_jobs=4, chunk_duration="1s")
si.set_global_job_kwargs(**global_job_kwargs)

# read nwb data and set it to "recording_nwb" object
recording_nwb = se.read_nwb_recording(file_path=folder_name+'/data_nwb')

# we want to let the recording_nwb "extractor" know that the data has not been filtered. This will prevent
# mistakes in the pipeline, like extracting waveforms from unfiltered data
recording_nwb.annotate(is_filtered=False)
se.recording_extractor_full_dict

# A RecordingExtractor object extracts information about channel IDs, channel locations (if present),
# the sampling frequency of the recording, and the extracellular traces (when prompted). 
channel_ids = recording_nwb.get_channel_ids() # treating it like a function, rather than an attribut (check my facts)
fs = recording_nwb.get_sampling_frequency()
num_chan = recording_nwb.get_num_channels()
num_segments = recording_nwb.get_num_segments()

print(f'Channel ids: {channel_ids}')
print(f'Sampling frequency: {fs}')
print(f'Number of channels: {num_chan}')
print(f"Number of segments: {num_segments}")

trace_snippet = recording_nwb.get_traces(start_frame=int(fs*0), end_frame=int(fs*2))
print('Traces shape:', trace_snippet.shape)

# -- properties -- #
print("Properties:\n", list(recording_nwb.get_property_keys()))

# remove CSC data for clustering spikes
remData = []
for chi in range(len(channel_ids)):
    if 'CSC' in channel_ids[chi]:
        remData.append(chi)
remData = np.array(remData)
# remove CSC
tt_nwb = recording_nwb.remove_channels(channel_ids[remData])

# generate tetrode
tt_ids = tt_nwb.get_channel_ids() # get IDs
tt_count = int(len(tt_ids)/4) # group data into tetrode
probeGroup = ProbeGroup() # create a "ProbeGroup"
for i in range(tt_count):
    tt = generate_tetrode()
    tt.set_device_channel_indices(np.arange(tt_count) + i * 4)
    probeGroup.add_probe(tt)
tt_grouped = tt_nwb.set_probegroup(probeGroup, group_mode='by_probe')
print(tt_grouped.get_channel_groups())
print(tt_grouped.get_property('group'))

# lets filter our data
from spikeinterface.preprocessing import bandpass_filter, common_reference
tt_filt = bandpass_filter(tt_grouped, freq_min=300, freq_max=6000)
tt_cmr = common_reference(tt_filt, operator="median")

# how can we visualize our progress?
chI = 0
exData = recording_nwb.get_traces(chI,start_frame = 0, end_frame = 32000)

plt.plot(exData)