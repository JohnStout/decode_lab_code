# decode_lab_code
 A python toolbox to support analyses performed in the DECODE lab at Nemours Childrens Hospital in Wilmington DE, led by A.Hernan and R.Scott

 This package is a work-in-progress and is in early development.

 As of 9/12/2023, most of the analysis/preprocessing tools have been focused on calcium imaging techniques, specifically using CaImAn. There is also code dedicated to NWB file creation. As such, you should have caiman and pynwb packages installed.

 If planning on using this code with caiman to use caiman_wrapper, then you should install this package, and with decode_lab_code in your directory and caiman as your environment, pip install e .

 -- NWB file types -- 

 For ephys, the lab uses Neuralynx and as such, it is easiest to use the neuroconv toolbox to get your data into NWB
    /preprocessing/ephys/nlx2nwb.py

For ophys, the lab uses different recording approaches (e.g. lionheart, UCLA miniscope), it is easiest to use the pynwb toolbox for this
    /preprocessing/ophys/createNWB_ophys.py


-- Current Goals -- 

Get NWB file types to work with CaImAn
--> CaImAn doesnt love reading these. It likes the .avi and .tif types.

Get NWB file types to work with PyNapple
--> This is already supported by PyNapple

Compile the read/write NWB functions
--> There will be a fixed set of things that you change each time (e.g. experimenter name, experiment notes, experiment title...)
--> There will be a fixed set of namings for the devices. 
-> I can therefore generalize the defining of NWB using neuroconv toolbox
 

One major issue with NWB is that you have to fill out so much information. I think if I make a way to initialize a save file that is then populated into a function, it would work best. So a user only ever has to adjust their parameters once