READ ME

This code provides some options to converting and working with your miniscope data.

First and foremost, we collect miniscope data as individual recording files of 1000 samples each. 
Every 1000 samples, a new video is made from the miniscope recordings.

---

If you want to convert your data to NWB

    miniscope-to-NWB.ipynb

If you want to write your NWB data to tif files and create one large memory mapped file

    NWB-to-memmap.ipynb

If you want to just use your raw and recently collected data, then memory map

    movies-to-mmap.ipynb


---

CNMFE results will almost certainly require dynamic interactions with mescore

---

Environment prep:

We will use a modified caiman environment for everything miniscope related

    conda install -n base -c conda-forge mamba   # install mamba in base environment
    mamba create -n caiman -c conda-forge caiman # install caiman
    mamba install -n caiman mesmerize-core       # install mescore for visualization
    conda activate caiman                        # activate virtual environment

    git clone https://github.com/JohnStout/decode_lab_code # install helper functions
    cd decode_lab_code
    pip install -e .

For converting data to NWB, use hernan-lab-to-nwb

    conda create -n nwb-env
    conda activate nwb-env
    git clone https://github.com/JohnStout/hernan-lab-to-nwb # install hernan-lab-to-nwb
    cd hernan-lab-to-nwb
    pip install -e .

For caiman help, follow the instructions on the CaImAn Flatiron institute github page:
https://github.com/flatironinstitute/CaImAn