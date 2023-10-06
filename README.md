**decode_lab_code**

 A python toolbox to support analyses performed in the DECODE lab at Nemours Childrens Hospital in Wilmington DE, led by A.Hernan and R.Scott

**Goal:** develop an open-source package to lower the energy barrier between data storage/loading, analysis, and visualization. New students should focus on experiments, analysis, and visualization with minimal programming knowledge.

**Some dependencies:**
- neo
- pynwb
- nwbwidgets
- pynapple

**Central idea:** _When interfacing between raw data, NWB, pynapple, or custom code, the data will be stored as dictionary arrays._

Since data is often collected and locally stored different between labs, there is a requirement to develop lab-specific packages that readily works with data standards like NWB, while allowing for the flexibility to develop your own code or run through open source packages. As such, the central requirement of this package is that loaded items are converted between dictionary arrays, as they are intuitive for element access and support loose organization.
