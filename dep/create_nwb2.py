from decode_lab_code.preprocessing.caiman_wrapper import caiman_preprocess as cp

class data2nwb():
    def __init__(self, folder_name: str, file_name: str):
        """ 
        folder_name: directory containing the data of interest
        file_name: file name that is within the directory (folder_name)
        """
        self.folder_name = folder_name
        self.file_name = file_name
    
        
