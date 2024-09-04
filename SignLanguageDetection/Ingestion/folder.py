
import numpy as np
import os

class FolderStructure():
    def __init__(self, path:str, no_sequences:int, sequence_length:int, start_folder:int, actions:list[str] ) -> None:
        # Path to root dir
        self.DATA_PATH = path
        # number of  videos worth of data
        self.no_sequences = no_sequences
        # length of each video
        self.sequence_length = sequence_length
        
        self.start_folder = start_folder
        # list of actions to store
        self.actions = actions
    
    def create(self):
        for action in self.actions: 
            for sequence in range(self.no_sequences):
                try: 
                    os.makedirs(os.path.join(self.DATA_PATH, action, str(sequence)))
                except:
                    pass
                        
        