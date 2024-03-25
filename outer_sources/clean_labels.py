import pandas as pd

# source : https://gitlab.com/minds-mines/alzheimers/alz/-/blob/master/alz/viz/brain/clean_labels.py

""" Represents an object used to clean up Alzheimer's Data
    
A DataCleaner object is able to clean up VBM data in order to match
the column labels to a nilearn atlas in order to create useful images
of the important parts of the brain
"""

class VBMDataCleaner:
    def __init__(self):
        """Initializes an empty list to hold the cleaned Alzheimer's Disease data"""
        self.raw_data = []
    
    def load_data(self, path, sheet=0):
        """Loads the (unclean) data into this DataCleaner"""
        if path.endswith(".csv"):
            data = pd.read_csv(path)
        elif path.endswith(".xlsx"):
            data = pd.read_excel(path, sheet_name=sheet)
        else:
            raise IOError
        self.raw_data = data.columns

    def strip_header(self):
        """Strips the header from each column entry"""
        for x in self.raw_data:
            to_strip = x[:17]
            if to_strip != 'BL_MPavg_MNI_mod_':
                raise TypeError(x + " does not have the correct header.\n" +
                                "trying to strip away: " + to_strip)
            yield x[17:]

    def move_handedness(self):
        """Moves the handedness designation, either 'L' or 'R' to the end of the label"""
        for x in self.raw_data:
            handedness = x[0]
            if handedness != 'R' and handedness != 'L':
                raise TypeError(x + " does not follow the correct format.\n" +
                                "trying to move: " + handedness)
            yield x[1:] + "_" + x[0]

    def move_to_end(self, substring):
        """Moves a given substring to the end of the label"""
        for x in self.raw_data:
            if substring in x:
                yield x.replace(substring, "") + "_" + substring
            else:
                yield x

    def replace(self, original, replacement):
        """Replaces a given substring in the label"""
        for x in self.raw_data:
            if original in x:
                yield x.replace(original, replacement)
            else:
                yield x

    def clean(self):
        """The set of ORDERED transformations necessary to reformat the labels correctly"""
        self.raw_data = list(self.strip_header())

        self.raw_data = list(self.move_to_end("Inf"))
        self.raw_data = list(self.move_to_end("Mid"))
        self.raw_data = list(self.move_to_end("Sup"))
        self.raw_data = list(self.move_to_end("Med"))
        self.raw_data = list(self.move_to_end("Ant"))
        self.raw_data = list(self.move_to_end("Post"))
        self.raw_data = list(self.move_to_end("Orb"))
        self.raw_data = list(self.move_to_end("Triang"))
        self.raw_data = list(self.move_to_end("Oper"))

        self.raw_data = list(self.move_handedness())

        self.raw_data = list(self.replace("Cingulate", "Cingulum"))
        self.raw_data = list(self.replace("Triang", "Tri"))
        self.raw_data = list(self.replace("ramarg_Sup", "SupraMarginal"))
        self.raw_data = list(self.replace("pMotorArea_Sup", "Supp_Motor_Area"))
        self.raw_data = list(self.replace("central_Post", "Postcentral"))
        self.raw_data = list(self.replace("TempPole", "Temporal_Pole"))
        self.raw_data = list(self.replace("Frontal_Mid_Orb", "Frontal_Med_Orb")) # Spelling error
        self.raw_data = list(self.replace("Frontal_Sup_Med", "Frontal_Sup_Medial"))
        self.raw_data = list(self.replace("Parahipp", "ParaHippocampal"))
        self.raw_data = list(self.replace("Paracentral", "Paracentral_Lobule"))

        self.raw_data = list(self.replace("__", "_"))

        return self.raw_data
