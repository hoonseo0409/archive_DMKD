import numpy as np

from nilearn import datasets, image, plotting

# source : https://gitlab.com/minds-mines/alzheimers/alz/-/blob/master/alz/viz/brain/roi_map.py

class VBMRegionOfInterestMap:
    """A Collection of Brain Region of Interests (ROIs)

    A RegionOfInterestMap is an encapsulated object that can be plotted
    using the nilearn.plotting package.

    Features:
      - Load an ROI Atlas
      - Add a weighted ROI
      - Calculate a map (brain image precursor) of all ROIs
    """
    def __init__(self):
        """Constructor initializes this object with the AAL atlas provided by nilearn (by default)"""
        aal_atlas = datasets.fetch_atlas_aal("SPM12") 
        self.load_atlas(aal_atlas)

        self.roi_maps = []
        self.current_map = None

    def load_atlas(self, atlas):
        """Loads the atlas file into this ROI plotting object

        The atlas should follow convention in the atlas' provided by
        nilearn.datasets.fetch* 
        """
        if not hasattr(atlas, 'indices'):
            raise TypeError("The provided atlas does not contain an 'indices' attribute.")

        if not hasattr(atlas, 'labels'):
            raise TypeError("The provided atlas does not contain an 'labels' attribute.")

        if not hasattr(atlas, 'maps'):
            raise TypeError("The provided atlas does not contain an 'maps' attribute.")

        self.atlas = atlas

    def add_roi(self, label, weight):
        """Add a weighted ROI 

        Adds a weighted ROI to the list of ROIs contained within the 
        RegionOfInterestMap.
        """
        if self.atlas is None:
            raise RuntimeError("The ROI atlas has not been loaded into this object." +
                               "Try to load_atlas(path_to_roi_atlas).")
        roi_idx = self.atlas.indices[self.atlas.labels.index(label)]
        roi_map = image.math_img("(img == %s) * %s" % (roi_idx, weight), img=self.atlas.maps)
        self.roi_maps.append(roi_map)

    def build_map(self, **kwargs):
        """Calculates the final ROI map from the previously provided ROIs

        The returned map can be smoothed if the correct named parameter
        is passed into the function.
        """
        _map = image.math_img("img * 0", img=self.roi_maps[0])

        for roi_map in self.roi_maps:
            _map = image.math_img("img1 + img2", img1=_map, img2=roi_map)

        if 'smoothed' in kwargs and kwargs['smoothed']:
            _map = image.smooth_img(_map, fwhm=5)

        self.current_map = _map

    def plot(self, _title):
        """Displays image with the current_map"""
        plotting.plot_glass_brain(self.current_map, black_bg=False, title=_title)
        plotting.show()

    def save(self, _path, _title):
        """Save plot to _pathname"""
        plot = plotting.plot_glass_brain(self.current_map, black_bg=False, title=_title, output_file=_path)

class FSRegionOfInterestMap:
    """A Collection of Brain Region of Interests (ROIs)

    A RegionOfInterestMap is an encapsulated object that can be plotted
    using the nilearn.plotting package.

    Features:
      - Load an ROI Atlas
      - Add a weighted ROI
      - Calculate a map (brain image precursor) of all ROIs
    """
    def __init__(self):
        """Constructor initializes this object with the an array of atlases provided by nilearn (by default)"""
        self.harvard_oxford_atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        self.aal_atlas = datasets.fetch_atlas_aal("SPM12")
        self.talairach_atlas = datasets.fetch_atlas_talairach("ba")

        self.harvard_oxford_atlas.maps = image.resample_img(self.harvard_oxford_atlas.maps)
        norm_affine = self.harvard_oxford_atlas.maps.affine
        norm_shape = tuple([2*x for x in self.harvard_oxford_atlas.maps.shape])

        self.harvard_oxford_atlas.maps = image.resample_img(self.harvard_oxford_atlas.maps, 
                                                            target_affine=norm_affine,
                                                            target_shape=norm_shape)

        self.aal_atlas.maps = image.resample_img(self.aal_atlas.maps, 
                                                 target_affine=norm_affine,
                                                 target_shape=norm_shape)

        self.talairach_atlas.maps = image.resample_img(self.talairach_atlas.maps, 
                                                       target_affine=norm_affine,
                                                       target_shape=norm_shape)

        self.roi_maps = []
        self.current_map = None

    def add_roi(self, label, weight, atlas):
        """Add a weighted ROI 

        Adds a weighted ROI to the list of ROIs contained within the 
        FSRegionOfInterestMap.
        """
        roi_idx = -1
        atlas_img = None

        if atlas == "Harvard-Oxford":
            roi_idx = self.harvard_oxford_atlas.labels.index(label)
            atlas_img = self.harvard_oxford_atlas.maps
        elif atlas == "AAL":
            roi_idx = self.aal_atlas.indices[self.aal_atlas.labels.index(label)]
            atlas_img = self.aal_atlas.maps
        elif atlas == "Talairach":
            roi_idx = self.talairach_atlas.labels.index(label)
            atlas_img = self.talairach_atlas.maps
        
        roi_map = image.math_img("(img == %s) * %s" % (roi_idx, weight), img=atlas_img)
        self.roi_maps.append(roi_map)

    def build_map(self, **kwargs):
        """Calculates the final ROI map from the previously provided ROIs

        The returned map can be smoothed if the correct named parameter
        is passed into the function.
        """
        _map = image.math_img("img * 0", img=self.roi_maps[0])

        for roi_map in self.roi_maps:
            _map = image.math_img("img1 + img2", img1=_map, img2=roi_map)

        if 'smoothed' in kwargs and kwargs['smoothed']:
            _map = image.smooth_img(_map, fwhm=5)

        self.current_map = _map

    def plot(self, _title):
        """Displays image with the current_map"""
        plotting.plot_glass_brain(self.current_map, black_bg=False, title=_title)
        plotting.show()

    def save(self, _path, _title):
        """Save plot to _pathname"""
        plot = plotting.plot_glass_brain(self.current_map, black_bg=False, title=_title, output_file=_path)
