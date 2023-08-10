"""
Class and helper functions to locate predetermined molecules in SPM images.
Images should be imported following the structure in Koen's open_image script.
"""

# Molecules are classifies based on their Zernike moments calculated by mahotas
import mahotas
# We use euclidian distance on the moments to categorize similarity
from scipy.spatial import distance
# Numpy
import numpy as np
# SPIEPy helps with image processing and flattening
import spiepy
# Koen's SPM_analysis for further image processing and analysis
import spmanalysis as sa
# interpolate 2D is used to put all images to the same resolution
from scipy.interpolate import interp2d
# Noise filter images with Gaussian
from scipy.ndimage import gaussian_filter
# Polygons for regions of interest
from skimage.draw import polygon
# Further filters for tresholding and image treatment
from skimage.filters import gaussian, threshold_otsu, threshold_local
# Countours are determined by skimagfe.measure.contour
from skimage import measure
# To calculate distance matrix
from sklearn.metrics import pairwise_distances
# Floor and ceil for range determination
from math import floor, ceil
# Make sure not to alter original data in critical moments
from copy import copy


class Locator():
    """ Class that will do the heavy lifting of finding reference molecules in images.
    """
    def __init__(self):
        """ Initialize the basic parameters with some reasonable values
        """
        self.param = {}
        self.param["resolution"] = 8 #px/nm
        self.param["min_radius"] = .5 #nm
        self.param["min_seperation"] = 1 #nm
        self.param["offset"] = .0
        self.param["plane"] = 0 # 0 = flatten by iterate mask, 1 = poly_xy order 2
        self.param["gauss_blur"] = 8
        self.param["line_filter_first"] = False


    def get_reference(self, ref_image, chiral = False):
        """ Initialize the reference moments with an image of the reference molecule.
        Carefull with the ciral option, can lead to false double detection.
        """
        self.ref_image = copy(ref_image)
        flipped_image = copy(ref_image)
        flipped_image.data = np.fliplr(flipped_image.data)
        self.ref_moments = [self.get_moments(self.ref_image)]
        # Molecules can be chiral, and maybe we do not wnt to detect both (or we do)
        if chiral == True:
            self.flipped_image = flipped_image
            self.ref_moments.append(self.get_moments(self.flipped_image))


    def set_parameters(self, parameters):
        """ Set parameters by inputting a dict with the parameters you want to change
        """
        for parameter in parameters.keys():
            self.param[parameter] = parameters[parameter]


    def get_parameters(self):
        """ Just your standard getter.
        """
        return self.param


    def get_moments(self, image):
        """ Function to locate the potential molecules in an image, and return their
            molecule contours, Zernike moments, and other relevant physical criteria.
        """
        # First, rescale the image in xy
        image = self.rescale(image)
        # Next, background removal, flattening, and normalization of Z
        if self.param["line_filter_first"]:
            image.rescaled = sa.line_filtered(image.rescaled, 2)

        image.smooth = gaussian_filter(image.rescaled, self.param["gauss_blur"])
        image.for_contour = image.rescaled - image.smooth
        image.for_contour = self.plane(image.for_contour)

        # find contours
        contours = self.get_contours(image)
        # If there are contours, proces them
        if len(contours) > 0:
            # Initialize arrays
            image.radii = []
            image.moments = []
            # Now for each contour
            for contour in contours:
                # Find min enclosing radius and centre
                image.radii.append(self.find_min_enclosing_radius(contour))
                # Select region of interest
                roi = self.get_roi(image, image.radii[-1])
                # Get Zernike moments for region of interest with  specified radius
                image.moments.append(mahotas.features.zernike_moments(roi, image.radii[-1][0]))
            return image.moments, image.radii
        else:
            return [], []


    def find_min_enclosing_radius(self, contour):
        """ Function to find the minmimun enclosing radius of a countour.
        This is done using brute force.
        It returns the radius and the centre of the contour.
        """
        assert len(contour.shape) == 2
        pairwise_distances_matrix = pairwise_distances(contour)
        i, j = np.unravel_index(np.argmax(pairwise_distances_matrix), pairwise_distances_matrix.shape)
        ptA, ptB = contour[i], contour[j]
        radius = np.max(pairwise_distances([ptA, ptB])) # in pixels
        centre = np.mean([ptA, ptB], axis=0) # in pixels
        return radius, centre

    def get_contours(self, image):
        """ Get contours of molecules based on:
        1) Tresholding using otsu
        2) Remove contours that are to short or touch edges
        3) Countours that are closer together than min_seperation count as one
        """
        otsu = threshold_otsu(image.for_contour)
        image.binary_local = image.for_contour >= otsu
        # now we have a binary image, so .5 is a good level to find countours
        self.contours = measure.find_contours(image.binary_local, .5)

        # Next step: filter small ones and edges
        # Countours are in px, so we need to transform min_radius to px
        min_pixels = 2 * np.pi * self.param["min_radius"] * self.param["resolution"]
        self.cleaned_contours = []
        for contour in self.contours:
            if len(contour) > min_pixels:
                # See if it touches the edges
                if min(contour[:,0]) > 2 and max(contour[:,0]) < image.binary_local.shape[0]-2:
                    if min(contour[:,1]) > 2 and max(contour[:,1]) < image.binary_local.shape[1]-2:
                        self.cleaned_contours.append(contour)

        # Ok, so cleaned contours are long enough (noise suppresion).
        # What about their separation
        # First, convert to pixels
        minimum_separation = self.param["min_seperation"] * self.param["resolution"]
        # things that are very close together, count them together
        # new countours just not to overwrite cleaned contours
        self.new_contours = []
        # Keep track of which countours we have already used, as some will be added together
        used_indexes = []
        # Go over each cleaned contour
        for ii, cc in enumerate(self.cleaned_contours):
            poly1 = polygon(cc[:,0], cc[:,1])
            for jj, cc2 in enumerate(self.cleaned_contours):
                if jj>ii and ii not in used_indexes:
                    poly2 = polygon(cc2[:,0], cc2[:,1])
                    dist = distance.cdist(cc, cc2)
                    ## check if within another contour
                    answerx = np.isin(poly2[0], poly1[0])
                    answery = np.isin(poly2[1], poly1[1])
                    if np.all(np.logical_and(answerx, answery)):
                        used_indexes.append(jj)
                    ## check if within minimum_separation of all other contours:
                    if np.amin(dist) < minimum_separation:
                        cc = np.concatenate((cc, cc2))
                        used_indexes.append(jj)
            if ii not in used_indexes:
                self.new_contours.append(cc)

        return self.new_contours

    def rescale(self, image):
        """ Rescales the image data to the specified resolution.
            Returns the image with the scaled data in image.rescaled
            If name == ref_image name, also adds the rescaled data to self.image
        """
        resolution = image.points/image.XY_width
        # if the resolution is the correct one, we don't need to do anything
        if resolution == self.param["resolution"]:
            image.rescaled = image.data
        # Else: "We have work to do"
        else:
            # For interpolation we need a raw and fine x and y grid,
            # based on the size of the images
            x_raw_grid = np.mgrid[0:image.XY_width:image.points*1j]
            y_raw_grid = np.mgrid[0:image.XY_height:image.lines*1j]
            x_rescale_grid = np.mgrid[0:image.XY_width:
                                       int(self.param["resolution"]*image.XY_width)*1j]
            y_rescale_grid = np.mgrid[0:image.XY_height:
                                      int(self.param["resolution"]*image.XY_height)*1j]
            # To interpolate one must first define a function
            interp = interp2d(x_raw_grid, y_raw_grid, image.data, kind="cubic")
            # And then do the interpolation with the fine grid
            image.rescaled = interp(x_rescale_grid, y_rescale_grid)

        if image.name == self.ref_image.name:
            self.ref_image.rescaled = image.rescaled

        return image


    def plane(self, image):
        """ Raw images are usually not ideal to detect molecules.
        For now 2 mayor plane substraction methods are implemented that should
        do a reasonable job on bulk imported images.
        """
        if self.param["plane"] == 0:
            flatened = sa.flatten_by_iterate_mask(image)
        elif self.param["plane"] == 1:
            flatened = sa.flatten_poly_xy(image, 2)
        else:
            raise NotImplementedError("Plane Method not inplemented")
        return flatened


    def get_roi(self, image, circle):
        """ Based on the circle data, show the ROI as seen in the binary image
        obtained from thresholding. I.e., this is what the contour detection is
        based on.
        Mainly used for troubleshooting or optimizing detection parameters.
        """
        extent_y = [floor(circle[1][0]-circle[0]/2), ceil(circle[1][0]+circle[0]/2)]
        extent_x = [floor(circle[1][1]-circle[0]/2), ceil(circle[1][1]+circle[0]/2)]
        return image.binary_local[extent_y[0]:extent_y[1], extent_x[0]:extent_x[1]]

    def fit(self, ref_image):
        """ For people that come from SKlearn and expect a fit function implemented.
        """
        get_reference(ref_image, chiral = False)
        return self.ref_image


    def transform(self, image_list):
        """ Terminology from SKlearn.
        Goes through the image list, and for each image does the contour and
        moment detection. Does not generate distance list yet.
        """
        self.moments_list = []
        for image in image_list:
            image_moments, image_radii = self.get_moments(image)
            for count, moment in enumerate(image_moments):
                self.moments_list.append((image.name, moment, image_radii[count]))
        return self.moments_list

    def estimate(self):
        """ Terminology from SKlearn.
        Generates distance matrix based on transformed data.
        """
        self.d_matrix = distance.cdist([x[0] for x in self.ref_moments], [x[0] for x in self.moments_list])

    def circle_to_roi(self, circle):
        """ Used in plotting results. Uses the circle parameters for each contour
        to return the coordinates of a bounding box for that circle.
        Used to plot final results.
        Name adjusted to avoid colission with the previous function do do this
        specifically on the tresholded image.
        """
        x_min = np.max([floor(circle[1][1]-circle[0]/2)/self.param["resolution"], 0])
        y_min = np.max([floor(circle[1][0]-circle[0]/2)/self.param["resolution"], 0])
        lowleft = (x_min, y_min)
        y_max = ceil(circle[1][0]+circle[0]/2)/self.param["resolution"]
        x_max = ceil(circle[1][1]+circle[0]/2)/self.param["resolution"]
        upright = (x_max, y_max)
        return (lowleft, upright)


# For testing purposes
if __name__ == "__main__":
    print(1 + 1)
