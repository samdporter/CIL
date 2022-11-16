import pywt
import numpy as np

from cil.optimisation.operators import LinearOperator
from cil.framework import BlockDataContainer, BlockGeometry
from sirf.STIR import ImageData

class bdc_to_array():

    def __init__(self, image):
        self.image = image

    def direct(self, bdc):
        im_lst = []
        for image in bdc.clone():
            im_lst.append(image.as_array())
        return np.array(im_lst)

    def adjoint(self, array):
        im_lst = []
        for i in range(array.shape[0]):
            im = self.image.clone().fill(array[i])
            im_lst.append(im)
        print(im_lst)
        return BlockDataContainer(*im_lst)


class WaveletOperator(LinearOperator):
    def __init__(self, domain_geometry, templ_sino, wavelet = 'bior1.3'):
        
        # Consider pseudo 2D geometries with one slice, e.g., (1,voxel_num_y,voxel_num_x)
        domain_shape = []
        self.ind = []
        for i, size in enumerate(list(domain_geometry.shape) ):
            if size!=1:
                domain_shape.append(size)
                self.ind.append(i)
     
        # Dimension of domain geometry        
        self.ndim = len(domain_shape) 

        self.wavelet = wavelet
        self.templ_sino = templ_sino
        self.coeff_list = []

        self.domain_image = domain_geometry

        darr = np.squeeze(self.domain_image.as_array())
        # wavelet transform
        coeffs = pywt.dwtn(darr, self.wavelet)

        range_image = self.image_init(domain_geometry, coeffs)[0]

        range_geometry = BlockGeometry(*[range_image for _ in range(2**self.ndim) ] )

        super(WaveletOperator, self).__init__(domain_geometry=domain_geometry, 
            range_geometry = range_geometry)

    def image_init(self, x, coeffs):
        # create new temporary imaget to fill with transformed numpy array
        tmp = ImageData(self.templ_sino) # use templ_sino to create image
        dims = list(coeffs.values())[1].shape # find required dimensions
        if self.ndim == 2:
            tmp_lst = list(dims)
            tmp_lst.insert(0,1)
            dims = tuple(tmp_lst)
            del tmp_lst
        voxel_size=x.voxel_sizes()
        tmp.initialise(tuple(dims),voxel_size)

        return tmp, dims
        
    def direct(self, x, out = None):

        # convert to numpy array. remove 1st dimension if pseudo-3D
        xarr = np.squeeze(x.as_array())

        # wavelet transform
        coeffs = pywt.dwtn(xarr, self.wavelet)

        tmp, dims = self.image_init(x, coeffs)

        # save titles of wavelet transform for later use
        if not self.coeff_list:
            self.coeff_list = list(coeffs.keys())

        # fill BlockDataContainer with images of transform
        if out is None:
            return BlockDataContainer(*[tmp.clone().fill(coeffs[_].reshape(dims)) for _ in coeffs] )

        else:
            out = BlockDataContainer(*[tmp.clone().fill(coeffs[_].reshape(dims)) for _ in coeffs] )
            
    def adjoint(self, x, out = None):

        # convert to dictionary of coefficiants
        image_dict = {self.coeff_list[_]:np.squeeze(x[_].as_array()) for _ in range(len(x))}

        # inverse wavelet transform
        image = pywt.idwtn(image_dict, self.wavelet)
        image = image.reshape(self.domain_image.allocate().shape)

        # fill Image and return
        if out is None:
            return self.domain_image.allocate().fill(image)

        else:
            self.domain_image.allocate().fill(image)
