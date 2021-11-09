import unittest
import os
import numpy as np
from iss.utils.morphology import hanning_diff, Strel, imfilter, top_hat, dilate
from iss.utils.matlab import load_array
import iss.utils.errors


class TestMorphology(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_hanning_diff(self):
        """
        Check whether hanning filters are same as with MATLAB
        and that sum of filter is 0.

        test files contain:
        r1: inner radius of hanning filter
        r2: outer radius of hanning filter
        h: hanning filter produced by MATLAB
        """
        folder = os.path.join(self.folder, 'hanning')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        iss.utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r1, r2, output_matlab = load_array(test_file, ['r1', 'r2', 'h'])
            output_python = hanning_diff(int(r1), int(r2))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB
            self.assertTrue(np.abs(output_python.sum()) <= self.tol)  # check sum is zero

    def test_disk(self):
        """
        Check whether disk_strel gives the same results as MATLAB strel('disk)

        test_files contain:
        r: radius of filter kernel
        n: 0, 4, 6 or 8
        nhood: filter kernel found by MATLAB
        """
        folder = os.path.join(self.folder, 'disk')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        iss.utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r, n, output_matlab = load_array(test_file, ['r', 'n', 'nhood'])
            output_python = Strel.disk(int(r), int(n))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_disk_3d(self):
        """
        Check whether 3d structuring element are same as with MATLAB
        function strel3D_2

        test files contain:
        rXY: xy radius of structure element
        rZ: z radius of structure element
        kernel: structuring element produced by MATLAB strel3D_2 function (In iss 3d branch)
        """
        folder = os.path.join(self.folder, 'disk_3d')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        iss.utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r_xy, r_z, output_matlab = load_array(test_file, ['rXY', 'rZ', 'kernel'])
            output_python = Strel.disk_3d(int(r_xy), int(r_z))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= 0)  # check match MATLAB

    def test_annulus(self):
        """
        Check whether annulus structuring element matches MATLAB code:
        # 2D
        [xr, yr] = meshgrid(-floor(rXY):floor(rXY),-floor(rXY):floor(rXY));
        Annulus = (xr.^2 + yr.^2)<=rXY.^2 & (xr.^2 + yr.^2) > r0.^2;
        # 3D
        [xr, yr, zr] = meshgrid(-floor(rXY):floor(rXY),-floor(rXY):floor(rXY),-floor(rZ):floor(rZ));
        Annulus = (xr.^2 + yr.^2+zr.^2)<=rXY.^2 & (xr.^2 + yr.^2+zr.^2) > r0.^2;

        test files contain:
        rXY: xy outer radius of structure element
        rZ: z outer radius of structure element (0 if 2D)
        r0: inner radius of structure element, within which it is 0.
        Annulus: structuring element produced by MATLAB
        """
        folder = os.path.join(self.folder, 'annulus')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        iss.utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            r_xy, r_z, r0, output_matlab = load_array(test_file, ['rXY', 'rZ','r0', 'Annulus'])
            if r_z == 0:
                output_python = Strel.annulus(float(r0), float(r_xy))
            else:
                output_python = Strel.annulus(float(r0), float(r_xy), float(r_z))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= 0)  # check match MATLAB

    def test_imfilter(self):
        """
        Check whether filter_imaging gives same results as MATLAB:
        I_mod = padarray(image,(size(kernel)-1)/2,'replicate','both');
        image_filtered = convn(I_mod, kernel,'valid');

        test_file contains:
        image: image to filter (no padding)
        kernel: array to convolve image with
        image_filtered: result of MATLAB filtering
        """
        folder = os.path.join(self.folder, 'filter')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        iss.utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = load_array(test_file, ['image', 'kernel', 'image_filtered'])
            output_python = imfilter(image, kernel)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_filter_dapi(self):
        """
        Check whether filter_dapi gives same results as MATLAB:
        I_mod = padarray(image,(size(kernel)-1)/2,'replicate','both');
        image_filtered = imtophat(image, kernel);

        test_file contains:
        image: image to filter (no padding)
        kernel: array to apply tophat filter to image with
        image_filtered: result of MATLAB filtering
        """
        folder = os.path.join(self.folder, 'dapi')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        iss.utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            # MATLAB and python differ if kernel has any odd dimensions and is not symmetric
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = load_array(test_file, ['image', 'kernel', 'image_filtered'])
            output_python = top_hat(image, kernel)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    def test_dilate(self):
        """
        Check whether dilate gives same results as MATLAB imdilate function.

        test_file contains:
        image: image to dilate (no padding)
        kernel: structuring element to dilate with
        image_dilated: result of dilation.
        """
        folder = os.path.join(self.folder, 'dilate')
        test_files = [s for s in os.listdir(folder) if "test" in s and "even" not in s]
        iss.utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = load_array(test_file, ['image', 'kernel', 'image_dilated'])
            output_python = dilate(image, kernel.astype(int))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)  # check match MATLAB

    @unittest.expectedFailure
    def test_dilate_even(self):
        """
        as above but with even kernels, should fail.

        test_file contains:
        image: image to dilate (no padding)
        kernel: structuring element to dilate with
        image_dilated: result of dilation.
        """
        folder = os.path.join(self.folder, 'dilate')
        test_files = [s for s in os.listdir(folder) if "test" in s and "even" in s]
        iss.utils.errors.empty('test_files', test_files)
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, kernel, output_matlab = load_array(test_file, ['image', 'kernel', 'image_dilated'])
            output_python = dilate(image, kernel.astype(int))


if __name__ == '__main__':
    unittest.main()
