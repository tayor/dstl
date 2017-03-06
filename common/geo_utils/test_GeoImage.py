
#
# Test GeoImage
#

from unittest import TestCase, TestLoader, TextTestRunner
import tempfile
import shutil
import os

# Numpy
import numpy as np

# GDAL
import gdal

# Project
from GeoImage import GeoImage
from common import get_gdal_dtype, get_dtype
from create_synthetic_images import create


class TestGeoImage(TestCase):

    def setUp(self):
        # Create local temp directory
        self.local_temp_folder = tempfile.mkdtemp()

    def tearDown(self):
        # Delete temp directory
        shutil.rmtree(self.local_temp_folder)

    def _create_synthetic_image_file(self, shape, depth, is_complex):
        # Create local synthetic image:
        filepath = os.path.join(self.local_temp_folder, 'test_small_image.tif')
        metadata = {'key_1': 'value_1', 'key_2': "1 2 3", 'key_3': '3'}
        geo_transform = (13.60746033, 0.001, 0.0, 50.25013288, 0.0, -0.001)
        geo_extent = np.array([
            [geo_transform[0], geo_transform[3]],
            [geo_transform[0] + (shape[1]-1)*geo_transform[1] + (shape[0]-1)*geo_transform[2],
             geo_transform[3]],
            [geo_transform[0] + (shape[1]-1)*geo_transform[1] + (shape[0]-1)*geo_transform[2],
             geo_transform[3] + (shape[1]-1)*geo_transform[4] + (shape[0]-1)*geo_transform[5]],
            [geo_transform[0],
             geo_transform[3] + (shape[1]-1)*geo_transform[4] + (shape[0]-1)*geo_transform[5]]
        ])
        data = create(shape[1], shape[0], shape[2], filepath,
                      depth=depth, is_complex=is_complex,
                      metadata=metadata, geo_transform=geo_transform, epsg=4326)
        self.assertTrue(os.path.exists(filepath))
        return filepath, data, geo_extent, metadata, geo_transform

    def test_with_synthetic_image(self):
        is_complex = False
        shape = (120, 100, 2)
        depth = 2
        filepath, data, geo_extent, metadata, geo_transform = self._create_synthetic_image_file(shape, depth,
                                                                                                is_complex)
        gimage = GeoImage(filepath)
        # Must add this metadata : 'IMAGE_STRUCTURE__INTERLEAVE': 'PIXEL', 'AREA_OR_POINT': 'Area'
        metadata['IMAGE_STRUCTURE__INTERLEAVE'] = 'PIXEL'
        metadata['AREA_OR_POINT'] = 'Area'
        self.assertEqual(metadata, gimage.metadata)
        self.assertTrue((geo_extent == gimage.geo_extent).all(),
                        "Wrong geo extent : {} != {}".format(geo_extent, gimage.geo_extent))

        gimage_data = gimage.get_data()
        self.assertEqual(shape, gimage_data.shape)
        self.assertEqual(get_dtype(depth, is_complex), gimage_data.dtype)
        # verify data
        self.assertEqual(float(np.sum(data - gimage_data)), 0.0)

    def test_with_synthetic_image_with_select_bands(self):
        is_complex = False
        shape = (120, 100, 5)
        depth = 2
        filepath, data, geo_extent, metadata, geo_transform = self._create_synthetic_image_file(shape, depth,
                                                                                                is_complex)

        gimage = GeoImage(filepath)
        # Must add this metadata : 'IMAGE_STRUCTURE__INTERLEAVE': 'PIXEL', 'AREA_OR_POINT': 'Area'
        metadata['IMAGE_STRUCTURE__INTERLEAVE'] = 'PIXEL'
        metadata['AREA_OR_POINT'] = 'Area'
        self.assertEqual(metadata, gimage.metadata)
        self.assertTrue((geo_extent == gimage.geo_extent).all(),
                        "Wrong geo extent : {} != {}".format(geo_extent, gimage.geo_extent))

        select_bands=[0, 2, 4]
        gimage_data = gimage.get_data(select_bands=select_bands)
        self.assertEqual(shape[:2], gimage_data.shape[:2])
        self.assertEqual(len(select_bands), gimage_data.shape[2])
        self.assertEqual(get_dtype(depth, is_complex), gimage_data.dtype)
        # verify data
        self.assertEqual(float(np.sum(data[:,:,select_bands] - gimage_data)), 0.0)

    def test_with_virtual_image(self):

        dataset, data = self._create_virt_image(100, 120, 2, np.uint16)
        gimage = GeoImage.from_dataset(dataset)

        gimage_data = gimage.get_data(nodata_value=0)

        # verify shape and dtype:
        self.assertEqual(data.shape, gimage_data.shape)
        self.assertEqual(data.dtype, gimage_data.dtype)

        # verify data
        self.assertEqual(float(np.sum(data - gimage_data)), 0.0)

    def test_with_virtual_image2(self):

        dataset, data = self._create_virt_image(100, 120, 2, np.float32)
        gimage = GeoImage.from_dataset(dataset)

        gimage_data = gimage.get_data(nodata_value=-123)

        # verify shape and dtype:
        self.assertEqual(data.shape, gimage_data.shape)
        self.assertEqual(data.dtype, gimage_data.dtype)

        # verify data
        self.assertEqual(float(np.sum(data - gimage_data)), 0.0)

    def _create_virt_image(self, w, h, c, dtype):
        # Create a synthetic gdal dataset
        data = np.arange(0, w*h*c, dtype=dtype).reshape((h, w, c))
        driver = gdal.GetDriverByName('MEM')
        gdal_dtype = get_gdal_dtype(data[0, 0, 0].itemsize,
                                    data[0, 0, 0].dtype == np.complex64 or
                                    data[0, 0, 0].dtype == np.complex128)
        ds = driver.Create('', w, h, c, gdal_dtype)
        for i in range(0, c):
            ds.GetRasterBand(i+1).WriteArray(data[:, :, i])
        driver = None
        return ds, data

    def test_from_dataset_with_select_bands(self):

        dataset, data = self._create_virt_image(100, 120, 5, np.float32)
        gimage = GeoImage.from_dataset(dataset)

        select_bands = [0, 2, 4]
        gimage_data = gimage.get_data(nodata_value=-123, select_bands=select_bands)

        # verify shape and dtype:
        self.assertEqual(data.shape[:2], gimage_data.shape[:2])
        self.assertEqual(len(select_bands), gimage_data.shape[2])
        self.assertEqual(data.dtype, gimage_data.dtype)

        # verify data
        self.assertEqual(float(np.sum(data[:,:,select_bands] - gimage_data)), 0.0)

        
if __name__ == "__main__":

    suite = TestLoader().loadTestsFromTestCase(TestGeoImage)
    TextTestRunner().run(suite)
