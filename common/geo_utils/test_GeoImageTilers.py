# Test GeoImage

import logging
import tempfile
import shutil
import os
from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np

from GeoImage import GeoImage
from GeoImageTilers import GeoImageTiler, GeoImageTilerConstSize
from create_synthetic_images import create


class TestGeoImageTiler(TestCase):
    
    def setUp(self):
        # Create local temp directory
        self.local_temp_folder = tempfile.mkdtemp()

        filepath = os.path.join(self.local_temp_folder, 'test_small_image.tif')
        metadata = {'key_1': 'value_1', 'key_2': "1 2 3", 'key_3': '3'}
        shape = (120, 100, 2)
        depth = 2
        is_complex = False
        geo_transform = (13.60746033, 0.001, 0.0, 50.25013288, 0.0, -0.001)
        _ = create(shape[1], shape[0], shape[2], filepath,
                   depth=depth, is_complex=is_complex,
                   metadata=metadata, geo_transform=geo_transform, epsg=4326)

        self.assertTrue(os.path.exists(filepath))
        self.geo_image = GeoImage(filepath)

    def tearDown(self):
        self.geo_image.close()
        # Delete temp directory
        shutil.rmtree(self.local_temp_folder)

    def test_tiling_no_overlapping_no_nodata(self):

        def _test(tile_size):
            tiled_image = np.zeros(self.geo_image.shape)
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size, overlapping=0, include_nodata=False)
            self.assertTrue(tiles.nx, int(np.ceil(tiled_image.shape[1] / tile_size[0])))
            self.assertTrue(tiles.ny, int(np.ceil(tiled_image.shape[0] / tile_size[1])))
            for tile, x, y in tiles:
                tiled_image[y:y+tile_size[1], x:x+tile_size[0], :] = tile

            err = float(np.sum(tiled_image - self.geo_image.get_data()))
            logging.debug("Err : %f" % err)
            self.assertTrue(np.abs(err) < 1e-10)

        _test((32, 32))
        _test((40, 40))
        _test((44, 46))
        _test((55, 76))

    # def test_tiling_no_overlapping_no_nodata_with_scale(self):
    #     tile_size = (32, 32)
    #     overlapping = 0
    #     scale = 1.234
    #
    #     def _f(_x):
    #         return int(np.ceil(_x))
    #
    #     # Total tiled image is smaller if scale > 1
    #     h, w, nc = self.geo_image.shape
    #     h = _f(h * 1.0 / scale)
    #     w = _f(w * 1.0 / scale)
    #     tiled_image = np.zeros((h, w, nc))
    #     logging.debug("tiled_image: {}".format(tiled_image.shape))
    #     logging.debug("geo_image.shape: {}".format(self.geo_image.shape))
    #
    #     tiles = GeoImageTiler(self.geo_image, tile_size=tile_size,
    #                           overlapping=overlapping,
    #                           include_nodata=False, scale=scale)
    #     for tile, x, y in tiles:
    #         # Offset is given for the original image
    #         logging.debug("-- tile, x, y | {}, {}, {}".format(tile.shape, x, y))
    #         x = _f(x * 1.0 / scale)
    #         y = _f(y * 1.0 / scale)
    #         logging.debug("--- tile, x, y | {}, {}, {}".format(tile.shape, x, y))
    #
    #         tiled_image[y:y+tile_size[1], x:x+tile_size[0], :] = tile
    #
    #     scaled_img = self.geo_image.get_data(dst_height=h, dst_width=w)
    #     err = float(np.sum(tiled_image - scaled_img))
    #     logging.debug("Err : %f" % err)
    #     self.assertTrue(np.abs(err) < 1e-10)

    def test_tiling_no_overlapping_with_nodata(self):

        def _test(tile_size):
            overlapping = 0
            h = int(np.ceil(self.geo_image.shape[0] * 1.0 /tile_size[1])) * tile_size[1]
            w = int(np.ceil(self.geo_image.shape[1] * 1.0 /tile_size[0])) * tile_size[0]
            nc = self.geo_image.shape[2]
            tiled_image = np.zeros((h, w, nc))
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size, overlapping=overlapping, include_nodata=True)
            for tile, x, y in tiles:
                tiled_image[y:y+tile_size[1], x:x+tile_size[0], :] = tile

            h, w, nc = self.geo_image.shape
            err = float(np.sum(tiled_image[:h, :w, :] - self.geo_image.get_data()))
            logging.debug("Err : %f" % err)
            self.assertTrue(np.abs(err) < 1e-10)

        _test((32, 32))
        _test((40, 40))
        _test((44, 46))
        _test((55, 76))

    def test_tiling_with_overlapping_no_nodata(self):

        def _test(tile_size, overlapping):
            h, w, nc = self.geo_image.shape
            tiled_image = np.zeros((h, w, nc))
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size, overlapping=overlapping, include_nodata=False)
            for tile, x, y in tiles:
                if x == 0:
                    xend = tile_size[0] - overlapping
                else:
                    xend = min(x + tile_size[0], w)

                if y == 0:
                    yend = tile_size[1] - overlapping
                else:
                    yend = min(y + tile_size[1], h)

                tiled_image[y:yend, x:xend, :] = tile

            self.assertTrue(tiled_image.shape == self.geo_image.shape)
            err = float(np.sum(tiled_image - self.geo_image.get_data()))
            logging.debug("Err : %f" % err)
            self.assertTrue(np.abs(err) < 1e-10)

        _test((32, 32), 8); _test((32, 32), 12); _test((32, 32), 13)
        _test((40, 40), 8); _test((40, 40), 11); _test((40, 40), 14)
        _test((44, 46), 7); _test((44, 46), 20); _test((44, 46), 11)
        _test((55, 76), 8); _test((55, 76), 24); _test((55, 76), 15)

    def test_tiling_with_overlapping_with_nodata(self):

        def _test(tile_size, overlapping):
            ny = int(np.ceil((self.geo_image.shape[0]+overlapping) * 1.0 / (tile_size[1] - overlapping)))
            nx = int(np.ceil((self.geo_image.shape[1]+overlapping) * 1.0 / (tile_size[0] - overlapping)))
            h = ny * tile_size[1]
            w = nx * tile_size[0]
            nc = self.geo_image.shape[2]
            tiled_image = np.zeros((h, w, nc))
            tiles = GeoImageTiler(self.geo_image, tile_size=tile_size, overlapping=overlapping, include_nodata=True)
            self.assertTrue(tiles.nx, nx)
            self.assertTrue(tiles.ny, ny)
            for tile, x, y in tiles:
                x += overlapping
                xend = x + tile_size[0]
                y += overlapping
                yend = y + tile_size[1]

                tiled_image[y:yend, x:xend, :] = tile

            h, w, nc = self.geo_image.shape
            x = overlapping
            xend = x + w
            y = overlapping
            yend = y + h
            err = float(np.sum(tiled_image[y:yend, x:xend, :] - self.geo_image.get_data()))
            logging.debug("Err : %f" % err)
            self.assertTrue(np.abs(err) < 1e-10)

        _test((32, 32), 8); _test((32, 32), 12); _test((32, 32), 13)
        _test((40, 40), 8); _test((40, 40), 11); _test((40, 40), 14)
        _test((44, 46), 7); _test((45, 46), 22); _test((44, 46), 11)
        _test((55, 76), 8); _test((55, 76), 24); _test((55, 76), 15)


class TestGeoImageTilerConstSize(TestCase):

    def setUp(self):
        # Create local temp directory
        self.local_temp_folder = tempfile.mkdtemp()

        filepath = os.path.join(self.local_temp_folder, 'test_small_image.tif')
        metadata = {'key_1': 'value_1', 'key_2': "1 2 3", 'key_3': '3'}
        shape = (120, 100, 2)
        depth = 2
        is_complex = False
        geo_transform = (13.60746033, 0.001, 0.0, 50.25013288, 0.0, -0.001)
        _ = create(shape[1], shape[0], shape[2], filepath,
                   depth=depth, is_complex=is_complex,
                   metadata=metadata, geo_transform=geo_transform, epsg=4326)

        self.assertTrue(os.path.exists(filepath))
        self.geo_image = GeoImage(filepath)

    def tearDown(self):
        self.geo_image.close()
        # Delete temp directory
        shutil.rmtree(self.local_temp_folder)

    def test_tiling_no_min_overlapping(self):

        def _test(tile_size):
            tiled_image = np.zeros(self.geo_image.shape)
            tiles = GeoImageTilerConstSize(self.geo_image, tile_size=tile_size, min_overlapping=0)
            self.assertTrue(tiles.nx, int(np.ceil(tiled_image.shape[1] / tile_size[0])))
            self.assertTrue(tiles.ny, int(np.ceil(tiled_image.shape[0] / tile_size[1])))
            for tile, x, y in tiles:
                self.assertTrue(tile.shape[:2] == (tile_size[1], tile_size[0]))
                tiled_image[y:y+tile_size[1], x:x+tile_size[0], :] = tile

            err = float(np.sum(tiled_image - self.geo_image.get_data()))
            logging.debug("Err : %f" % err)
            self.assertTrue(np.abs(err) < 1e-10)

        _test((32, 32))
        _test((40, 40))
        _test((44, 46))
        _test((55, 76))

    def test_tiling_with_min_overlapping(self):

        def _test(tile_size, overlapping):
            h, w, nc = self.geo_image.shape
            tiled_image = np.zeros((h, w, nc))
            tiles = GeoImageTilerConstSize(self.geo_image, tile_size=tile_size, min_overlapping=overlapping)
            for tile, x, y in tiles:
                self.assertTrue(tile.shape[:2] == (tile_size[1], tile_size[0]))
                tiled_image[y:y+tile_size[1], x:x+tile_size[0], :] = tile

            self.assertTrue(tiled_image.shape == self.geo_image.shape)
            err = float(np.sum(tiled_image - self.geo_image.get_data()))
            logging.debug("Err : %f" % err)
            self.assertTrue(np.abs(err) < 1e-10)

        _test((32, 32), 8); _test((32, 32), 12); _test((32, 32), 13)
        _test((40, 40), 8); _test((40, 40), 11); _test((40, 40), 14)
        _test((44, 46), 7); _test((44, 46), 20); _test((44, 46), 11)
        _test((55, 76), 8); _test((55, 76), 24); _test((55, 76), 15)


if __name__ == "__main__":

    suite = TestLoader().loadTestsFromTestCase(TestGeoImageTiler)
    TextTestRunner().run(suite)

    suite = TestLoader().loadTestsFromTestCase(TestGeoImageTilerConstSize)
    TextTestRunner().run(suite)

