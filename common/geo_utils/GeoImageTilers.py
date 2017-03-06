# -*- coding:utf-8 -*-

# Python
import logging

# Numpy
import numpy as np

# Project
from GeoImage import GeoImage

logger = logging.getLogger(__name__)


class BaseGeoImageTiler(object):
    """
    Base class to tile a GeoImage
    See the implementations
        - GeoImageTiler
        - GeoImageTilerConstSize

    Essential parameters:
    :param geo_image: instance of GeoImage
    :param tile_size: output tile size in pixels
    :param scale: (integer) tile resolution scale, default 1, corresponds to a scaling 
        between requested ROI from the image and returned tile.
        For example, when scale=2 returned tiles represent two times larger zone in pixels than the same zone in the image, e.g
        source image rectangle `[x, y, scale*tile_size_x, scale*tile_size_x]` is sampled to a tile of size `tile_size`        
    """
    def __init__(self, geo_image, tile_size=(1024, 1024), scale=1):
        assert isinstance(geo_image, GeoImage), logger.error("geo_image argument should be instance of GeoImage")
        assert len(tile_size) == 2, logger.error("tile_size argument should be a tuple (sx, sy)")

        self._geo_image = geo_image
        self.tile_size = tile_size
        self.scale = scale
        self._index = 0
        self._maxIndex = 0
        self.nodata_value = 0

    def __iter__(self):
        return self

    def get_lin_index(self):
        return self._index - 1

    def compute_geo_extent(self, tile_offset, tile_size):
        """
        Compute tile geo extent
        :param tile_offset: [x, y] tile offset (top-left coordinates) in the original image
        :param tile_size: [width, height] of the tile
        """
        x, y = tile_offset
        tile_size[0] *= self.scale
        tile_size[1] *= self.scale
        points = np.array([[x, y],
                           [x + tile_size[0] - 1, y],
                           [x + tile_size[0] - 1, y + tile_size[1] - 1],
                           [x, y + tile_size[1] - 1]])
        return self._geo_image.transform(points)

    def _get_current_tile_extent(self):
        raise Exception("Not implemented")

    def next(self):
        """
            Method to get current tile
        """
        if self._index < 0 or self._index >= self._maxIndex:
            raise StopIteration()

        # Define ROI to extract
        extent, x_tile_index, y_tile_index = self._get_current_tile_extent()
        # extent = [xoffset, yoffset, tile_size_x, tile_size_y]
        scaled_extent = [floor_int(e * self.scale) for e in extent]
        logger.debug("{}/{} = ({},{}) | extent={}".format(self._index, self._maxIndex, x_tile_index, y_tile_index, scaled_extent))
        
        # Extract data
        dst_width = ceil_int(extent[2])  # ceil need when tile size is computed (image_size/scale - offset)
        dst_height = ceil_int(extent[3])
        logger.debug("dst_width={}, dst_height={}".format(dst_width, dst_height))
        
        data = self._geo_image.get_data(scaled_extent,
                                        dst_width=dst_width,
                                        dst_height=dst_height,
                                        nodata_value=self.nodata_value)
        logger.debug("data.shape={}, {},{}".format(data.shape, extent[0], extent[1]))
        # ++
        self._index += 1

        return data, scaled_extent[0], scaled_extent[1]

    __next__ = next


class GeoImageTiler(BaseGeoImageTiler):
    """

        Helper class to iterate over GeoImage
        Tiles of maximum size `(tile_size[0], tile_size[1])` are extracted with a four-sided overlapping (overlapping)

        There are two options to produce tile:
        a) `include_nodata=True`: All tiles have the same size `tile_size` and nodata values are filled with `nodata_value`.
        For example, the first top-left tile starts outside the image
        <--- tile_size[0] -->
      overlapping     overlapping
        <--->           <--->
         ____________________
        |NDV|NDV NDV NDV|NDV|
        |---*************---|
        |NDV*   *   *   *   |
        |NDV*   *   *   *   |
        |NDV*   *   *   *   |
        |---*************---|
        |NDV|___________|___|

        and the last bottom-right tile can look like this :

        <--- tile_size[0] -->
      overlapping     overlapping
        <--->           <--->
         ____________________
        |   |       |NDV NDV|
        |---*********-------|
        |   *   *   *NDV NDV|
        |   *   *   *NDV NDV|
        |   *   *   *NDV NDV|
        |---*********-------|
        |NDV NDV NDV NDV NDV|

        In the case (a) the tile offset is computed as
        ```
        x_tile_offset = i*(tile_size[0] - overlapping) - overlapping
        y_tile_offset = j*(tile_size[1] - overlapping) - overlapping
        ```

        b) `include_nodata=False`: All tiles have the same size `tile_size`, except boundary tiles.
        For example, the first top-left tile starts at (0,0) of the image

        tile_size[0] - overlapping
        <--------------->
                   overlapping
                    <--->
        *************---|
        *   *   *   *   |
        *   *   *   *   |
        *   *   *   *   |
        *************---|
        |___________|___|

        Somewhere in the middle of the image, the tile has the size `tile_size` :

        <--- tile_size[0] -->
      overlapping     overlapping
        <--->           <--->
         ____________________
        |   |           |   |
        |---*************---|
        |   *   *   *   *   |
        |   *   *   *   *   |
        |   *   *   *   *   |
        |---*************---|
        |___|___________|___|

        and the last bottom-right tile is reduced to stop at the last pixel:

        <--- tile_size[0] -->
      overlapping
        <--->
         ____________
        |   |       |
        |---********|
        |   *   *   |
        |   *   *   |
        |   *   *   |
        |---********|


        In the case (b) the tile offset is computed as
        ```
        x_tile_offset = { i == 0 :  0,
                        { i > 0  : i*(tile_size[0] - overlapping) - overlapping
        y_tile_offset = { j == 0 :  0,
                        { j > 0  : j*(tile_size[1] - overlapping) - overlapping
        ```

        Usage :

            gimage = GeoImage('path/to/file')
            tiles = GeoImageTiler(gimage, tile_size=(1024, 1024), overlapping=256)

            for tile, xoffset, yoffset in tiles:
                assert instanceof(tile, np.ndarray), "..."


        If the option include_nodata=False, then at the boundaries the outside image overlapping is not added.
        For example, the 1st top-left tile looks like
         tile_size[0] - overlapping
            <--------------->
                       overlapping
                        <--->
            *************---|
            *   *   *   *   |
            *   *   *   *   |
            *   *   *   *   |
            *************---|
            |___________|___|

    """
    def __init__(self, geo_image, tile_size=(1024, 1024), overlapping=256, 
                    include_nodata=False, nodata_value=0, scale=1):

        super(GeoImageTiler, self).__init__(geo_image, tile_size, scale)
        assert overlapping >= 0 and 2*overlapping < min(tile_size[0], tile_size[1]), \
            logger.error("overlapping argument should be less than half of the min of tile_size")

        self.overlapping = overlapping
        h = self._geo_image.shape[0] * 1.0 / self.scale
        w = self._geo_image.shape[1] * 1.0 / self.scale
        self.nx = GeoImageTiler._compute_number_of_tiles(self.tile_size[0], w, overlapping)
        self.ny = GeoImageTiler._compute_number_of_tiles(self.tile_size[1], h, overlapping)
        self._maxIndex = self.nx * self.ny
        self.include_nodata = include_nodata
        self.nodata_value = nodata_value

    @staticmethod
    def _compute_number_of_tiles(tile_size, image_size, overlapping):
        """
            Method to compute number of overlapping tiles for a given image size
            n = ceil((imageSize + overlapping)/(tileSize - overlapping ))
            imageSize :  [01234567891] (isFourSided=true), tileSize=6, overlapping=2
            tile   0  :[xx0123]
                   1  :    [234567]
                   2  :        [67891x]
                   3  :            [1xxxxx]
              n = ceil ( (11+2) / (6 - 2) ) = 4

            imageSize :  [012345678901234] (isFourSided=true), tileSize=7, overlapping=2
            tile   0  :[xx01234]
                   1  :     [3456789]
                   2  :          [8901234]
                   3  :               [34xxxxx]
              n = ceil ( (16+2) / (7 - 2) ) = 4
        """
        return ceil_int((image_size + overlapping)*1.0/(tile_size - overlapping))

    def _get_current_tile_extent(self):
        """
        isFourSided = true:
        tileSize = 6, overlapping = 1

        offset:  0    4   9    14   19
        Image:  [----------------------]
        Tiles: [x....O]                ]
                [   [O....O]           ]
                [        [O....O]      ]
                [             [O....O] ]
                [                  [O..xxx]

        offset(i) = {i == 0: 0,
                    {i > 0: i * (tileSize - overlapping) - overlapping

        size(i) = {i == 0: tileSize - overlapping,
                  {i > 0: offset(i) + tileSize < imageWidth ? tileSize: imageWidth - offset(i)

        bufferOffset(i) = {i == 0: dataPtr + overlapping
                          {i > 0: dataPtr + 0
        """
        image_width = self._geo_image.shape[1] * 1.0 / self.scale
        image_height = self._geo_image.shape[0] * 1.0 / self.scale
        x_tile_index = self._index % self.nx
        y_tile_index = floor_int(self._index * 1.0 / self.nx)

        x_tile_size = self.tile_size[0] 
        y_tile_size = self.tile_size[1] 
        x_tile_offset = x_tile_index * (self.tile_size[0] - self.overlapping) - self.overlapping
        y_tile_offset = y_tile_index * (self.tile_size[1] - self.overlapping) - self.overlapping

        if not self.include_nodata:
            if x_tile_index == 0:
                x_tile_offset = 0
                x_tile_size -= self.overlapping
            if y_tile_index == 0:
                y_tile_offset = 0
                y_tile_size -= self.overlapping
            x_tile_size = min(image_width - x_tile_offset, x_tile_size)
            y_tile_size = min(image_height - y_tile_offset, y_tile_size)

        return [x_tile_offset, y_tile_offset, x_tile_size, y_tile_size], x_tile_index, y_tile_index


class GeoImageTilerConstSize(BaseGeoImageTiler):
    """
        Helper class to iterate over GeoImage
        Tiles of size (tile_size[0], tile_size[1]) are extracted with a four-sided overlapping
        starting from the top-left corner and all tiles have the same size without going outbound of the image.
        This is done computing the horizontal and vertical overlappings to minimize the number of tiles needed to
        cover the image. User can also specify a minimum overlapping `min_overlapping`.

        For example, tiling can look like this:
          tile 0      tile 2      tile 4
        |<------>|  |<------>|  |<------>|
                tile 1      tile 3      tile 5
              |<------>|  |<------>|  |<------>|
        |<------------------------------------>|
                        IMAGE

        Usage :

            gimage = GeoImage('path/to/file')
            tiles = GeoImageTilerConstSize(gimage, tile_size=(1024, 1024), min_overlapping=256)

            for tile, xoffset, yoffset in tiles:
                assert instanceof(tile, np.ndarray), "..."

    """
    def __init__(self, geo_image, tile_size=(1024, 1024), min_overlapping=256, scale=1):

        super(GeoImageTilerConstSize, self).__init__(geo_image, tile_size, scale)
        assert 0 <= min_overlapping < min(tile_size[0], tile_size[1]), \
            logger.error("minimal overlapping should be between 0 and min tile_size")

        self.min_overlapping = min_overlapping
        h = self._geo_image.shape[0] * 1.0 / self.scale
        w = self._geo_image.shape[1] * 1.0 / self.scale
        
        self.nx = GeoImageTilerConstSize._compute_number_of_tiles(self.tile_size[0], w, min_overlapping)
        self.ny = GeoImageTilerConstSize._compute_number_of_tiles(self.tile_size[1], h, min_overlapping)
        self.float_overlapping_x = GeoImageTilerConstSize._compute_float_overlapping(self.tile_size[0], w,
                                                                                     self.nx)
        self.float_overlapping_y = GeoImageTilerConstSize._compute_float_overlapping(self.tile_size[1], h,
                                                                                     self.ny)
        self._maxIndex = self.nx * self.ny

    @staticmethod
    def _compute_number_of_tiles(tile_size, image_size, min_overlapping):
        """
            Method to compute number of overlapping tiles for a given image size
            n = ceil(image_size / (tile_size - min_overlapping))
        """
        return ceil_int(image_size * 1.0 / (tile_size - min_overlapping))

    @staticmethod
    def _compute_float_overlapping(tile_size, image_size, n):
        """
            Method to float overlapping

            delta = tile_size * n - image_size
            overlapping = delta / (n - 1)
        """
        return (tile_size * n - image_size) * 1.0 / (n - 1.0)

    def _get_current_tile_extent(self):
        """

        offset(i) = round(i * (tileSize - float_overlapping_x))
          size(i) = tileSize

        :return: current tile extent as
            [x_tile_offset, y_tile_offset, x_tile_size, y_tile_size], x_tile_index, y_tile_index
        """
        x_tile_index = self._index % self.nx
        y_tile_index = floor_int(self._index * 1.0 / self.nx)
        x_tile_size = self.tile_size[0]
        y_tile_size = self.tile_size[1]
        x_tile_offset = int(np.round(x_tile_index * (self.tile_size[0] - self.float_overlapping_x)))
        y_tile_offset = int(np.round(y_tile_index * (self.tile_size[1] - self.float_overlapping_y)))

        return [x_tile_offset, y_tile_offset, x_tile_size, y_tile_size], x_tile_index, y_tile_index


def floor_int(x):
    return int(np.floor(x))


def ceil_int(x):
    return int(np.ceil(x))
