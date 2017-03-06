# -*- coding:utf-8 -*-

# Python
import os
import logging

# Numpy
import numpy as np

# GDAL
import gdal
import osgeo.osr
import gdalconst

from common import get_gdal_dtype, gdal_to_numpy_datatype

logger = logging.getLogger(__name__)


class GeoImage:
    """
    Wrapper structure to GDAL Dataset
    Requires GDAL >= 1.11

    Usage :
        gimage = GeoImage('path/to/geo/image/filename')

        # Display image dimensions:
        print gimage.shape # (height, width, nb_bands)

        # Display image geo extent in lat/lon if gdal geo transformation is initialized
        print gimage.get_extent
        # [[top-left_x, top-left_y], [top-right_x, top-right_y],
           [bottom-left_x, bottom-left_y], [bottom-right_x, bottom-right_y]]
        # or None

        # Display image metadata if exists
        print gimage.metadata
        # {'key_1': 'value_1', '<domain_name>__key_2': 'value_2', ...
        #  'BAND_<i>__key_3': 'value_3', 'BAND_<i>__<domain_name>__key_3': 'value_3'}
        # or {}

        # Display image gcps and its project:
        print gimage.gcps
        # [[pixel_1_x, pixel_1_y, gcp_1_x, gcp_1_y, gcp_1_z], [pixel_2_x, pixel_2_y, gcp_2_x, gcp_2_y, gcp_2_z], ...]
        # or []
        print gimage.gcp_projection
        #
        # or ""

        # Get image data :
        # - whole image
        data = gimage.get_data()
        # - ROI from (x, y) of size (w,h) without rescale
        # - Note: x, y, w, h can be such that the request is out of bounds. Such pixels have np.nan values
        # or specified by 'nodata_value' argument
        data = gimage.get_data([x, y, w, h])

        # - ROI from (x, y) of size (w,h) with a rescale conserving aspect ratio
        data = gimage.get_data([x, y, w, h], 512)
        # data.shape = (512*h/w, 512, gimage.shape[2])

        # - ROI from (x, y) of size (w,h) with a rescale not conserving aspect ratio
        data = gimage.get_data([x, y, w, h], 512, 200)
        # data.shape = (200, 512, gimage.shape[2])


        # Transform pixel to lat/lon coordinates if gdal transformation exists
        points = np.array([[0,0], [100, 20], ...])
        geo_points = gimage.transform(points)
        # [[lon_0, lat_0], [lon_1, lat_1], ...]
        # or raise AssertionError


        # if image contains subdataset : (hdf5, netcdf files)
        print gimage.has_subsets()
        # True or False
        print gimage.subset_count()
        # 0, 1, 2, 3, 4, ...
        gimage_subset_0 = gimage.get_subset_geoimage(0)


        # Specific information can be acquired directly from gdal dataset
        ds = gimage.get_dataset()

    """

    def __init__(self, filename=""):
        """
        Initialize GeoImage
        :param filename: input image filename
        """
        self.close()
        if len(filename) > 0:
            self.open(filename)

    def close(self):
        self._dataset = None
        self.shape = None
        self.projection = ""
        self.geo_extent = None
        self.metadata = None
        self.gcps = None
        self.gcp_projection = ""
        self._pix2geo = None
        self._subsets = []
            
    @staticmethod
    def from_dataset(dataset):
        gimage = GeoImage()
        gimage._open(dataset)
        return gimage

    def has_subsets(self):
        return len(self._subsets) > 0

    def subset_count(self):
        return len(self._subsets)

    def get_subset_geoimage(self, index):
        assert 0 <= index < len(self._subsets), \
            "Subset index %i is out of bounds [0. %i]" % index % len(self._subsets)
        return self._subsets[index]

    def _open(self, dataset):
        subsets = dataset.GetMetadata('SUBDATASETS')
        if subsets is not None:
            for item in subsets.keys():
                if "NAME" in item:
                    subset = gdal.Open(subsets[item], gdalconst.GA_ReadOnly)
                    if subset is None:
                        logger.error("Failed to open the subset '%s'" % str(subsets[item]))
                        continue
                    subset_image = GeoImage.from_dataset(subset)
                    self._subsets.append(subset_image)

        self._dataset = dataset
        self.shape = (self._dataset.RasterYSize, self._dataset.RasterXSize, self._dataset.RasterCount)
        if self._setup_geo_transformers():
            # get geo extent of the image:
            self.geo_extent = self._compute_geo_extent()

        # get metadata
        self.metadata = self._fetch_metadata()

        # get gcps
        self._setup_gcps()

    def open(self, filename):
        """
            Method to load image from filename
            - self.geoExtent is a numpy array of 4 points (long, lat) : [[left,top], [right,top], [right,bottom], [left,bottom]]
            - self.metadata is an array of image metadata
        """
        dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
        assert dataset is not None, "Failed to open the file: %s " % filename
        self._open(dataset)

    def get_image_resolution_in_degrees(self):
        assert self._dataset is not None, "Dataset is None"
        geotransform = self._dataset.GetGeoTransform()
        return abs(geotransform[1]), abs(geotransform[5])

    def transform(self, points, option="pix2geo"):
        """
            Method to transform points a) from geo to pixels, b) from pixels to geo
            Option can take values : "pix2geo" and "geo2pix"
            points should be a numpy array of type [[x1,y1],[x2,y2],...]

            Return numpy array of transformed points.
            In case of 'geo2pix' output value can be [-1, -1] which means point is not in the image

        """
        assert self._pix2geo is not None, "Geo transformer is None"
        points = np.array(points)
        _assert_numpy_points(points)

        if option is "pix2geo":
            out = np.zeros((len(points), 2))
            for count, pt in enumerate(points):
                g = self._pix2geo.TransformPoint(0, float(pt[0]), float(pt[1]), 0.0)
                out[count, 0] = g[1][0]
                out[count, 1] = g[1][1]
            return out
        elif option is "geo2pix":
            out = np.zeros((len(points), 2), dtype=np.int16)-1

            def f(xx):
                return abs(round(xx))

            w = self.shape[1]
            h = self.shape[0]
            for count, pt in enumerate(points):
                g = self._pix2geo.TransformPoint(1, float(pt[0]), float(pt[1]), 0.0)
                x = f(g[1][0])
                y = f(g[1][1])
                if 0 <= x < w and 0 <= y < h:
                    out[count, 0] = x
                    out[count, 1] = y
            return out
        else:
            return None

    def _fetch_metadata(self):
        assert self._dataset is not None, "Dataset is None"

        def __fetch_metadata(gdal_object, prefix_str=None):

            if not hasattr(gdal_object, "GetMetadataDomainList"):
                # GetMetadataDomainList exists since gdal 1.11
                domains = None
            else:
                domains = gdal_object.GetMetadataDomainList()

            if domains is None:
                domains = ''
            metadata = {}
            for d in domains:
                if len(d) == 0 and prefix_str is None:
                    # copy simply the metadata
                    metadata.update(gdal_object.GetMetadata(d))
                else:
                    # should prefix the key name with domain:
                    md = gdal_object.GetMetadata(d)
                    if len(md) == 0 or not isinstance(md, dict):
                        continue
                    nmd = {}
                    for key in md.keys():
                        new_key = "" if prefix_str is None else prefix_str.upper()
                        new_key += key if len(d) == 0 else d.upper() + '__' + key
                        nmd[new_key] = md[key]
                    metadata.update(nmd)

            if not hasattr(gdal_object, "GetRasterBand"):
                return metadata

            nb_bands = gdal_object.RasterCount
            for i in range(1, nb_bands+1):
                band = gdal_object.GetRasterBand(i)
                if band is None:
                    logger.error("Raster band %i is None" % i)
                    continue
                metadata.update(__fetch_metadata(band, "BAND_%i__" % i))
            return metadata

        return __fetch_metadata(self._dataset)

    def _setup_gcps(self):
        self.gcp_projection = self._dataset.GetGCPProjection()
        count = self._dataset.GetGCPCount()
        if count > 0:
            assert len(self.gcp_projection) > 0, "GCP projection is empty, but there is %i of GCPs found" % count
            gcps = self._dataset.GetGCPs() # gcps is a tuple of <osgeo.gdal.GCP>
            # Setup transformer from gcp projection to lat/lon
            srs = osgeo.osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            options = ['SRC_SRS=' + self.gcp_projection, 'DST_SRS=' + srs.ExportToWkt()]
            transformer = gdal.Transformer(self._dataset, None, options)
            assert transformer.this is not None, "No geo transformer found to transform GCP"
            self.gcps = []
            for gcp in gcps:
                # print gcp.Id, gcp.Info, gcp.GCPX, gcp.GCPY, gcp.GCPZ, gcp.GCPPixel, gcp.GCPLine
                g = transformer.TransformPoint(0, float(gcp.GCPX), float(gcp.GCPY), float(gcp.GCPZ))
                self.gcps.append([gcp.GCPPixel, gcp.GCPLine, g[1][0], g[1][1], g[1][2]])
            self.gcps = tuple(self.gcps)

    def _setup_geo_transformers(self):
        assert self._dataset is not None, "Dataset is None"
        if self._pix2geo is not None:
            return False

        self.projection = self._dataset.GetProjection()
        # Init pixel to geo transformer :
        srs = osgeo.osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dstSRSWkt = srs.ExportToWkt()
        options = ['DST_SRS=' + dstSRSWkt]

        transformer = gdal.Transformer(self._dataset, None, options)
        if transformer.this is None:
            logger.warn("No geo transformer found")
            return False

        self._pix2geo = transformer
        return True

    def _compute_geo_extent(self):
        assert self._dataset is not None, "Dataset is None"

        if self._pix2geo is None:
            return None

        # transform 4 image corners
        w = self._dataset.RasterXSize
        h = self._dataset.RasterYSize
        pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        return self.transform(pts, "pix2geo")

    def get_data(self, src_rect=None, dst_width=None, dst_height=None, nodata_value=0, dtype=None, select_bands=None):
        """
        Method to read data from image
        :param src_rect: is source extent in pixels : [x,y,w,h] where (x,y) is top-left corner. Can be None and whole image extent is used.
        :param dst_width is the output array width. Can be None and src_rect[2] (width) is used.
        :param dst_height is the output array heigth. Can be None and src_rect[3] (height) is used.
        :param nodata_value: value to fill out of bounds pixels with.
        :param dtype: force type of returned numpy array
        :param select_bands: tuple of band indices (zero-based) to select from dataset, e.g. [0, 3, 4]. 
        Returns a numpy array
        """
        assert self._dataset is not None, "Dataset is None"
        assert self.shape[2] > 0, "Dataset has no bands" 
        
        if select_bands is not None:
            assert isinstance(select_bands, list) or isinstance(select_bands, tuple), \
                "Argument select_bands should be a tuple or list"
            available_bands = list(range(self.shape[2]))
            for index in select_bands:
                assert index in available_bands, \
                    "Index {} from select_bands is outside of available bands: {}".format(index, available_bands)

        if src_rect is None:
            src_req_extent = [0, 0, self.shape[1], self.shape[0]]
            src_extent = src_req_extent
        else:
            src_req_extent = intersection(src_rect, [0, 0, self.shape[1], self.shape[0]])
            src_extent = src_rect

        if src_req_extent is None:
            logger.warn('source request extent is None')
            return None
        
        if dst_width is None and dst_height is None:
            dst_extent = [src_extent[2], src_extent[3]]
        elif dst_height is None:
            h = int(dst_width * src_extent[3] * 1.0 / src_extent[2])
            dst_extent = [dst_width, h]
        elif dst_width is None:
            w = int(dst_height * src_extent[2] * 1.0 / src_extent[3])
            dst_extent = [w, dst_height]
        else:
            dst_extent = [dst_width, dst_height]

        scale_x = dst_extent[0] * 1.0 / src_extent[2]
        scale_y = dst_extent[1] * 1.0 / src_extent[3]
        req_scaled_w = int(min(np.ceil(scale_x * src_req_extent[2]), dst_extent[0]))
        req_scaled_h = int(min(np.ceil(scale_y * src_req_extent[3]), dst_extent[1]))

        r = [int(np.floor(scale_x * (src_req_extent[0] - src_extent[0]))),
             int(np.floor(scale_y * (src_req_extent[1] - src_extent[1]))),
             req_scaled_w,
             req_scaled_h]

        band_indices = range(self.shape[2]) if select_bands is None else select_bands
        nb_bands = len(band_indices) 

        if dtype is None:
            datatype = gdal_to_numpy_datatype(self._dataset.GetRasterBand(1).DataType)
            datatype = update_dtype(datatype, nodata_value)
        else:
            datatype = dtype

        out = np.empty((dst_extent[1], dst_extent[0], nb_bands), dtype=datatype)
        out.fill(nodata_value)

        for i, index in enumerate(band_indices):
            band = self._dataset.GetRasterBand(index+1)
            data = band.ReadAsArray(src_req_extent[0],
                                    src_req_extent[1],
                                    src_req_extent[2],
                                    src_req_extent[3],
                                    r[2], r[3])
            out[r[1]:r[1]+r[3], r[0]:r[0]+r[2], i] = data[:, :]

        return out
    
    def get_filename(self):
        """
        Method to get image file name without path
        """
        return os.path.basename(self.get_filepath())

    def get_filepath(self):
        """
        Method to get image file name with path
        """
        assert self._dataset is not None, "Dataset is None"
        filelist = self._dataset.GetFileList()
        return filelist[0] if len(filelist) > 0 else ""

    def get_dataset(self):
        """
        Method to get image gdal dataset
        """
        return self._dataset


def intersection(r1, r2):
    """
    Helper method to obtain intersection of two rectangles
    r1 = [x1,y1,w1,h1]
    r2 = [x2,y2,w2,h2]
    returns [x,y,w,h]
    """
    assert len(r1) == 4 and len(r2) == 4, "Rectangles should be defined as [x,y,w,h]"

    rOut = [0, 0, 0, 0]
    rOut[0] = max(r1[0], r2[0])
    rOut[1] = max(r1[1], r2[1])
    rOut[2] = min(r1[0]+r1[2]-1, r2[0]+r2[2]-1) - rOut[0] + 1
    rOut[3] = min(r1[1]+r1[3]-1, r2[1]+r2[3]-1) - rOut[1] + 1

    if rOut[2] <= 0 or rOut[3] <= 0:
        return None
    return rOut


def from_ndarray(data):
    """
    Method instanciates GeoImage object from ndarray `data` using virtual gdal driver
    :param data: ndarray of shape (h, w, nc)
    :return: GeoImage instance
    """
    assert isinstance(data, np.ndarray), "Input should be a Numpy array"
    assert len(data.shape) == 3, "Input data should be of shape (h, w, nc)"
    h, w, nc = data.shape
    # Create a synthetic gdal dataset
    driver = gdal.GetDriverByName('MEM')
    gdal_dtype = get_gdal_dtype(data[0, 0, 0].itemsize,
                                data[0, 0, 0].dtype == np.complex64 or
                                data[0, 0, 0].dtype == np.complex128)
    ds = driver.Create('', w, h, nc, gdal_dtype)
    for i in range(0, nc):
        ds.GetRasterBand(i+1).WriteArray(data[:, :, i])

    geo_image = GeoImage.from_dataset(ds)
    return geo_image


def points_to_envelope(points):
    """
    Method to convert array of points [[x1,y1],[x2,y2],...] into
    (minX,maxX,minY,maxY)
    """
    _assert_numpy_points(points)
    return points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()


def _assert_numpy_points(points):
    assert isinstance(points, np.ndarray), "Input should be a Numpy array"
    assert len(points.shape) == 2 and points.shape[1] == 2, \
        "Points should be a Numpy array of shape : (nbPts,2)"


def update_dtype(dtype, v):
    """
    Update dtype to be conform with nodata_value
    """
    if np.array(v).astype(dtype) != np.array(v):
        return type(v)
    return dtype



