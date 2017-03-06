
import gdal
import osr
import numpy as np
import argparse
import os

# Project
from common import get_gdal_dtype, get_dtype

"""
    Script to create synthetic images
"""


def create(width, height, nb_bands, filepath, depth=2, is_complex=False, metadata=None,
           geo_transform=(13.60746033, 0.001, 0.0, 50.25013288, 0.0, -0.001),
           epsg=4326):
    """
        Write a synthetic image
    """
    # Write a small test image
    data = np.zeros((height, width, nb_bands), dtype=get_dtype(depth, is_complex))
    step_h = height//10
    step_w = width//10
    for i in range(0, height, step_h):
        for j in range(0, width, step_w):
            data[i:i+step_h, j:j+step_w, :] += np.random.randint(0, 255, size=(1, 1, nb_bands), dtype=np.uint16)

    driver = gdal.GetDriverByName('GTiff')
    dt = get_gdal_dtype(depth, is_complex)
    ds = driver.Create(filepath, width, height, nb_bands, dt)
    for i in range(0, nb_bands):
        ds.GetRasterBand(i+1).WriteArray(data[:, :, i])

    # Add metadata
    if metadata is None:
        ds.SetMetadata({'TEST0': '0', 'TEST1': '123', 'TEST2': 'abc'})
    else:
        ds.SetMetadata(metadata)

    # GeoTransform = [x, dx, dy, y, dx, dy]
    if geo_transform is not None:
        ds.SetGeoTransform(geo_transform)

    if epsg is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())

    src = None
    ds = None
    driver = None
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='/path/to/output/image', help="Output image file path")
    parser.add_argument('width', metavar='image width', type=int, help="Output image width", default=10)
    parser.add_argument('height', metavar='image height', type=int, help="Output image height", default=12)
    parser.add_argument('nb_bands', metavar='image channels', type=int, help="Output image number of bands", default=1)
    args = parser.parse_args()

    if os.path.exists(args.filename):
        os.remove(args.filename)
        os.remove(args.filename + '.aux.xml')
        if os.path.exists(args.filename):
            print("Failed to remove existing file %s" % args.filename)
            exit(1)

    create(args.width, args.height, args.nb_bands, args.filename)
