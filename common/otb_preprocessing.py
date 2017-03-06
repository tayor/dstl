
# 
# Utils to preprocess images with OTB
#

import os
import sys
import subprocess
import logging
# from image_utils import get_filename

# Configure the OTB path (folder with bin, lib, share)
import yaml


otb_conf_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "otb_conf.yaml"))
assert os.path.exists(otb_conf_file), \
    "OTB configuration file is not found. Modify and rename otb_conf.yaml.example to otb_conf.yaml"
with open(otb_conf_file, 'r') as f:
    cfg = yaml.load(f)
    assert "OTB_PATH" in cfg, "otb_conf.yaml does not contain OTB_PATH"
    OTB_PATH = cfg['OTB_PATH']

assert os.path.exists(os.path.join(OTB_PATH, 'lib', 'python', 'otbApplication.py')), "Orfeo-ToolBox is not found"
os.environ['PATH'] += os.pathsep + os.path.join(OTB_PATH, 'bin')
os.environ['OTB_APPLICATION_PATH'] = os.path.join(OTB_PATH, 'lib', 'otb', 'applications')
sys.path.append(os.path.join(OTB_PATH, 'lib', 'python'))
sys.path.append(os.path.join(OTB_PATH, 'lib'))


def generate_rm_indices(image_id):
    """
    Method to generate radiometric indices (ndvi, gemi, ndwi2, ndti, bi, bi2)
    See https://www.orfeo-toolbox.org/CookBook/Applications/app_RadiometricIndices.html
    """
    app_name = 'otbcli_RadiometricIndices'
    if sys.platform == 'win32':
        app_name += '.bat'
    app_path = os.path.join(OTB_PATH, 'bin', app_name)
    assert os.path.exists(app_path), "OTB application 'RadiometricIndices' is not found"

    in_fname = get_filename(image_id, '17b')
    out_fname = get_filename(image_id, 'multi')
    if os.path.exists(out_fname):
        logging.warn("File '%s' is already existing" % out_fname)
        return 

    list_ch = [
        'Vegetation:NDVI', 
        'Vegetation:GEMI', 
        'Water:NDWI2', 
        'Water:NDTI', 
        'Soil:BI',
        'Soil:BI2'
    ]

    ram = ['-ram', '1024']
    channels = [
        '-channels.red', '6',
        '-channels.green', '3',
        '-channels.blue', '2',
        '-channels.nir', '7',
        '-channels.mir', '17'
    ]

    program = [app_path, '-in', in_fname, '-out', out_fname, '-list']
    program.extend(list_ch)
    program.extend(ram)
    program.extend(channels)

    p = subprocess.Popen(program, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.stdout.readlines()
    err = p.stderr.readlines()
    if len(err) > 0:
        logging.error("RadiometricIndices failed with error : %s" % err)
        print(err)
    p.wait()

    
try:

    import numpy as np
    import otbApplication

    from image_utils import get_image_data  

    # The following line creates an instance of the RadiometricIndices application
    RadiometricIndices = otbApplication.Registry.CreateApplication("RadiometricIndices")
    assert RadiometricIndices is not None, "OTB application 'RadiometricIndices' is not found"
    RadiometricIndices.SetParameterInt("ram", 1024)

    # Vegetation:NDVI - Normalized difference vegetation index (Red, NIR) 
    # Vegetation:TNDVI - Transformed normalized difference vegetation index (Red, NIR) 
    # Vegetation:RVI - Ratio vegetation index (Red, NIR) 
    # Vegetation:SAVI - Soil adjusted vegetation index (Red, NIR) 
    # Vegetation:TSAVI - Transformed soil adjusted vegetation index (Red, NIR) 
    # Vegetation:MSAVI - Modified soil adjusted vegetation index (Red, NIR) 
    # Vegetation:MSAVI2 - Modified soil adjusted vegetation index 2 (Red, NIR) 
    # Vegetation:GEMI - Global environment monitoring index (Red, NIR) 
    # Vegetation:IPVI - Infrared percentage vegetation index (Red, NIR) 
    # Water:NDWI - Normalized difference water index (Gao 1996) (NIR, MIR) 
    # Water:NDWI2 - Normalized difference water index (Mc Feeters 1996) (Green, NIR) 
    # Water:MNDWI - Modified normalized difference water index (Xu 2006) (Green, MIR) 
    # Water:NDPI - Normalized difference pond index (Lacaux et al.) (MIR, Green) 
    # Water:NDTI - Normalized difference turbidity index (Lacaux et al.) (Red, Green) 
    # Soil:RI - Redness index (Red, Green) 
    # Soil:CI - Color index (Red, Green) 
    # Soil:BI - Brightness index (Red, Green) 
    # Soil:BI2


    # Define additional info
    _out_channels_dict = {
        'ndvi': 'Vegetation:NDVI',
        'ndwi': 'Water:NDWI',
        'ndwi2': 'Water:NDWI2',
        'ndpi': 'Water:NDPI',
        'ndti': 'Water:NDTI',
        'mndwi': 'Water:MNDWI',
        'bi': 'Soil:BI',
        'bi2': 'Soil:BI2',
        'ri': 'Soil:RI',
        'ci': 'Soil:CI'
    }
    _channels = ['red', 'green', 'blue', 'nir', 'mir']    


    def compute_rm_indices(image_5b, user_out_channels):
        """
        Method to compute radiometric indices from image with 5 bands: R, G, B, NIR, MIR            
        """
        RadiometricIndices.SetVectorImageFromNumpyArray("in", image_5b)
        list_ch = [_out_channels_dict[c] for c in user_out_channels]
        RadiometricIndices.SetParameterStringList("list", list_ch)
        
        for i, c in enumerate(_channels):
            RadiometricIndices.SetParameterInt('channels.%s' % c, i+1)
        RadiometricIndices.Execute()
        output = RadiometricIndices.GetVectorImageAsNumpyArray('out')
        return output.copy()


    def compute_rm_indices_image(image_id, user_out_channels, user_channels_dict):
        """
        Method to compute radiometric indices using OTB python wrapper
        """        
        # Determine needed image type
        image_types = np.unique([image_type for image_type, _ in user_channels_dict.values()])

        h, w = (0, 0)
        imgs = {}
        for image_type in image_types:
            imgs[image_type] = get_image_data(image_id, image_type)
            h = max(h, imgs[image_type].shape[0])
            w = max(w, imgs[image_type].shape[1])

        # ndarray composed of red, green, blue, nir, mir        
        in_array = np.zeros((h, w, 5), dtype=np.uint16)

        for i, c in enumerate(_channels):
            image_type, band_index = user_channels_dict[c]
            h, w, _ = imgs[image_type].shape
            in_array[:h,:w,i] = imgs[image_type][:,:,band_index]
           
        return compute_rm_indices(in_array, user_out_channels)

except Exception as e:
    print("OTB python wrapper is not available. Error : %s" % e.message)

        
