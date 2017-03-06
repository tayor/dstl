#   
# Helper methods to visualize layer outputs
#
import logging

import numpy as np
import matplotlib.pylab as plt

import keras.backend as K


def get_layer_output_func(layer_name, model):
    inputs = [K.learning_phase()] + model.inputs
    output_layer = model.get_layer(name=layer_name)
    outputs = output_layer.output
    return K.function(inputs, [outputs])
    
    
def compute_layer_output(input_data, layer_output_f):
    return layer_output_f([0] + [input_data])


def compute_layer_outputs(input_data, model, layer_output_f_dict={}, layer_names=None):
    """
    Method to compute (all or only those specified by `layer_names`) layer outputs on `input_data` for a given `model`
    :return: tuple of pairs: [("layer_name_1", ndarray), ...]
    """
    if layer_names is None:
        inputs_outputs = model.input_layers
        inputs_outputs.extend(model.output_layers)
        layer_names = [layer.name for layer in model.layers if layer not in inputs_outputs]        
    else:
        all_layer_names = [layer.name for layer in model.layers]
        assert set(layer_names) & set(all_layer_names) == set(layer_names), \
            "Items {} of layer_names are not in model".format(set(layer_names) - set(layer_names))

    layer_outputs = []
    for layer_name in layer_names:
        logging.info("-- %s" % layer_name)
        if layer_name not in layer_output_f_dict:
            layer_output_f_dict[layer_name] = get_layer_output_func(layer_name, model)
        layer_outputs.append((layer_name, compute_layer_output(input_data, layer_output_f_dict[layer_name])))
    return layer_outputs


def display_layer_output(layer_name, layer_output, layer_plots_limit=None, **kwargs):
    assert len(layer_output.shape) == 3, "Layer output should be 3D : (n_channels, height, width) "
    plt.suptitle("%s" % layer_name)
    nc = layer_output.shape[0] if layer_plots_limit is None else min(layer_plots_limit, layer_output.shape[0])
    n_cols = 4 if nc >= 4 else nc
    n_rows = int(np.floor(nc / n_cols))
    for i in range(nc):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(layer_output[i,:,:], interpolation='none')
        plt.colorbar()
        plt.title("%i" % i)
