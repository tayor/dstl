import logging 

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon, Patch

from shapely.geometry import box
from shapely.affinity import translate
from shapely.validation import explain_validity

from data_utils import ORDERED_LABEL_IDS, LABELS
from image_utils import normalize


def plt_st(l1,l2):
    plt.figure(figsize=(l1,l2))


def scale_percentile(matrix, q_min=0.5, q_max=99.5):
    is_gray = False
    if len(matrix.shape) == 2:
        is_gray = True
        matrix = matrix.reshape(matrix.shape + (1,))
    matrix = (255*normalize(matrix, q_min, q_max)).astype(np.uint8)
    if is_gray:
        matrix = matrix.reshape(matrix.shape[:2])
    return matrix


def display_img_1b(img_1b_data, roi=None, no_colorbar=False, **kwargs):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        img_1b_data = img_1b_data[y:yh,x:xw]

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'

    if 'clim' not in kwargs:
        vmin = np.percentile(img_1b_data, 0.1)
        vmax = np.percentile(img_1b_data, 99.9)
        kwargs['clim'] = [vmin, vmax]
    plt.imshow(img_1b_data, **kwargs)
    if not no_colorbar:
        plt.colorbar(orientation='horizontal')


def display_img_3b(img_3b_data, roi=None, **kwargs):
    nc = img_3b_data.shape[2]
    assert nc < 4, "Input data should have at most 3 bands"
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        img_3b_data = img_3b_data[y:yh,x:xw,:]
    ax_array = []
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'

    for i in range(nc):
        ax = plt.subplot(1,nc,i+1)
        ax_array.append(ax)
        vmin = np.percentile(img_3b_data[:, :, i], 0.1)
        vmax = np.percentile(img_3b_data[:, :, i], 99.9)
        kwargs['clim'] = [vmin, vmax]
        plt.imshow(img_3b_data[:,:,i], **kwargs)
        plt.colorbar(orientation='horizontal')
        plt.title("Channel %i" % i)


def display_img_8b(img_ms_data, roi=None, **kwargs):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        img_ms_data = img_ms_data[y:yh,x:xw,:]
    ax_array = []
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray'
    for i in range(8):
        ax = plt.subplot(2,4,i+1)
        ax_array.append(ax)
        vmin = np.percentile(img_ms_data[:, :, i], 0.1)
        vmax = np.percentile(img_ms_data[:, :, i], 99.9)
        kwargs['clim'] = [vmin, vmax]
        plt.imshow(img_ms_data[:, :, i],  **kwargs)
        plt.colorbar(orientation='horizontal')
        plt.title("Channel %i" % i)
    return ax_array


def display_labels(label_img, roi=None, ax_array=None, show_legend=True, **kwargs):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        label_img = label_img[y:yh,x:xw]
    
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.get_cmap('Paired', 11)
    cmap = kwargs['cmap'] if not isinstance(kwargs['cmap'], str) else plt.get_cmap(kwargs['cmap'])
   
    if ax_array is None:
        ax_array = [plt.gca()]
        
    for ax in ax_array:
        ax.imshow(label_img, **kwargs)
        
    if show_legend:
        legend_handles = []
        for i in range(len(ORDERED_LABEL_IDS)):
            class_type = LABELS[ORDERED_LABEL_IDS[i]]
            legend_handles.append(Patch(color=cmap(i), label='{}'.format(class_type)))
            
        index = 0 if len(ax_array) == 1 else len(ax_array)//2 - 1
        ax = ax_array[index]
        ax.legend(handles=legend_handles, 
                  bbox_to_anchor=(1.05, 1), 
                  loc=2, 
                  borderaxespad=0.,
                  fontsize='x-small',
                  title='Objects:',
                  framealpha=0.3)
    

# Buggy method to plot polygons. No internal polygons drawn
#
# def display_polygons(polygons, roi=None, ax_array=None, show_legend=True):
#     if roi is not None:
#         # roi is [minx, miny, maxx, maxy]
#         b = box(*roi)
#     if ax_array is None:
#         ax_array = [plt.gca()]
#
#     cmap = plt.get_cmap('Paired', len(LABELS))
#
#     def _draw_polygon(ax, polygon, class_type):
#         _add_mpl_polygon(ax, polygon.exterior, cmap(class_type))
#         for lin_ring in polygon.interiors:
#             _add_mpl_polygon(ax, lin_ring, 'k')
#
#     def _add_mpl_polygon(ax, linear_ring, color):
#         mpl_poly = Polygon(np.array(linear_ring), color=color, lw=0, alpha=0.5)
#         ax.add_patch(mpl_poly)
#
#     legend_handles = []
#     for class_type in ORDERED_LABEL_IDS:
#         if class_type not in polygons:
#             continue
#         for i, polygon in enumerate(polygons[class_type]):
#             draw_polygon = roi is None
#             if roi is not None and polygon.intersects(b):
#                 if not polygon.is_valid:
#                     logging.warn("Polygon (%i, %i) is not valid: %s" % (class_type, i, explain_validity(polygon)))
#                     continue
#                 polygon = polygon.intersection(b)
#                 polygon = translate(polygon, -roi[0], -roi[1])
#                 draw_polygon = True
#             if draw_polygon:
#                 if polygon.type == 'MultiPolygon':
#                     for p in polygon:
#                         for ax in ax_array:
#                             _draw_polygon(ax, p, class_type)
#                 else:
#                     for ax in ax_array:
#                         _draw_polygon(ax, polygon, class_type)
#
#         legend_handles.append(Patch(color=cmap(class_type), label='{} ({})'.format(LABELS[class_type], len(polygons[class_type]))))
#
#     for ax in ax_array:
#         ax.relim()
#         ax.autoscale_view()
#     if show_legend:
#         index = 0 if len(ax_array) == 1 else len(ax_array)//2 - 1
#         ax = ax_array[index]
#         ax.legend(handles=legend_handles,
#                   bbox_to_anchor=(1.05, 1),
#                   loc=2,
#                   borderaxespad=0.,
#                   fontsize='x-small',
#                   title='Objects:',
#                   framealpha=0.3)