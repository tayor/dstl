#
# Code from Cogitae repository
#

import os
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import jaccard_similarity_score

target_labels = ['Buildings', 'Misc manmade struct', 'Road', 'Track', 'Trees', 'Crops', 'Waterway', 'Standing water', 'Vehicle Large', 'Vehicle Small']


def calc_jacc_vals(model, img, msk, range_thres=range(5, 10), batch_size=4):
        
    prd = model.predict(img, batch_size=batch_size)
    avg, trs = [], []
    for i in range(prd.shape[1]):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])

        m, b_tr = 0, 0
        for j in range_thres:
            tr = j / 10.0
            pred_binary_mask = t_prd > tr

            jk = jaccard_similarity_score(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / len(avg)
    return score, trs, avg


def calc_jacc(model, img=None, msk=None, range_thres=range(5, 10), batch_size=4):
    score, trs, avg = calc_jacc_vals(model, img=img, msk=msk, range_thres=range_thres, batch_size=batch_size)
    for i, m, b_tr in zip(range(len(avg)), avg, trs):
        print(i, m, b_tr, target_labels[i])
    return score, trs


class CalcJaccard(Callback):
    """
    Callback that computes Jaccard score 
    """
    def __init__(self, X_val, Y_val, period=1, model_str='', weight_dir = None, batch_size=4):
        super(CalcJaccard, self).__init__()
        self.period = period
        self.img = X_val
        self.msk = Y_val
        self.model_str = model_str
        self.weight_dir = weight_dir
        self.batch_size = batch_size
        self.best_loss = np.inf
        self.best_jac_int = 0.
        self.best_jac = 0.


    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        do_compute = (epoch + 1) % self.period == 0
        reason = 'cycle'

        if self.best_jac_int < logs['jaccard_coef_int']:
            self.best_jac_int = logs['jaccard_coef_int']
            do_compute = True
            reason = 'jac_int'

        if self.best_jac < logs['jaccard_coef']:
            self.best_jac = logs['jaccard_coef']
            do_compute = True
            reason = 'jac'

        if self.best_loss > logs['loss']:
            self.best_loss = logs['loss']
            do_compute = True
            reason = 'loss'

        if do_compute:
            score, trs, avg = calc_jacc_vals(self.model, img=self.img, msk=self.msk, range_thres=range(5,10), batch_size=self.batch_size)
            print( 'val jk', score)
            if self.weight_dir is not None:
                self.model.save(os.path.join(self.weight_dir, '{}_{}_{}_{:.6f}.h5'.format(self.model_str, epoch, reason, score)))
                if logs is not None:
                    logs['jk'] = score
                    for i, val in enumerate(avg):
                         logs['jk_' + str(i+1)] = val
