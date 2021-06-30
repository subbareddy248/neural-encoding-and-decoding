import argparse
import csv
import numpy as np
from os import path
from scipy.stats import pearsonr
import scipy.io as sio
from numpy import arctanh
from tqdm import tqdm
import heapq
import pickle
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import scale
from itertools import combinations, permutations
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA

def single_svr_decoder(reg, X, X_test, y):
    #reg = linear_model.RidgeCV(alphas=(0.1, 0.5, 1, 10), cv=5)
    #reg = MLPRegressor(hidden_layer_sizes =(300,100), random_state=1)
    #reg = SVR(kernel='linear', C=1e3)
    #reg = linear_model.SGDRegressor()
    #reg = linear_model.Lasso(alpha=0.1)
    reg.fit(X, y)
    # print('fitting over')
    y_test = reg.predict(X_test)
    y_test = y_test #[:, np.newaxis]
    return y_test

def parse_args():
    parser = argparse.ArgumentParser()
    ###### parameters for voxels loading
    parser.add_argument('--voxels', type=str, default='../voxels', help="folder for voxels")
    parser.add_argument('--scores', type=str, default='../results', help="pre-calculated selection scores")
    parser.add_argument('--subject', type=str, default='M02', help="subject being tested")
    parser.add_argument('--pooling', type=str, default='avg', help="composition method used")
    parser.add_argument('--sents_vec', type=str, default='../sents',
                        help="folder for sentence vecs")
    # parser.add_argument('--num_select', type=int, default=5000, help="number of voxels to be selected")
    parser.add_argument('--regressor', type=str, default='las', help="regressor to be used")
    parser.add_argument('--njobs', type=int, default=8, help="number of parallel jobs")
    parser.add_argument('--roi', type=int, default=3, help="roi to be covered")
    parser.add_argument('--atlas', type=int, default=3, help="roi to be covered")
    parser.add_argument('--pca', type=int, default=300, help="dimension reduction")
    return parser.parse_args()

def combo(combos, topic_dict, passa_dict):
    # num_pre0 = 188
    # combos = [i for i in range(num_pre0)]
    combo_list = list(combinations(combos, 2))
    print('all combinations', len(combo_list))
    combo_topic = []
    combo_passa = []
    combo_sents = []
    ind_dict = {}
    for ia, ib in enumerate(combos):
        ind_dict[ib] = ia

    for i, j in combo_list:
        if not topic_dict[i] == topic_dict[j]:
            combo_topic.append([ind_dict[i], ind_dict[j]])
        else:
            if not passa_dict[i] == passa_dict[j]:
                combo_passa.append([ind_dict[i], ind_dict[j]])
            elif not i == j:
                combo_sents.append([ind_dict[i], ind_dict[j]])
    print('sentences in different topic', len(combo_topic))
    print('sentences in save topic but different passages', len(combo_passa))
    print('sentences in save passage ', len(combo_sents))
    return combo_topic, combo_passa, combo_sents

def sim_combo(indexes):
    ind_dict = defaultdict(int, {i: j for i, j in enumerate(indexes)})
    #ind_dict = {i: j for i, j in enumerate(indexes)}
    combos = []
    for i in ind_dict.keys():
        for j in ind_dict.keys():
            if not i == j:
                combos.append([i, j])
    return combos, ind_dict

if __name__ == '__main__':
    opt = parse_args()
    print('loading sentence vector')
    #sent_path = path.join(opt.sents_vec, opt.pooling, 'allsents_vec.npy')
    #ground_vectors = scale(np.load(sent_path)[243:])
    sent_path = path.join('../task_sents', opt.pooling+'.npy')
    ground_vectors = scale(np.load(sent_path)[243:])

    print('loading voxels')
    voxel_npy =  path.join(opt.voxels, opt.subject, '384_voxel.npy')
    if not path.exists(voxel_npy):
        voxel_path = path.join(opt.voxels, opt.subject, 'data_384sentences.mat')
        voxels = sio.loadmat(voxel_path)['examples_passagesentences']
        np.save(voxel_npy, voxels)
    else:
        voxels = np.load(voxel_npy)
    # print('loading topic and passage dicts')
    # with open(path.join(opt.voxels, 'topic_dict.pkl'), 'rb') as tt:
    #     topic_dict = pickle.load(tt)
    # with open(path.join(opt.voxels, 'passa_dict.pkl'), 'rb') as tt:
    #     passa_dict = pickle.load(tt)
    # print('loading scores')
    # sco_path = path.join(opt.scores, opt.pooling, opt.subject, 'score.mat')
    # all_scores = sio.loadmat(sco_path)['scores']
    # all_scores = np.sum(all_scores, axis=0)
    # print('selecting topk infoprmative voxels')
    # top_ind = heapq.nlargest(opt.num_select, range(len(all_scores)), all_scores.take)
    # voxels_new = scale(voxels[:, top_ind])
    print('selecting brain atlas voxels')
    meta_path = path.join(opt.voxels, opt.subject, 'data_384sentences.mat')
    metas = sio.loadmat(meta_path)['meta']
    maski = np.array(metas['roiColumns'][0][0][0][opt.atlas][opt.roi][0]).T[0]
    #atlas_name = metas['atlases'][0, 0][0][opt.atlas]
    roi_name = metas['rois'][0][0][0][opt.atlas][opt.roi][0][0]
    #maski = list(maski[0])
    voxels_new = scale(voxels[:, maski])
    print('voxels in this', roi_name, len(maski))

    kf = RepeatedKFold(n_repeats=1, n_splits=5, random_state=1)
    #random_list = [1, 25, 49]
    pear_list = []
    sper_list = []
    # final_accu11 = final_accu12 = final_accu21 = final_accu22 = final_accu31 = final_accu32 = 0
    # final_pears1 = final_pears2 = final_pears3 = 0
    final_accu11 = final_accu12 = 0
    final_pears1 = final_pears2 = 0

    fold_state = 0
    accu_dict = defaultdict(list)
    wrong_matches = []
    wrong_matches_half = []
    for tr_indexe, te_indexe in kf.split(voxels_new):
        # splits_name = path.join(opt.voxels, str(state)+'split.mat')
        # tr_indexe = sio.loadmat(splits_name)['trainInd'][0]
        # te_indexe = sio.loadmat(splits_name)['testInd'][0]
        # tr_indexe = tr_indexe - 1
        # te_indexe = te_indexe - 1
        # print('start combination')
        # combo_topic, combo_passa, combo_sents = combo(list(te_indexe), topic_dict, passa_dict)
        x_tr = ground_vectors[tr_indexe]
        x_te = ground_vectors[te_indexe]

        y_tr = voxels_new[tr_indexe]
        y_te = voxels_new[te_indexe]
        # x_tr, x_te, y_tr, y_te = train_test_split(voxels_new, ground_vectors, test_size=0.3, random_state=state)
        print('train/test split over, gets:\n',
              'train x:', x_tr.shape,
              'train y:', y_tr.shape,
              'test_x:', x_te.shape,
              'test_y:', y_te.shape)
        _, dim = voxels_new.shape
        regressor_dicts = {
            'mlp': MLPRegressor(hidden_layer_sizes =(dim,100), random_state=1),
            'rid':linear_model.RidgeCV(alphas=(0.1, 0.5, 1, 10), cv=5),
            'las':linear_model.Lasso(alpha=0.1)
        }
        logo_dict = {
            'mlp':'mlp',
            'rid':'',
            'las':'las'
        }
        reg = regressor_dicts[opt.regressor]
        logo = logo_dict[opt.regressor]
        predict_path = path.join(opt.scores, opt.pooling, opt.subject, logo+str(fold_state)+'_atl'+str(opt.atlas)+'_roi'+str(opt.roi)+'_enc384.npy')
        fold_state += 1

        # start normal predicting
        #if not path.exists(predict_path):
        print(' start training')
        all_predicts = Parallel(n_jobs=opt.njobs)(
        delayed(single_svr_decoder)(reg, x_tr, x_te, y_tr[:, i]) for i in range(dim))
        #np.save(predict_path, all_predicts)
        #else:
            #print('found saved predicts, loading', predict_path)
            #all_predicts = np.load(predict_path)

            # start pca predicting if required

        # print('we get ', np.array(all_predicts).shape, ' predicts')
        all_predicts = np.array(all_predicts).T
        print('we get ', np.array(all_predicts).shape, ' predicts in final')
        # np.save('all_predicts_final.npy', all_predicts)
        p0 = 0
        s0 = 0
        num_pre = len(y_te)

        # topic classification
        clas_accu1 = 0
        clas_accu2 = 0
        pears = 0
        combo_index, ind_dict = sim_combo(te_indexe)

        pearss = []
        pv = 0
        for ci, cj in combo_index:
            rii, _ = pearsonr(all_predicts[ci], y_te[ci])
            rij, _ = pearsonr(all_predicts[ci], y_te[cj])
            rjj, _ = pearsonr(all_predicts[cj], y_te[cj])
            rji, _ = pearsonr(all_predicts[cj], y_te[ci])
            if rii + rjj > rij + rji:
                clas_accu1 += 1
            if rii>rij and rjj>rji:
                clas_accu2 += 1
            if rii>rij:
                accu_dict[ind_dict[ci]].append(1)
            else:
                accu_dict[ind_dict[ci]].append(0)
            if rjj>rji:
                accu_dict[ind_dict[cj]].append(1)
            else:
                accu_dict[ind_dict[cj]].append(0)
            # r, _ = spearmanr(i, j)
            # s0 += r
            pears += rii
        final_pears1 += pears / len(combo_index)
        final_accu11 += clas_accu1 / len(combo_index)
        final_accu12 += clas_accu2 / len(combo_index)


    dict_name = path.join('./saved_pickles', '_'.join([opt.pooling, opt.subject, 'atlas_'+str(opt.atlas)+'_roi_'+str(opt.roi)+'score_384']) + '.pkl')
    with open(dict_name,'wb') as ss:
        pickle.dump(accu_dict, ss)
    #fold_state = 0

    #print(opt.subject, opt.pooling)
    sum_accu = {i: sum(j)/len(j) for i,j in accu_dict.items()}
    csv_name = path.join('./saved_pickles', '_'.join([opt.pooling, opt.subject, 'atlas_'+str(opt.atlas)+'_roi_'+str(opt.roi)+'sum_sco_384']) + '.csv')
    w = csv.writer(open(csv_name, 'w'))
    for kk, vv in sum_accu.items():
        w.writerow([kk, vv])
    print(opt.subject, opt.pooling, opt.atlas, opt.roi, roi_name,'topic the final classfication 1 accuracy', final_accu11 / 5, ' final pears ', final_pears1)
    print(opt.subject, opt.pooling, opt.atlas, opt.roi, roi_name, 'topic the final classfication 2 accuracy', final_accu12 / 5, ' final pears ', final_pears1)


    if opt.pca > 0:
        final_accu11 = final_accu12 = 0
        final_pears1 = final_pears2 = 0

        print("*"*10 + "start pca series"+ "*"*10)

        pca_sent_path = path.join('../jon_encoding', opt.pooling+'_pca.npy')
        if path.exists(pca_sent_path):
            pca_ground_vectors = np.load(pca_sent_path)
        else:
            pca_ground_vectors = PCA(opt.pca).fit_transform(ground_vectors)
            np.save(pca_sent_path, pca_ground_vectors)

        fold_state = 0
        accu_dict = defaultdict(list)
        for tr_indexe, te_indexe in kf.split(voxels_new):
            # combo_topic, combo_passa, combo_sents = combo(list(te_indexe), topic_dict, passa_dict)
            x_tr = pca_ground_vectors[tr_indexe]
            x_te = pca_ground_vectors[te_indexe]
            y_tr = voxels_new[tr_indexe]
            y_te = voxels_new[te_indexe]
            # x_tr, x_te, y_tr, y_te = train_test_split(voxels_new, ground_vectors, test_size=0.3, random_state=state)
            print('train/test split over, gets:\n',
                  'train x:', x_tr.shape,
                  'train y:', y_tr.shape,
                  'test_x:', x_te.shape,
                  'test_y:', y_te.shape)
            _, dim = voxels_new.shape
            regressor_dicts = {
                'mlp': MLPRegressor(hidden_layer_sizes=(dim, 100), random_state=1),
                'rid': linear_model.RidgeCV(alphas=(0.1, 0.5, 1, 10), cv=5),
                'las': linear_model.Lasso(alpha=0.1)
            }
            logo_dict = {
                'mlp': 'mlp',
                'rid': '',
                'las': 'las'
            }
            reg = regressor_dicts[opt.regressor]
            logo = logo_dict[opt.regressor]
            predict_path = path.join(opt.scores, opt.pooling, opt.subject,
                                     logo + str(fold_state) +'_atl' + str(opt.atlas) + '_roi' + str(opt.roi) + '_enc_pca384.npy')
            fold_state += 1

            # start normal predicting
            if not path.exists(predict_path):
                print('no existing predictions, start training')
                all_predicts = Parallel(n_jobs=opt.njobs)(
                    delayed(single_svr_decoder)(reg, x_tr, x_te, y_tr[:, i]) for i in range(dim))
                np.save(predict_path, all_predicts)
            else:
                print('found saved predicts, loading', predict_path)
                all_predicts = np.load(predict_path)
            # print('we get ', np.array(all_predicts).shape, ' predicts')
            all_predicts = np.array(all_predicts).T
            print('we get ', np.array(all_predicts).shape, ' predicts in final')
            # np.save('all_predicts_final.npy', all_predicts)
            p0 = 0
            s0 = 0
            num_pre = len(y_te)

            # topic classification
            clas_accu1 = 0
            clas_accu2 = 0
            pears = 0
            combo_index, ind_dict = sim_combo(te_indexe)

            pearss = []
            pv = 0
            for ci, cj in combo_index:
                rii, _ = pearsonr(all_predicts[ci], y_te[ci])
                rij, _ = pearsonr(all_predicts[ci], y_te[cj])
                rjj, _ = pearsonr(all_predicts[cj], y_te[cj])
                rji, _ = pearsonr(all_predicts[cj], y_te[ci])
                if rii + rjj > rij + rji:
                    clas_accu1 += 1
                else:
                    wrong_matches.append((ind_dict[ci], ind_dict[cj]))
                if rii > rij and rjj > rji:
                    clas_accu2 += 1
                if rii > rij:
                    accu_dict[ind_dict[ci]].append(1)
                else:
                    accu_dict[ind_dict[ci]].append(0)
                    wrong_matches_half.append((ind_dict[ci], ind_dict[cj]))

                if rjj > rji:
                    accu_dict[ind_dict[cj]].append(1)
                else:
                    accu_dict[ind_dict[cj]].append(0)
                    wrong_matches_half.append((ind_dict[ci], ind_dict[cj]))
                # r, _ = spearmanr(i, j)
                # s0 += r
                pears += rii
            final_pears1 += pears / len(combo_index)
            final_accu11 += clas_accu1 / len(combo_index)
            final_accu12 += clas_accu2 / len(combo_index)


        dict_name = path.join('./saved_pickles', '_'.join([opt.pooling, opt.subject, 'atlas_'+str(opt.atlas)+'_roi_'+str(opt.roi)+'score_pca384']) + '.pkl')
        with open(dict_name, 'wb') as ss:
            pickle.dump(accu_dict, ss)
        wrong_name = path.join('./saved_pickles', '_'.join([opt.pooling, opt.subject, 'atlas_'+str(opt.atlas)+'_roi_'+str(opt.roi)+'wrong_mat_pca384']) + '.pkl')
        wrong_name_half = path.join('./saved_pickles', '_'.join([opt.pooling, opt.subject, 'atlas_'+str(opt.atlas)+'_roi_'+str(opt.roi)+'wrong_half_pca384']) + '.pkl')
        with open(wrong_name, 'wb') as ss:
            pickle.dump(wrong_matches, ss)
        with open(wrong_name_half, 'wb') as ss:
            pickle.dump(wrong_matches_half, ss)
        # fold_state = 0

        print(opt.subject, opt.pooling)
        sum_accu = {i: sum(j) / len(j) for i, j in accu_dict.items()}
        csv_name = path.join('./saved_pickles', '_'.join([opt.pooling, opt.subject, 'atlas_'+str(opt.atlas)+'_roi_'+str(opt.roi)+'sum_sco_pca384']) + '.csv')
        w = csv.writer(open(csv_name, 'w'))
        for kk, vv in sum_accu.items():
            w.writerow([kk, vv])
        print(opt.subject, opt.pooling, opt.atlas, opt.roi,'after pca, topic the final classfication 1 accuracy', final_accu11 / 5, ' final pears ', final_pears1)
        print(opt.subject, opt.pooling, opt.atlas, opt.roi,'after pca, topic the final classfication 2 accuracy', final_accu12 / 5, ' final pears ', final_pears1)
