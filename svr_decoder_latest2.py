from sklearn.model_selection import RepeatedKFold
import argparse
import os
import numpy as np
from os import path
from scipy.stats import pearsonr
import scipy.io as sio
from numpy import arctanh
from tqdm import tqdm
import heapq
from collections import defaultdict
import pickle
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.externals.joblib import Parallel, delayed
from sklearn.preprocessing import scale
from itertools import combinations, permutations
from sklearn.decomposition import PCA



def single_svr_decoder(X, X_test, y):
    reg = linear_model.Lasso(alpha=0.1)
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
    parser.add_argument('--num_select', type=int, default=5000, help="number of voxels to be selected")
    parser.add_argument('--njobs', type=int, default=4, help="number of parallel jobs")
    #parser.add_argument('--twofourthr', action='store_true', help="dataset 243(True) or 384(False)")
    parser.add_argument('--twofourthr', type=int, default=1, help="dataset 243(True) or 384(False)")
    parser.add_argument('--pca', type=int, default=0, help="do pca or not")
    parser.add_argument('--layer', type=int, default=23, help="do pca or not")
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()
    if opt.pca:
        plab = 'pca'
    else:
        plab = 'sel'
    print('loading sentence vector')
    sent_path = path.join('../task_sents', opt.pooling+'.npy')
    #sent_path = path.join('../bert_sents', opt.pooling + '_layer' + str(opt.layer) + '.npy')
    ground_vectors = np.load(sent_path)
    if ground_vectors.shape[0] >= 627:
        if opt.twofourthr == 1:
            ground_vectors = ground_vectors[:243, :]
        else:
            ground_vectors = ground_vectors[243:, :]
    ground_vectors = scale(ground_vectors)
    print(ground_vectors.shape)
    print('loading voxels')
    if opt.pca:
        if opt.twofourthr == 1:
            voxel_path = path.join(opt.voxels, opt.subject, '243_voxel_pca200.npy')
        else:
            voxel_path = path.join(opt.voxels, opt.subject, '384_voxel_pca300.npy')
        if path.exists(voxel_path):
            print('loading pca reduced voxels')
            voxels_new = np.load(voxel_path)
        else:
            if opt.twofourthr == 1:
                voxel_path = path.join(opt.voxels, opt.subject, '243_voxel.npy')
                voxels_new = PCA(200).fit_transform(np.load(voxel_path))
            else:
                voxel_path = path.join(opt.voxels, opt.subject, '384_voxel.npy')
                voxels_new = PCA(300).fit_transform(np.load(voxel_path))
            voxels_new = scale(voxels_new)
            if opt.twofourthr == 1:
                voxel_path = path.join(opt.voxels, opt.subject, '243_voxel_pca200.npy')
            else:
                voxel_path = path.join(opt.voxels, opt.subject, '384_voxel_pca300.npy')
            print('no saved pca voxels, do pca and save to ', voxel_path)
            np.save(voxel_path, voxels_new)
    else:
        if opt.twofourthr == 1:
            voxel_path = path.join(opt.voxels, opt.subject, '243_voxel_sel2000.npy')
        else:
            voxel_path = path.join(opt.voxels, opt.subject, '384_voxel_sel2000.npy')
        if path.exists(voxel_path):
            voxels_new = np.load(voxel_path)
        else:
            print('loading scores')
            sco_path = path.join(opt.scores, opt.pooling, opt.subject, 'score243.mat')
            if not path.exists(sco_path):
                sco_path = path.join(opt.scores, opt.pooling, opt.subject, 'score.mat')
                if not path.exists(sco_path):
                    print('no score for ', opt.pooling, 'loading avg')
                    sco_path = path.join(opt.scores, 'avg', opt.subject, 'score.mat')
            # if opt.twofourthr == 0 and not path.exists(sco_path):
            #     sco_path = path.join(opt.scores, opt.pooling, opt.subject, 'score384.mat')
            # if opt.twofourthr == 1 and not path.exists(sco_path):
            #     sco_path = path.join(opt.scores, opt.pooling, opt.subject, 'score243.mat')
            all_scores = sio.loadmat(sco_path)['scores']
            all_scores = np.sum(all_scores, axis=0)
            print('selecting voxels')
            if opt.twofourthr == 1:
                voxel_path = path.join(opt.voxels, opt.subject, '243_voxel.npy')
            else:
                voxel_path = path.join(opt.voxels, opt.subject, '384_voxel.npy')
            top_ind = heapq.nlargest(opt.num_select, range(len(all_scores)), all_scores.take)
            voxels_new = scale(np.load(voxel_path)[:, top_ind])
            if opt.twofourthr == 1:
                voxel_path = path.join(opt.voxels, opt.subject, '243_voxel_sel5000.npy')
            else:
                voxel_path = path.join(opt.voxels, opt.subject, '384_voxel_sel5000.npy')
            np.save(voxel_path, voxels_new)

    print('loading topic and passage dicts')
    if opt.twofourthr == 1:
        with open(path.join(opt.voxels, 'td243.pkl'), 'rb') as tt:
            topic_dict = pickle.load(tt)
        with open(path.join(opt.voxels, 'pd243.pkl'), 'rb') as tt:
            passa_dict = pickle.load(tt)
    else:
        with open(path.join(opt.voxels, 'td384.pkl'), 'rb') as tt:
            topic_dict = pickle.load(tt)
        with open(path.join(opt.voxels, 'pd384.pkl'), 'rb') as tt:
            passa_dict = pickle.load(tt)

    #random_list = [1, 25, 49]
    pear_list = []
    sper_list = []

    final_accu11= final_accu12= final_accu21=final_accu22=final_accu31=final_accu32 = 0

    kf = RepeatedKFold(n_repeats=1, n_splits=5, random_state=1)
    fold_id = 0
    accu_dict = defaultdict(list)
    wrong_matches = []
    wrong_matches_half = []
    spl_idd = 0
    for tr_indexe, te_indexe in kf.split(voxels_new):
        x_tr = voxels_new[tr_indexe]
        x_te = voxels_new[te_indexe]
        y_tr = ground_vectors[tr_indexe]
        y_te = ground_vectors[te_indexe]
        # x_tr, x_te, y_tr, y_te = train_test_split(voxels_new, ground_vectors, test_size=0.3, random_state=state)
        print('train/test split over, gets:\n',
              'train x:', x_tr.shape,
              'train y:', y_tr.shape,
              'test_x:', x_te.shape,
              'test_y:', y_te.shape)
        _, dim = ground_vectors.shape
        pred_path = path.join('../results', opt.pooling, opt.subject, 
            plab+str(spl_idd)+'layer'+str(opt.layer)+'twfoth'+str(opt.twofourthr)+'dec_las.npy')
        if not path.exists(pred_path):
            print('no saved predctions, training')
            all_predicts = Parallel(n_jobs=opt.njobs)( 
            delayed(single_svr_decoder)(x_tr, x_te, y_tr[:, i]) for i in range(dim))

            # print('we get ', np.array(all_predicts).shape, ' predicts')
            all_predicts = np.array(all_predicts).T
            if not path.exists(path.join('../results', opt.pooling, opt.subject)):
                os.makedirs(path.join('../results', opt.pooling, opt.subject))
            np.save(pred_path, all_predicts)
        else:
            print('found saved predcitions')
            all_predicts = np.load(pred_path)
        spl_idd += 1
        print('we get ', np.array(all_predicts).shape, ' predicts in final')
        # np.save('all_predicts_final.npy', all_predicts)
        fold_id += 1

        p0 = 0
        s0 = 0
        num_pre = len(y_te)

        combo_list = list(combinations(te_indexe, 2))
        index_dict = {kk: vv for vv,kk in enumerate(te_indexe)}
        inv_index_dict = {vv: kk for vv,kk in enumerate(te_indexe)}

        combo_topic = []
        combo_passa = []
        combo_sents = []
        for i, j in combo_list:
            if not topic_dict[i] == topic_dict[j]:
                combo_topic.append([index_dict[i], index_dict[j]])
            else:
                if not passa_dict[i] == passa_dict[j]:
                    combo_passa.append([index_dict[i], index_dict[j]])
                elif not i == j:
                    combo_sents.append([index_dict[i], index_dict[j]])

        # topic classification
        clas_accu1 = 0
        clas_accu2 = 0
        for i, j in combo_topic:
            rii, _ = pearsonr(all_predicts[i], y_te[i])
            rij, _ = pearsonr(all_predicts[i], y_te[j])
            rjj, _ = pearsonr(all_predicts[j], y_te[j])
            rji, _ = pearsonr(all_predicts[j], y_te[i])
            if rii + rjj > rij + rji:
                clas_accu1 += 1
            if rii > rij and rjj > rji:
                clas_accu2 += 1
            if rii > rij:
                accu_dict[inv_index_dict[i]].append(1)
            else:
                accu_dict[inv_index_dict[i]].append(0)
            if rjj > rji:
                accu_dict[inv_index_dict[j]].append(1)
            else:
                accu_dict[inv_index_dict[j]].append(0)

        final_accu11 += clas_accu1 / len(combo_topic)
        final_accu12 += clas_accu2 / len(combo_topic)
        #pear_list.append(p0 / y_te.shape[0])
        # sper_list.append(s0/y_te.shape[0])

        # passage classification
        clas_accu1 = 0
        clas_accu2 = 0
        for i, j in combo_passa:
            rii, _ = pearsonr(all_predicts[i], y_te[i])
            rij, _ = pearsonr(all_predicts[i], y_te[j])
            rjj, _ = pearsonr(all_predicts[j], y_te[j])
            rji, _ = pearsonr(all_predicts[j], y_te[i])
            if rii + rjj > rij + rji:
                clas_accu1 += 1
            if rii > rij and rjj > rji:
                clas_accu2 += 1
            # r, _ = spearmanr(i, j)
            # s0 += r
            # print('classification accuracy a', clas_accu1 / len(combo_list))
            # print('classification accuracy b', clas_accu2 / len(combo_list))
            # print('spearson correlation ', s0/y_te.shape[0])
        final_accu21 += clas_accu1 / len(combo_passa)
        final_accu22 += clas_accu2 / len(combo_passa)

        # passage classification
        clas_accu1 = 0
        clas_accu2 = 0
        for i, j in combo_sents:
            rii, _ = pearsonr(all_predicts[i], y_te[i])
            rij, _ = pearsonr(all_predicts[i], y_te[j])
            rjj, _ = pearsonr(all_predicts[j], y_te[j])
            rji, _ = pearsonr(all_predicts[j], y_te[i])
            if rii + rjj > rij + rji:
                clas_accu1 += 1
            if rii > rij and rjj > rji:
                clas_accu2 += 1
            # r, _ = spearmanr(i, j)
            # s0 += r
            # print('classification accuracy a', clas_accu1 / len(combo_list))
            # print('classification accuracy b', clas_accu2 / len(combo_list))
            # print('spearson correlation ', s0/y_te.shape[0])
        final_accu31 += clas_accu1 / len(combo_sents)
        final_accu32 += clas_accu2 / len(combo_sents)

    if opt.twofourthr == 1:
        dict_name = path.join('./saved_pickles', '_'.join(
            [opt.pooling, opt.subject + '_score243_decode']) + '.pkl')
    else:
        dict_name = path.join('./saved_pickles', '_'.join(
            [opt.pooling, opt.subject + '_score384_decode']) + '.pkl')

    with open(dict_name, 'wb') as ss:
        pickle.dump(accu_dict, ss)



    print('final results', opt.subject, opt.pooling)
    print(opt.subject, opt.pooling, opt.layer, plab, 'topic the final classfication 1 accuracy', final_accu11 / 5)
    # print(plab, 'topic the final classfication 2 accuracy', final_accu12 / 5)
    print(opt.subject, opt.pooling, opt.layer, plab, 'passa the final classfication 1 accuracy', final_accu21 / 5)
    # print(plab, 'passa the final classfication 2 accuracy', final_accu22 / 5)
    print(opt.subject, opt.pooling, opt.layer, plab, 'sents the final classfication 1 accuracy', final_accu31 / 5)
    # print(plab, 'sents the final classfication 2 accuracy', final_accu32 / 5)
