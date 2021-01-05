from __future__ import print_function, absolute_import
import time

import torch
import numpy as np
import torch.nn.functional as F

from PIL import ImageFile
from utils.meters import AverageMeter

from .ranking import cmc, mean_ap
from .cnn import extract_cnn_feature

ImageFile.LOAD_TRUNCATED_IMAGES = True


def extract_features(model, data_loader, print_freq=1):
    model.eval()
    batch_time = AverageMeter('Process Time', ':6.3f')
    data_time = AverageMeter('Test Date Time', ':6.3f')

    features = []
    labels = []

    end = time.time()
    for i, (imgs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for output, pid in zip(outputs, targets):
            features.append(output)
            labels.append(pid)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, old_features=None, query=None, gallery=None, metric=None):
    if old_features is None:
        old_features = features
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(features)
        y = torch.cat(old_features)
        x = x.view(n, -1)
        y = y.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
            y = metric.transform(y)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, n).t()
        dist = dist - 2 * torch.mm(x, y.t())
        return dist

    x = torch.cat([features[i].unsqueeze(0) for i, _ in enumerate(query)], 0)
    y = torch.cat([old_features[i].unsqueeze(0) for i, _ in enumerate(gallery)], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 cmc_topk=(1, 5), compute_meanap=False):
    if query is not None and gallery is not None:
        query_ids = [label for label in query]
        gallery_ids = [label for label in gallery]

    # Compute mean AP
    if compute_meanap:
        meanap = mean_ap(distmat, query_ids=query_ids, gallery_ids=gallery_ids)
        print('Mean AP: {:4.1%}'.format(meanap))

    # Compute CMC scores
    cmc_scores = cmc(distmat, query_ids, gallery_ids, topk=5,
                     single_gallery_shot=False,
                     first_match_break=True)

    for k in cmc_topk:
        print('top-{:<4}{:12.1%}'
              .format(k, cmc_scores[k - 1]))

    # Use the cmc top-1 and top-5 score for validation criterion
    return cmc_scores[0], cmc_scores[4]


class Evaluator:
    def __init__(self, model, old_model=None):
        super(Evaluator, self).__init__()
        self.model = model
        self.old_model = old_model

    def evaluate(self, data_loader, query=None, gallery=None, metric=None):

        features, labels = extract_features(self.model, data_loader)

        if query is not None and gallery is not None:
            distmat = pairwise_distance(features, query, gallery, metric=metric)
            return evaluate_all(distmat, query=query, gallery=gallery)

        if self.old_model is not None:
            old_features, _ = extract_features(self.old_model, data_loader)
            distmat = pairwise_distance(features, old_features=old_features,
                                        query=None, gallery=None, metric=metric)
        else:
            distmat = pairwise_distance(features, old_features=None,
                                        query=None, gallery=None, metric=metric)
        return evaluate_all(distmat, query=labels, gallery=labels)


class ClassifierGenerator:
    def __init__(self, model, cls_num=1000):
        self.model = model
        self.cls_num = cls_num

    def generate_classifier(self, data_loader):
        features, labels = extract_features(self.model, data_loader)
        feature_dim = features[0].size(0)
        saved_classifier = np.zeros((self.cls_num, feature_dim))
        feature_list = [[] for _ in range(self.cls_num)]
        for feature, label in zip(features, labels):
            saved_classifier[label] += feature.numpy()
        labels = torch.tensor(labels)
        # remove duplicate labels
        non_dup_labels = torch.unique(labels)
        cnt = {int(label): 0 for label in non_dup_labels}
        for label in non_dup_labels:
            cnt[int(label)] = np.sum((labels == label).numpy())
            if cnt[int(label)] != 0:
                saved_classifier[label] /= cnt[int(label)]
        return saved_classifier



