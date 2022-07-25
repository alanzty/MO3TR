# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmdet.datasets.builder import PIPELINES
import random


@PIPELINES.register_module()
class MatchInstances(object):
    """Matching objects on a pair of images.

    Args:
        skip_nomatch (bool, optional): Whether skip the pair of image
        during training when there are no matched objects. Default
        to True.
    """

    def __init__(self, skip_nomatch=True):
        self.skip_nomatch = skip_nomatch

    def _match_gts(self, instance_ids, ref_instance_ids):
        """Matching objects according to ground truth `instance_ids`.

        Args:
            instance_ids (ndarray): of shape (N1, ).
            ref_instance_ids (ndarray): of shape (N2, ).

        Returns:
            tuple: Matching results which contain the indices of the
            matched target.
        """
        ins_ids = list(instance_ids)
        ref_ins_ids = list(ref_instance_ids)
        match_indices = np.array([
            ref_ins_ids.index(i) if (i in ref_ins_ids and i >= 0) else -1
            for i in ins_ids
        ])
        ref_match_indices = np.array([
            ins_ids.index(i) if (i in ins_ids and i >= 0) else -1
            for i in ref_ins_ids
        ])
        return match_indices, ref_match_indices

    def __call__(self, results):
        if len(results) != 2:
            raise NotImplementedError('Only support match 2 images now.')

        match_indices, ref_match_indices = self._match_gts(
            results[0]['gt_instance_ids'], results[1]['gt_instance_ids'])
        nomatch = (match_indices == -1).all()
        if self.skip_nomatch and nomatch:
            return None
        else:
            results[0]['gt_match_indices'] = match_indices.copy()
            results[1]['gt_match_indices'] = ref_match_indices.copy()

        return results


@PIPELINES.register_module()
class MatchInstancesMO3TR(MatchInstances):
    def __call__(self, results):
        # Rearange the results in temporal order
        if random.random() < 0.5:
            results.sort(key=lambda res: int(res['filename'].split("/")[-1].replace(".jpg", "")), reverse=True)
        else:
            results.sort(key=lambda res: int(res['filename'].split("/")[-1].replace(".jpg", "")), reverse=False)
        match_indices, ref_match_indices = self._match_gts(
            results[-1]['gt_instance_ids'], results[-2]['gt_instance_ids'])
        nomatch = (match_indices == -1).all()
        if self.skip_nomatch and nomatch:
            return None
        else:
            results[-1]['gt_match_indices'] = match_indices.copy()
            results[-2]['gt_match_indices'] = ref_match_indices.copy()

        for result in results:
            if 'gt_match_indices' not in result:
                result['gt_match_indices'] = []
        return results