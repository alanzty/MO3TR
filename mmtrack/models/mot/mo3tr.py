import torch
from mmdet.models import build_detector

from mmtrack.core import outs2results, results2outs
from ..builder import MODELS, build_motion, build_tracker
from .base import BaseMultiObjectTracker
from scipy.optimize import linear_sum_assignment
import random
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import collections
import motmetrics as mm
import numpy as np
import pandas as pd
from torch import nn
import math


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, max_len=40):
        super(PositionalEncoding1D, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


def match_gts(instance_ids, ref_instance_ids):
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


@MODELS.register_module()
class MO3TR(BaseMultiObjectTracker):
    def __init__(self,
                 detector=None,
                 tracker=None,
                 motion=None,
                 init_cfg=None,
                 fn_rate=0.4,
                 fp_rate=0.1,
                 dup_rate=0.1,
                 noise=1e-4,
                 fpdb_rate=0.1,
                 grad="separate"):
        super().__init__(init_cfg)
        self.random_fn_rate = fn_rate
        self.random_fp_rate = fp_rate
        self.random_dup_rate = dup_rate
        self.max_track_query = 100
        self.noise = noise
        self.grad = grad
        self.fpdb_rate = fpdb_rate
        if detector is not None:
            self.detector = build_detector(detector)

        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_instance_ids,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_instance_ids,
                      ref_gt_match_indices,
                      **kwargs):

        track_prev, loss_det = self.forward_prev(ref_img, ref_img_metas, ref_gt_bboxes, ref_gt_labels, ref_gt_instance_ids, ref_gt_match_indices, gt_instance_ids)
        img, img_metas, gt_bboxes, gt_labels, gt_instance_ids, gt_match_indices = img[:1], img_metas[:1], gt_bboxes[:1], gt_labels[:1], gt_instance_ids[:1], gt_match_indices[:1]

        outs = self.forward_current(img, img_metas, track_prev)

        loss_inputs = outs[1:] + (gt_bboxes, gt_labels, gt_instance_ids, track_prev, img_metas)

        losses = self.detector.bbox_head.loss_mo3tr(*loss_inputs)
        if self.grad == "separate":
            losses = {**losses, **loss_det}
        return losses

    def forward_det(self, ref_img, ref_img_metas):
        batch_input_shape = tuple(ref_img[0].size()[-2:])
        for img_meta in ref_img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        # Forward Propagate of First Image
        x = self.detector.extract_feat(ref_img)
        hs, outputs_classes, outputs_coords = self.detector.bbox_head.forward_prev(x, ref_img_metas)
        prev_hs, prev_cls, prev_boxes = hs[-1].detach().clone(), outputs_classes[-1].detach().clone(), outputs_coords[-1].detach().clone()
        return prev_hs, prev_cls, prev_boxes

    def forward_prev(self,
                     ref_img,
                     ref_img_metas,
                     ref_gt_bboxes,
                     ref_gt_labels,
                     ref_gt_instance_ids,
                     ref_gt_match_indices=None,
                     gt_instance_ids=None):

        batch_input_shape = tuple(ref_img[0].size()[-2:])
        for img_meta in ref_img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        ref_img, ref_gt_bboxes = self.detector._preprocess(ref_img, ref_gt_bboxes)

        # Forward Propagate of First Image
        x = self.detector.extract_feat(ref_img)
        hs, outputs_classes, outputs_coords = self.detector.bbox_head.forward_prev(x, ref_img_metas)

        if self.grad == "separate":
            prev_hs, prev_cls, prev_boxes = hs[-1].detach().clone(), outputs_classes[-1].detach().clone(), outputs_coords[-1].detach().clone()
            loss_inputs = (outputs_classes, outputs_coords, None, None, ref_gt_bboxes, ref_gt_labels, ref_img_metas)
            loss_det = self.detector.bbox_head.loss(*loss_inputs)
            loss_det = {"det_" + key: val for key, val in loss_det.items()}
        else:
            prev_hs, prev_cls, prev_boxes = hs[-1], outputs_classes[-1], outputs_coords[-1]
            loss_det = {}

        bs = len(ref_img_metas)
        track_query_matched_ids, track_query_hs_embeds, track_query_boxes, track_query_masks = [], [], [], []
        for i in range(bs):
            assign_result = self.detector.bbox_head.assigner.assign(prev_boxes[i], prev_cls[i], ref_gt_bboxes[i], ref_gt_labels[i], ref_img_metas[i], None)
            prev_out_ind = torch.arange(len(assign_result.gt_inds))
            prev_out_ind = prev_out_ind[assign_result.gt_inds != 0]
            prev_target_ind = assign_result.gt_inds[assign_result.gt_inds != 0] - 1

            prev_out_ind_unmatched = torch.ones(len(assign_result.gt_inds), device=prev_out_ind.device, dtype=torch.bool)
            prev_out_ind_unmatched[prev_out_ind] = 0
            prev_out_ind_unmatched = list(torch.arange(len(assign_result.gt_inds))[prev_out_ind_unmatched].numpy())

            # Create False Negative
            if self.random_fn_rate:
                random_subset_mask = torch.empty(len(prev_target_ind)).uniform_()
                random_subset_mask = random_subset_mask.ge(self.random_fn_rate)

                prev_out_ind_fn = prev_out_ind[random_subset_mask]

                prev_target_ind_fn = prev_target_ind[random_subset_mask]

            # Create False Positive
            # if self.random_fp_rate:
            #     fp_idx = assign_result.iou_cost.min(-1)[0] > 0
            #     random_subset_mask = torch.empty(len(fp_idx)).uniform_()
            #     random_subset_mask = random_subset_mask.ge(1 - self.random_fp_rate)
            #     fp_idx = torch.arange(len(fp_idx))[fp_idx.detach().cpu() * random_subset_mask]
            #     prev_out_ind_fn_fp = torch.cat((prev_out_ind_fn, fp_idx))
            # else:
            # fp_num = self.max_track_query - len(prev_out_ind_fn)
            # _, fp_idx = torch.topk(assign_result.iou_cost.min(-1)[0], fp_num)
            # prev_out_ind_fn_fp = torch.cat((prev_out_ind_fn, fp_idx.cpu()))
            fp_indices = []
            prev_duplicates_indices = []
            prev_target_dup = []
            for k, prev_idx in enumerate(prev_out_ind_fn):

                if random.uniform(0, 1) < self.random_fp_rate:
                    prev_box_matched = prev_boxes[i][prev_idx]
                    prev_boxes_unmatched = prev_boxes[i][prev_out_ind_unmatched]

                    prev_box_ious = bbox_overlaps(
                        bbox_cxcywh_to_xyxy(prev_box_matched.unsqueeze(dim=0)),
                        bbox_cxcywh_to_xyxy(prev_boxes_unmatched))
                    box_weights = prev_box_ious[0]

                    if box_weights.gt(0.0).any():
                        fp_idx = prev_out_ind_unmatched.pop(
                            torch.multinomial(box_weights.cpu(), 1).item())
                        fp_indices.append(fp_idx)

                    # Adding true postive as false positive to kill
                    pass

                if random.uniform(0, 1) < self.random_dup_rate:
                    # Duplicate FP_idx
                    prev_duplicates_indices.append(prev_idx)
                    prev_target_dup.append(prev_target_ind_fn[k])

            fp_indices = torch.tensor(fp_indices, device=prev_out_ind_fn.device).long()
            prev_duplicates_indices = torch.tensor(prev_duplicates_indices, device=prev_out_ind_fn.device).long()
            prev_target_dup = torch.tensor(prev_target_dup, device=prev_target_ind_fn.device, dtype=prev_target_ind_fn.dtype)

            prev_out_ind_fn_dup = torch.cat((prev_out_ind_fn, prev_duplicates_indices))
            prev_out_ind_fn_dup_fp = torch.cat((prev_out_ind_fn_dup, fp_indices))
            prev_target_ind_fn_dup = torch.cat((prev_target_ind_fn, prev_target_dup))
            # prev_track_id = ref_gt_instance_ids[i][prev_target_ind_fn]
            prev_track_match_id = ref_gt_match_indices[i][prev_target_ind_fn_dup]

            track_query_hs_embed = prev_hs[i][prev_out_ind_fn_dup_fp]
            track_query_box = prev_boxes[i][prev_out_ind_fn_dup_fp]

            # Create Matching
            track_query_mask = torch.ones(len(prev_track_match_id))
            track_query_mask[prev_track_match_id == -1] = -1
            track_query_mask = torch.cat((track_query_mask, -torch.ones(len(fp_indices)), torch.zeros(self.detector.bbox_head.num_query)))

            # Collect object information
            track_query_matched_ids.append(prev_track_match_id)
            track_query_hs_embeds.append(track_query_hs_embed)
            track_query_boxes.append(track_query_box)
            track_query_masks.append(track_query_mask)

        # Only forward the first one, use the second one to get FP.
        if bs > 1:
            fpdb_num = (track_query_masks[-1] == 1).sum()
            fpdb_valid = torch.rand(fpdb_num) < self.fpdb_rate * len(track_query_hs_embeds[0]) / (len(track_query_hs_embeds[1]) + 1)
            fpdb_emb = track_query_hs_embeds[-1][:fpdb_num][fpdb_valid]
            fpdb_box = track_query_boxes[-1][:fpdb_num][fpdb_valid]
            fpdb_mask = -track_query_masks[-1][:fpdb_num][fpdb_valid]
            track_query_masks[0] = torch.cat([track_query_masks[0][:len(track_query_hs_embeds[0])], fpdb_mask, track_query_masks[0][len(track_query_hs_embeds[0]):]])
            track_query_hs_embeds[0] = torch.cat([track_query_hs_embeds[0], fpdb_emb])
            track_query_boxes[0] = torch.cat([track_query_boxes[0], fpdb_box])

        track_prev = {"track_query_matched_ids": track_query_matched_ids[:1],
                      "track_query_hs_embeds": torch.stack(track_query_hs_embeds[:1]),
                      "track_query_boxes": torch.stack(track_query_boxes[:1]),
                      "track_query_mask": track_query_masks[:1]}

        track_prev['track_query_hs_embeds'] += self.noise * torch.randn(track_prev['track_query_hs_embeds'].shape, device=track_prev['track_query_hs_embeds'].device)
        track_prev['track_query_boxes'] += self.noise * torch.randn(track_prev['track_query_boxes'].shape, device=track_prev['track_query_boxes'].device)
        return track_prev, loss_det

    def forward_current(self,
                        img,
                        img_metas,
                        # gt_bboxes,
                        # gt_labels,
                        # gt_instance_ids,
                        # gt_match_indices,
                        track_prev):

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        # if self._input_size != self._default_input_size:
        #     img, gt_bboxes = self.detector._preprocess(img, gt_bboxes)
        x = self.detector.extract_feat(img)
        outs = self.detector.bbox_head.forward_current(x, img_metas, track_prev)
        return outs

    # def simple_test(self, img, img_metas, rescale=False, **kwargs):
    #     """Test without augmentations.
    #
    #     Args:
    #         img (Tensor): of shape (N, C, H, W) encoding input images.
    #             Typically these should be mean centered and std scaled.
    #         img_metas (list[dict]): list of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #         rescale (bool, optional): If False, then returned bboxes and masks
    #             will fit the scale of img, otherwise, returned bboxes and masks
    #             will fit the scale of original image shape. Defaults to False.
    #
    #     Returns:
    #         dict[str : list(ndarray)]: The tracking results.
    #     """
    #     frame_id = img_metas[0].get('frame_id', -1)
    #     if frame_id == 0:
    #         self.tracker.reset()
    #
    #     det_results = self.detector.simple_test(
    #         img, img_metas, rescale=rescale)
    #     assert len(det_results) == 1, 'Batch inference is not supported.'
    #     bbox_results = det_results[0]
    #     num_classes = len(bbox_results)
    #
    #     outs_det = results2outs(bbox_results=bbox_results)
    #     det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
    #     det_labels = torch.from_numpy(outs_det['labels']).to(img).long()
    #
    #     track_bboxes, track_labels, track_ids = self.tracker.track_byte(
    #         img=img,
    #         img_metas=img_metas,
    #         model=self,
    #         bboxes=det_bboxes,
    #         labels=det_labels,
    #         frame_id=frame_id,
    #         rescale=rescale,
    #         **kwargs)
    #
    #     track_results = outs2results(
    #         bboxes=track_bboxes,
    #         labels=track_labels,
    #         ids=track_ids,
    #         num_classes=num_classes)
    #     det_results = outs2results(
    #         bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)
    #
    #     return dict(
    #         det_bboxes=det_results['bbox_results'],
    #         track_bboxes=track_results['bbox_results'])

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        # det_results = self.detector.simple_test(
        #     img, img_metas, rescale=rescale)
        # assert len(det_results) == 1, 'Batch inference is not supported.'
        # bbox_results = det_results[0]
        # num_classes = len(bbox_results)
        #
        # outs_det = results2outs(bbox_results=bbox_results)
        # det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        # det_labels = torch.from_numpy(outs_det['labels']).to(img).long()
        with torch.no_grad():
            track_bboxes, track_labels, track_ids, det_bboxes, det_labels = self.tracker.track(
                img=img,
                img_metas=img_metas,
                model=self,
                frame_id=frame_id,
                rescale=rescale,
                **kwargs)

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=1)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=1)

        return dict(det_bboxes=det_results['bbox_results'], track_bboxes=track_results['bbox_results'])


@MODELS.register_module()
class MO3TRnF(MO3TR):
    def __init__(self, **kwargs):
        super(MO3TRnF, self).__init__(**kwargs)

        self.temporal_model = MO3TRTemp()

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(kwargs['img'], kwargs['img_metas'], rescale=kwargs['rescale'])

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=1)

        return outputs

    def forward_train(self,
                      all_frames,
                      **kwargs):
        last_frame = all_frames.pop()
        with torch.no_grad():
            track_prev = self.forward_prev_nf(all_frames, last_frame)

        img, img_metas, gt_bboxes, gt_labels, gt_instance_ids, gt_match_indices = last_frame["img"][:1], last_frame["img_metas"][:1], last_frame["gt_bboxes"][:1], last_frame["gt_labels"][:1], last_frame["gt_instance_ids"][:1], \
                                                                                  last_frame["gt_match_indices"][:1]
        outs = self.forward_current(img, img_metas, track_prev)

        loss_inputs = outs[1:] + (gt_bboxes, gt_labels, gt_instance_ids, track_prev, img_metas)

        losses = self.detector.bbox_head.loss_mo3tr(*loss_inputs)
        # if self.grad == "separate":
        #     losses = {**losses, **loss_det}
        return losses

    def forward_prev_nf(self, all_frames, last_frame):
        self.tracker.reset()
        inference_outs = collections.defaultdict(list)
        metric = "track"
        # gt = DataFrame("FrameId", "Id", "X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility")
        gt_frame = []
        dt_frame = []
        indexes = []
        indexes_dt = []
        for i, frame in enumerate(all_frames):
            track_bboxes, track_labels, track_ids, det_bboxes, det_labels = self.tracker.track(
                img=frame["img"],
                img_metas=frame["img_metas"],
                model=self,
                frame_id=i,
                rescale=False,
                remove_dup=False,
                temp=True)

            track_bboxes[:, 2] = track_bboxes[:, 2] - track_bboxes[:, 0]
            track_bboxes[:, 3] = track_bboxes[:, 3] - track_bboxes[:, 1]
            track_bboxes = list(torch.cat([track_bboxes.cpu(), -torch.ones((len(track_bboxes), 2))], dim=-1).numpy())
            dt_frame += track_bboxes
            indexes_dt += [(i, int(oid)) for oid in track_ids]

            for j in range(len(frame["gt_bboxes"][0])):
                x, y, w, h = frame["gt_bboxes"][0][j][0], frame["gt_bboxes"][0][j][1], frame["gt_bboxes"][0][j][2] - frame["gt_bboxes"][0][j][0], frame["gt_bboxes"][0][j][3] - \
                             frame["gt_bboxes"][0][j][1]
                gt_j = [round(float(x)), round(float(y)), round(float(w)), round(float(h)), 1, 1, 1.0]
                indexes.append((i, int(frame["gt_instance_ids"][0][j])))
                gt_frame.append(gt_j)

        # Compute gt and dt
        multi_idx = pd.MultiIndex.from_tuples(indexes, names=["FrameId", "Id"])
        gt = pd.DataFrame(gt_frame, columns=["X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility"], index=multi_idx)

        multi_idx_dt = pd.MultiIndex.from_tuples(indexes_dt, names=["FrameId", "Id"])
        dt = pd.DataFrame(dt_frame, columns=["X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility"], index=multi_idx_dt)
        ini_file = frame['img_metas'][0]['filename'][:-15] + "seqinfo.ini"

        acc, ana = mm.utils.CLEAR_MOT_M(gt, dt, ini_file, distth=0.5)
        # mh = mm.metrics.create()
        # summary = mh.compute_many(
        #     [acc],
        #     names=["test"],
        #     metrics=mm.metrics.motchallenge_metrics,
        #     generate_overall=True)
        # str_summary = mm.io.render_summary(
        #     summary,
        #     formatters=mh.formatters,
        #     namemap=mm.io.motchallenge_metric_names)
        # print(str_summary)

        prev_ids, prev_hs, prev_bboxes, track_prev, track_ids = [], [], [], {}, []
        for track_id, track in self.tracker.tracks.items():
            prev_ids.append(track['ids'][-1])
            hs_temp = torch.stack(track['hs'])
            loc_temp = torch.stack(track['bboxes'])
            prev_hs.append(hs_temp)
            prev_bboxes.append(loc_temp)
        prev_hs, prev_bboxes = self.temporal_model(prev_hs, prev_bboxes)

        track_prev["track_query_hs_embeds"] = prev_hs
        track_prev["track_query_boxes"] = prev_bboxes[..., :4]

        oid2gid = acc.res_m
        prev_ids = torch.LongTensor([oid2gid[int(oid)] if int(oid) in oid2gid else -1 for oid in prev_ids]).cuda()
        track_query_matched_ids, matches_b = match_gts(prev_ids, last_frame["gt_instance_ids"][0])
        track_query_mask = torch.ones((len(track_query_matched_ids)))
        track_query_mask[track_query_matched_ids < 0] = -1
        track_query_mask = torch.cat([track_query_mask, torch.zeros(self.detector.bbox_head.num_query)]).cuda()
        track_prev["track_query_matched_ids"] = [track_query_matched_ids]
        track_prev["track_query_mask"] = [track_query_mask]

        return track_prev


class MO3TRTemp(nn.Module):
    def __init__(self, d_model=320, nhead=8, num_layers=3, seq_len=30, pred_len=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, 3)
        self.with_pe = PositionalEncoding1D(d_model, seq_len+pred_len)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.box_model = BoxTransformer()
        # self.box_model.load_state_dict(torch.load("/storage/alan/workspace/generic_storage/temporal_transformer/max_input_30_pred_len_1_layer_3_b128_v1/weights/160000.pth"))

    def forward(self, prev_hs, prev_loc):
        prev_hs = torch.cat(([torch.cat([torch.zeros((self.seq_len - len(hs), 1, hs.shape[-1]), device=hs.device), hs]) for hs in prev_hs]), dim=1)
        x = torch.cat([prev_hs, prev_hs[-1:].repeat(self.pred_len, 1, 1)])
        # x = x.transpose(0, 1)
        input_len, batch_size, _ = x.shape
        # Transformer
        tgt = self.with_pe(x)
        tgt = self.transformer(tgt)

        # prev_loc = [loc[:, 0, :4] for loc in prev_loc]
        # prev_loc = torch.cat([loc[-1:] for loc in prev_loc], dim=1)
        predict_loc = self.box_model([loc[:, 0, :4] for loc in prev_loc])
        predict_loc = predict_loc.transpose(0, 1)[-1:]
        return tgt[self.seq_len + 1:self.seq_len + 2], torch.cat([loc[-1:] for loc in prev_loc], dim=1)


class BoxTransformer(nn.Module):
    def __init__(self, d_in=4, d_model=256, nhead=8, max_input_len=30, pred_len=1, layer=3):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_in, d_model)
        self.linear2 = nn.Linear(d_model, d_in)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, layer)
        self.with_pe = PositionalEncoding1D(d_model, max_input_len + pred_len)
        self.max_input_len = max_input_len
        self.pred_len = pred_len

    def forward(self, x):
        x = [self.linear1(inverse_sigmoid(x_)) for x_ in x]
        x = [torch.cat((torch.zeros((self.max_input_len - x_.shape[0], x_.shape[-1]), device=x_.device), x_)) for x_ in x]
        x = torch.stack(x)
        curr_x = x[:, -1:].repeat(1, self.pred_len, 1)

        # x = torch.cat((x, torch.zeros((len(x), self.pred_len, self.d_model), device=x.device)), axis=1)
        x = torch.cat((x, curr_x), axis=1)
        x = x.transpose(0, 1)

        # Transformer
        x = self.with_pe(x)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.linear2(x)

        x = x.sigmoid()
        return x
