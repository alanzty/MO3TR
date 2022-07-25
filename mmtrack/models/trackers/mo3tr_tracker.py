import lap
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from torchvision.ops import nms


@TRACKERS.register_module()
class Mo3trTracker(BaseTracker):
    def __init__(self,
                 obj_score_thrs=dict(high=0.6, low=0.1),
                 init_track_thr=0.7,
                 prop_thr=0.9,
                 weight_iou_with_det_scores=True,
                 match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
                 num_tentatives=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thrs = obj_score_thrs
        self.init_track_thr = init_track_thr
        self.prop_thr = prop_thr
        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thrs = match_iou_thrs

        self.num_tentatives = num_tentatives

    @force_fp32(apply_to=('img', 'bboxes'))
    def track(self,
              img,
              img_metas,
              model,
              frame_id,
              rescale=True,
              remove_dup=True,
              temp=True,
              **kwargs):
        if not self.tracks:
            # Initialization
            det_hs, det_cls, det_bboxes = model.forward_det(img, img_metas)
            det_hs = det_hs[0]
            det_bboxes = det_bboxes[0]
            det_cls = det_cls[0]

            # Select with threshold
            det_scores = det_cls.sigmoid()[:, 0]
            valid_det_idx = det_scores > self.init_track_thr
            det_labels = torch.zeros_like(det_scores, dtype=torch.int32)

            # Generate Valid Dets
            valid_det_bboxes = det_bboxes[valid_det_idx]
            ids = torch.arange(len(valid_det_bboxes), device=valid_det_bboxes.device)
            valid_det_labels = det_labels[valid_det_idx]
            valid_det_hs = det_hs[valid_det_idx]
            valid_det_scores = det_scores[valid_det_idx]
            valid_det_cls = det_cls[valid_det_idx]
            valid_det_bboxes = torch.cat([valid_det_bboxes, valid_det_scores.unsqueeze(-1)], dim=-1)
            self.max_track_id = len(valid_det_bboxes) - 1
            self.update(ids=ids, bboxes=valid_det_bboxes, labels=valid_det_labels, frame_ids=frame_id, hs=valid_det_hs)

            return self.to_output(valid_det_bboxes[:, :4], valid_det_cls, ids, det_bboxes, det_cls, model, img_metas[0], rescale=rescale)

        else:
            # Tracking
            prev_ids, prev_hs, prev_bboxes, track_prev, track_ids = [], [], [], {}, []
            if not temp:
                for track_id, track in self.tracks.items():
                    prev_ids.append(track['ids'][-1])
                    prev_hs.append(track['hs'][-1])
                    prev_bboxes.append(track['bboxes'][-1])
                prev_hs = torch.cat(prev_hs).unsqueeze(0)
                prev_bboxes = torch.cat(prev_bboxes).unsqueeze(0)
            else:
                for track_id, track in self.tracks.items():
                    prev_ids.append(track['ids'][-1])
                    hs_temp = torch.stack(track['hs'][-30:])
                    loc_temp = torch.stack(track['bboxes'][-30:])
                    prev_hs.append(hs_temp)
                    prev_bboxes.append(loc_temp)
                prev_hs, prev_bboxes = model.temporal_model(prev_hs, prev_bboxes)

            prev_ids = torch.cat(prev_ids)
            track_prev["track_query_hs_embeds"] = prev_hs
            track_prev["track_query_boxes"] = prev_bboxes[..., :4]
            hs, cls, bboxes, _, _ = model.forward_current(img, img_metas, track_prev)
            num_prev = len(prev_ids)

            track_cls = cls[-1, 0, :num_prev]
            track_bboxes = bboxes[-1, 0, :num_prev]
            track_hs = hs[-1, 0, :num_prev]
            det_cls = cls[-1, 0, num_prev:]
            det_bboxes = bboxes[-1, 0, num_prev:]
            det_hs = hs[-1, 0, num_prev:]

            # Compute Valid Track and Valid Det
            # Select with threshold
            track_scores = track_cls.sigmoid()[:, 0]
            valid_track_idx = track_scores > self.prop_thr
            track_labels = torch.zeros_like(track_scores, dtype=torch.int32)

            # Generate Valid tracks
            valid_track_bboxes = track_bboxes[valid_track_idx]
            valid_track_labels = track_labels[valid_track_idx]
            valid_track_hs = track_hs[valid_track_idx]
            valid_track_scores = track_scores[valid_track_idx]
            valid_track_cls = track_cls[valid_track_idx]
            valid_track_bboxes = torch.cat([valid_track_bboxes, valid_track_scores.unsqueeze(-1)], dim=-1)
            valid_track_ids = prev_ids[valid_track_idx]


            det_scores = det_cls.sigmoid()[:, 0]
            valid_det_idx = det_scores > self.init_track_thr
            det_labels = torch.zeros_like(det_scores, dtype=torch.int32)

            valid_det_bboxes = det_bboxes[valid_det_idx]
            valid_det_labels = det_labels[valid_det_idx]
            valid_det_hs = det_hs[valid_det_idx]
            valid_det_scores = det_scores[valid_det_idx]
            valid_det_cls = det_cls[valid_det_idx]
            valid_det_bboxes = torch.cat([valid_det_bboxes, valid_det_scores.unsqueeze(-1)], dim=-1)
            valid_det_ids = torch.arange(self.max_track_id + 1, self.max_track_id + 1 + len(valid_det_scores), device=valid_det_scores.device)

            # Combine track and init
            valid_ids = torch.cat((valid_track_ids, valid_det_ids))
            valid_boxes = torch.cat((valid_track_bboxes, valid_det_bboxes))
            valid_hs = torch.cat((valid_track_hs, valid_det_hs))
            valid_labels = torch.cat((valid_track_labels, valid_det_labels))
            valid_cls = torch.cat((valid_track_cls, valid_det_cls))
            try:
                self.max_track_id = max(self.max_track_id, max(valid_ids))
            except:
                self.max_track_id = 1

            if remove_dup:
                post_nms_idx = nms(bbox_cxcywh_to_xyxy(valid_boxes[..., :4]), valid_boxes[..., -1], 0.95)
                valid_ids = valid_ids[post_nms_idx]
                valid_boxes = valid_boxes[post_nms_idx]
                valid_labels = valid_labels[post_nms_idx]
                valid_hs = valid_hs[post_nms_idx]
                valid_cls = valid_cls[post_nms_idx]

            self.update(ids=valid_ids, bboxes=valid_boxes, labels=valid_labels, frame_ids=frame_id, hs=valid_hs)

            return self.to_output(valid_boxes[:, :4], valid_cls, valid_ids, bboxes[-1, 0], cls[-1, 0], model, img_metas[0], rescale=rescale)

    def to_output(self, track_bboxes, track_cls, ids, det_bboxes, det_cls, model, img_meta, rescale=True):
        det_bboxes, det_labels = model.detector.bbox_head._get_bboxes_single(det_cls, det_bboxes, img_meta["img_shape"], img_meta["scale_factor"], rescale=rescale)
        track_bboxes, track_labels = model.detector.bbox_head._get_bboxes_single_track(track_cls, track_bboxes, img_meta["img_shape"], img_meta["scale_factor"], rescale=rescale)
        return track_bboxes, track_labels, ids, det_bboxes, det_labels

