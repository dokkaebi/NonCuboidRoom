from math import ceil
import os
import numpy as np
import torch
import yaml
from easydict import EasyDict
import argparse
import cv2
import copy
from models.loss import _transpose_and_gather_feat
from models.reconstruction import (PreProcess, MergeNeighborWithSameParam,
                                   CommonRegion)
from models.utils import line_gaussian
from datasets.scannet import ScannetDataset

from datasets import NYU303, CustomDataset, Structured3D
from models import (ConvertLayout, Detector, DisplayLayout, display2Dseg, Loss,
                    MobileViTDetector, Reconstruction, _validate_colormap,
                    post_process)
from scipy.optimize import linear_sum_assignment


def match_by_Hungarian(gt, pred):
    n = len(gt)
    m = len(pred)
    gt = np.array(gt)
    pred = np.array(pred)
    valid = (gt.sum(0) > 0).sum()
    if m == 0:
        raise IOError
    else:
        gt = gt[:, np.newaxis, :, :]
        pred = pred[np.newaxis, :, :, :]
        cost = np.sum((gt+pred) == 2, axis=(2, 3))  # n*m
        row, col = linear_sum_assignment(-1 * cost)
        inter = cost[row, col].sum()
        PE = inter / valid
        return 1 - PE


def evaluate(gtseg, gtdepth, preseg, predepth, evaluate_2D=True, evaluate_3D=True):
    image_iou, image_pe, merror_edge, rmse, us_rmse = 0, 0, 0, 0, 0
    if evaluate_2D:
        # Parse GT polys
        gt_polys_masks = []
        h, w = gtseg.shape
        gt_polys_edges_mask = np.zeros((h, w))
        edge_thickness = 1
        gt_valid_seg = np.ones((h, w))
        labels = np.unique(gtseg)
        for i, label in enumerate(labels):
            gt_poly_mask = gtseg == label
            if label == -1:
                gt_valid_seg[gt_poly_mask] = 0  # zero pad region
            else:
                contours_, hierarchy = cv2.findContours(gt_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(gt_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                gt_polys_masks.append(gt_poly_mask.astype(np.int32))

        def sortPolyBySize(mask):
            return mask.sum()
        gt_polys_masks.sort(key=sortPolyBySize, reverse=True)

        # Parse predictions
        pred_polys_masks = []
        pred_polys_edges_mask = np.zeros((h, w))
        pre_invalid_seg = np.zeros((h, w))
        labels = np.unique(preseg)
        for i, label in enumerate(labels):
            pred_poly_mask = np.logical_and(preseg == label, gt_valid_seg == 1)
            if pred_poly_mask.sum() == 0:
                continue
            if label == -1:
                # zero pad and infinity region
                pre_invalid_seg[pred_poly_mask] = 1
            else:
                contours_, hierarchy = cv2.findContours(pred_poly_mask.astype(
                    np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE
                cv2.polylines(pred_polys_edges_mask, contours_, isClosed=True, color=[
                              1.], thickness=edge_thickness)
                pred_polys_masks.append(pred_poly_mask.astype(np.int32))
        if len(pred_polys_masks) == 0.:
            pred_polys_edges_mask[edge_thickness:-
                                  edge_thickness, edge_thickness:-edge_thickness] = 1
            pred_polys_edges_mask = 1 - pred_polys_edges_mask
            pred_poly_mask = np.ones((h, w))
            pred_polys_masks = [pred_poly_mask]

        pred_polys_masks_cand = copy.copy(pred_polys_masks)
        # Assign predictions to ground truth polygons
        ordered_preds = []
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            best_iou_score = 0.3
            best_pred_ind = None
            best_pred_poly_mask = None
            if len(pred_polys_masks_cand) == 0:
                break
            for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand):
                gt_pred_add = gt_poly_mask + pred_poly_mask
                inter = np.equal(gt_pred_add, 2.).sum()
                union = np.greater(gt_pred_add, 0.).sum()
                iou_score = inter / union

                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_pred_ind = pred_ind
                    best_pred_poly_mask = pred_poly_mask
            ordered_preds.append(best_pred_poly_mask)

            pred_polys_masks_cand = [pred_poly_mask for pred_ind, pred_poly_mask in enumerate(pred_polys_masks_cand)
                                     if pred_ind != best_pred_ind]
            if best_pred_poly_mask is None:
                continue

        ordered_preds += pred_polys_masks_cand
        class_num = max(len(ordered_preds), len(gt_polys_masks))
        colormap = _validate_colormap(None, class_num + 1)

        # Generate GT poly mask
        gt_layout_mask = np.zeros((h, w))
        gt_layout_mask_colored = np.zeros((h, w, 3))
        for gt_ind, gt_poly_mask in enumerate(gt_polys_masks):
            gt_layout_mask = np.maximum(
                gt_layout_mask, gt_poly_mask * (gt_ind + 1))
            gt_layout_mask_colored += gt_poly_mask[:,
                                                   :, None] * colormap[gt_ind + 1]

        # Generate pred poly mask
        pred_layout_mask = np.zeros((h, w))
        pred_layout_mask_colored = np.zeros((h, w, 3))
        for pred_ind, pred_poly_mask in enumerate(ordered_preds):
            if pred_poly_mask is not None:
                pred_layout_mask = np.maximum(
                    pred_layout_mask, pred_poly_mask * (pred_ind + 1))
                pred_layout_mask_colored += pred_poly_mask[:,
                                                           :, None] * colormap[pred_ind + 1]

        # Calc IOU
        ious = []
        for layout_comp_ind in range(1, len(gt_polys_masks) + 1):
            inter = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                   np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fp = np.logical_and(np.not_equal(gt_layout_mask, layout_comp_ind),
                                np.equal(pred_layout_mask, layout_comp_ind)).sum()
            fn = np.logical_and(np.equal(gt_layout_mask, layout_comp_ind),
                                np.not_equal(pred_layout_mask, layout_comp_ind)).sum()
            union = inter + fp + fn
            iou = inter / union
            ious.append(iou)

        image_iou = sum(ious) / class_num

        # Calc PE
        image_pe = 1 - np.equal(gt_layout_mask[gt_valid_seg == 1],
                                pred_layout_mask[gt_valid_seg == 1]).sum() / (np.sum(gt_valid_seg == 1))
        # Calc PE by Hungarian
        image_pe_hung = match_by_Hungarian(gt_polys_masks, pred_polys_masks)
        # Calc edge error
        # ignore edges at image borders
        img_bound_mask = np.zeros_like(pred_polys_edges_mask)
        img_bound_mask[10:-10, 10:-10] = 1

        pred_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - pred_polys_edges_mask)).astype(np.uint8),
                                                cv2.DIST_L2, 3)
        gt_dist_trans = cv2.distanceTransform((img_bound_mask * (1 - gt_polys_edges_mask)).astype(np.uint8),
                                              cv2.DIST_L2, 3)

        chamfer_dist = pred_polys_edges_mask * gt_dist_trans + \
            gt_polys_edges_mask * pred_dist_trans
        merror_edge = 0.5 * np.sum(chamfer_dist) / np.sum(
            np.greater(img_bound_mask * (gt_polys_edges_mask), 0))

    # Evaluate in 3D
    if evaluate_3D:
        max_depth = 50
        gt_layout_depth_img_mask = np.greater(gtdepth, 0.)
        gt_layout_depth_img = 1. / gtdepth[gt_layout_depth_img_mask]
        gt_layout_depth_img = np.clip(gt_layout_depth_img, 0, max_depth)
        gt_layout_depth_med = np.median(gt_layout_depth_img)
        # max_depth = np.max(gt_layout_depth_img)
        # may be max_depth should be max depth of all scene
        predepth[predepth == 0] = 1 / max_depth
        pred_layout_depth_img = 1. / predepth[gt_layout_depth_img_mask]
        pred_layout_depth_img = np.clip(pred_layout_depth_img, 0, max_depth)
        pred_layout_depth_med = np.median(pred_layout_depth_img)

        # Calc MSE
        ms_error_image = (pred_layout_depth_img - gt_layout_depth_img) ** 2
        rmse = np.sqrt(np.sum(ms_error_image) /
                       np.sum(gt_layout_depth_img_mask))

        # Calc up to scale MSE
        if np.isnan(pred_layout_depth_med) or pred_layout_depth_med == 0:
            d_scale = 1.
        else:
            d_scale = gt_layout_depth_med / pred_layout_depth_med
        us_ms_error_image = (
            d_scale * pred_layout_depth_img - gt_layout_depth_img) ** 2
        us_rmse = np.sqrt(np.sum(us_ms_error_image) /
                          np.sum(gt_layout_depth_img_mask))

    return image_iou, image_pe, merror_edge, rmse, us_rmse, image_pe_hung


def bbox_ctr(pts):
    return [
        (np.min(pts[:, 0]) + np.max(pts[:, 0])) / 2,
        (np.min(pts[:, 1]) + np.max(pts[:, 1])) / 2
    ]

def find_inds(ups, downs, pwalls, pfloor, pceiling, size, downsample=4):
    if not len(ups):
        print('no ups')
        return torch.tensor(np.zeros((0,)), dtype=torch.int64)
    h, w = size
    ih, iw = h * downsample, w * downsample
    centers = []

    ups = np.array(ups).astype(np.int32)
    ups[:, 0] = np.clip(ups[:, 0], 0, h)
    ups[:, 1] = np.clip(ups[:, 1], 0, w)
    downs = np.array(downs).astype(np.int32)
    downs[:, 0] = np.clip(downs[:, 0], 0, h)
    downs[:, 1] = np.clip(downs[:, 1], 0, w)
    centers.append(bbox_ctr(ups) if len(pceiling) > 0 else [-1, -1])
    centers.append(bbox_ctr(downs) if len(pfloor) > 0 else [-1, -1])

    assert len(ups) == len(pwalls) + 1
    for i in range(len(ups)-1):
        u0 = ups[i]
        u1 = ups[i+1]
        d0 = downs[i]
        d1 = downs[i+1]
        if pwalls[i] is None:
            assert i > 0 and i < len(ups)-2
            continue
        pts = np.array([u0, d0, d1, u1])
        centers.append(bbox_ctr(pts))

    #centers = [
    #    c for c in centers if c[0] <= ih and c[1] <= iw
    #]
    ct = np.array(centers, dtype=np.float32) / downsample
    ct_int = torch.tensor(ct, dtype=torch.int64)
    ret = ct_int[:, 1] * w + ct_int[:, 0]
    if ret.max() >= w*h:
        print(centers)
        print(ct)
        print(ret)
    return ret

def targets_from_model_out(inputs, outputs, seg, ups, downs, dt_lines, inds, depth, device):

    upsample = 1
    downsample = 4
    h, w = seg.shape
    oh, ow = ceil(h / downsample), ceil(w / downsample)

    # teacher_ouputs contains these:
    # 'plane_params_pixelwise'  # NOT USED! (B, 4, w, h)
    # 'plane_params_instance': # (B, 4, w, h)

    # 'plane_center': plane_center,
    # 'plane_wh': plane_wh,
    # 'plane_offset': plane_xy,
    # 'line_region': line_region,
    # 'line_params': line_params,
    # 'feature': x


    # utils.py post_process()
    # takes raw model output, produces:
    # - dt_planes: (B, K1, 6) x1,y1,x2,y2,score,class
    # - dt_lines: (B, K2, 4) x,y,alpha,score
    # - plane_params_instance: renormalized and selected from plane centers
    # - plane_params_pixelwise: renormalized and selected from plane centers

    # reconstruction.PreProcess()
    # filters planes output from post_process to threshold values, splits
    # into walls / floor / ceiling
    #

    # line_hm
    # example endpoint: [308.44584340438365, 359.5, 308.4105742802468, 28.842726503181716]
    # example line: array([[77.11146085, 89.875     ],
    #                      [77.10264357,  7.21068163]])
    # line_hm = np.zeros((3, oh, ow), dtype=np.float32)
    # for line in endpoints:
    #     line = np.array(line) / self.config.downsample
    #     line = np.reshape(line, [2, 2])
    #     line_gaussian(line_hm, line, 2)
    #
    # reconstruction.py:390
    # example dtl=array([ -7.3940454, 392.09537  ,   9.913257 ], dtype=float32)
    #                        m            b           confidence
    #      dtl = dtls[i]  # x=my+b
    #      fake_line = np.array([[dtl[1], 0], [dtl[0]+dtl[1], 1]]) * upsample
    line_hm = np.zeros((3, oh, ow))
    # for (up, down) in list(zip(ups, downs))[1:-1]:
    #     line = np.array([up, down]).T / downsample
    #     line_gaussian(line_hm, line, 2)
    for dtl in dt_lines[dt_lines[:, -1] == 1]:
        line = np.array([[dtl[1], 0], [dtl[0]+dtl[1], oh]])# / downsample
        line_gaussian(line_hm, line, 2)

    reg_mask = np.zeros((20,), dtype=np.uint8)
    reg_mask[:len(inds)] = inds >= 0
    ind = np.zeros((20,), dtype=np.int64)
    ind[:len(inds)] = np.maximum(0, inds)
    ind = torch.tensor(ind, dtype=torch.int64).unsqueeze(0).to(device)

    params3d = _transpose_and_gather_feat(
        outputs['plane_params_instance'], ind).detach().cpu().numpy()[0]

    oseg = cv2.resize(seg, (ow, oh), interpolation=cv2.INTER_NEAREST)
    odepth = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_NEAREST)

    plane_params = np.zeros((4, oh, ow), dtype=np.float32)
    for i, param in enumerate(params3d):
        param = np.array(param)
        plane_params[:3, oseg == i] = param[:3, np.newaxis]  # normal
        plane_params[3, oseg == i] = param[3]  # offset

    plane_wh = _transpose_and_gather_feat(outputs['plane_wh'], ind)
    plane_offset = _transpose_and_gather_feat(outputs['plane_offset'], ind)


    result = {
        # values that are compared directly in Loss can just be copied
        'plane_hm': outputs['plane_center'],
        'line_hm': line_hm,
        'line_offset': outputs['line_params'][:, 0:1],
        'line_alpha': outputs['line_params'][:, 1:2],

        # int64 (max_objs,): center coord in (w,h) for plane i's 2d
        # bounding box, flattened to yw+x
        # Used in _transpose_and_gather_feat to select per-plane values
        # from tensors of size (b,w,h,*).
        # ? Generate by finding topk peaks of plane_center output?
        'ind': ind,

        # uint8 (max_objs,): 1 if this entry is a gt plane, 0 otherwise
        # used to handle the fixed-size nature of params3d vs. the
        # variable number of planes that might exist in a given image
        # --- 1s for length of params3d and then 0s
        'reg_mask': reg_mask,

        # oseg shape: (w, h) int
        # each plane is represented by an integer 0..count
        # areas not covered by any plane are -1
        # --- generate as in Dataset.dataload by iterating over planes and using cv2.fillPoly
        # ? expensive to generate for every example, possible to parallelize and/or run on gpu?
        'oseg': oseg,

        # oxy1map: (w, h, 3) float32 where each entry is the homogeneous
        # coordinate of that pixel in the original unscaled input image
        # --- generate as in Dataset.getitem, eg. structured3d.py:129
        # ? reuse across all examples?
        'oxy1map': inputs['oxy1map'],

        # inverse of camera intrinsics Kinv
        # --- fixed value from Dataset
        'intri_inv': inputs['intri_inv'],

        # odepth: (w, h) depth in meters
        # --- generate using Dataset.inverdepth
        'odepth': odepth,



        # how to generate these?
        # w, h here are size of image after downsampling by config.downsample

        # same name, but maybe not directly comparable
        # loss selects from plane_wh / plane_offset using batch['ind'] and reshapes
        'plane_wh': plane_wh,
        'plane_offset': plane_offset,

        # params3d is built from annotations in Dataset.dataload
        # each entry is 4d vector [*normal, offset/1000] representing one plane
        # only planes larger than 1000 Polygon area are kept
        # params3d has shape (max_objs, 4) - remaining entries are zeros
        # --- should correspond to plane_params_instance after postprocess
        # ? sensitive to ordering?
        'params3d': params3d,

        # plane_params shape: (4, w, h)
        # Uses generated segmentation boundaries oseg to look up pixels
        # corresponding to each plane index (indexes of params3d).
        # All pixels are just set equal to the corresponding params3d
        # entry.
        # Since these are all set equal for a given plane in GT data,
        # and since the per-pixel model predictions are never used in
        # testing or visual rendering, we'll use the teacher's instance-
        # level predictions as GT, similarly copied to each pixel for
        # the plane.
        # --- params3d replicated for each pixel in the plane
        'plane_params': plane_params,



        # plane_params_instance_loss uses:
        # - reg_mask: (B, 20) [1, 1, 1, 0, 0, 0, ..., 0]
        # - ind: (B, 20) [7158, 7078, 556, 0, 0, ..., 0]
        # - params3d: (B, 20, 4) normal and distance for each plane
        # 1) entries from model output are selected using `ind`
        # 2) selected outputs and targets are masked with reg_mask
        # 3) compared using smooth L1 loss
        # Can use this approach for student-teacher, but need to fully
        # postprocess teacher output to threshold predictions and
        # combine overlapping planes etc.
    }

    return result


def test(model, dataloader, device, cfg, criterion):
    model.eval()
    for iters, (inputs, fnames) in enumerate(dataloader):
        print(f'{iters}/{len(dataloader)}')
        # set device
        for key, value in inputs.items():
            inputs[key] = value.to(device)

        # forward
        with torch.no_grad():
            x = model(inputs['img'])

        # post process on output feature map size and extract plane and line detection results
        dt_planes, dt_lines, dt_params3d_instance, dt_params3d_pixelwise = post_process(x, Mnms=1)

        for i in range(1):
            # generate layout with a post-process according to detection results
            (_ups, _downs, _attribution, _params_layout), (ups, downs, attribution, params_layout), (pfloor, pceiling) = Reconstruction(
                dt_planes[i],
                dt_params3d_instance[i],
                dt_lines[i],
                K=inputs['intri'][i].cpu().numpy(),
                size=ScannetDataset.image_size,
                threshold=(0.3, 0.3, 0.3, 0.3),
                downsample=cfg.Dataset.Scannet.downsample,
                cat='opt',
            )

            inds = find_inds(ups, downs, params_layout, pfloor, pceiling, x['plane_params_instance'].shape[-2:])

            # convert opt results to segmentation and depth map and evaluate results
            seg, depth, img, polys = ConvertLayout(
                inputs['img'][i], ups, downs, attribution,
                K=inputs['intri'][i].cpu().numpy(), pwalls=params_layout,
                pfloor=pfloor, pceiling=pceiling,
                ixy1map=inputs['ixy1map'][i].cpu().numpy(),
                valid=inputs['iseg'][i].cpu().numpy(),
                oxy1map=inputs['oxy1map'][i].cpu().numpy(), pixelwise=None)

            targets = targets_from_model_out(inputs, x, seg, ups, downs, dt_lines[0], inds, depth, device)
            for k, v in targets.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v).to(device)
                if v.shape[0] != 1:
                    # insert batch dimension
                    v = v.unsqueeze(0)
                targets[k] = v

            # Save the targets
            fname = fnames[i].replace('.jpg', '.pt')
            print(fname)
            torch.save(targets, fname)

            # l2, s2 = criterion(x, **inputs)

            # res = evaluate(inputs['iseg'][i].cpu().numpy(),
            #                inputs['idepth'][i].cpu().numpy(), seg, depth)


def parse(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='data/scannet/images')
    parser.add_argument('--pretrained', type=str, default=None, required=True, help='the pretrained model')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with open('cfg.yaml', 'r') as f:
        config = yaml.safe_load(f)
        cfg = EasyDict(config)
    args = parse()
    cfg.update(vars(args))

    dataset = ScannetDataset(cfg.Dataset.Scannet, phase='eval_teacher', files=cfg.img_dir)
    # dataset = Structured3D(cfg.Dataset.Structured3D, 'test')
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=cfg.num_workers, shuffle=False)

    # create network
    model = Detector()
    criterion = Loss(cfg.Weights)

    state_dict = torch.load(cfg.pretrained,
                            map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # set data parallel
    # if cfg.num_gpus > 1 and torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test(model, dataloader, device, cfg, criterion)
