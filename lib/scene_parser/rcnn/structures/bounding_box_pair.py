# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from .bounding_box import BoxList
# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxPairList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox_pair, image_size, mode="xyxy"):
        device = bbox_pair.device if isinstance(bbox_pair, torch.Tensor) else torch.device("cpu")
        bbox_pair = torch.as_tensor(bbox_pair, dtype=torch.float32, device=device)
        if bbox_pair.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox_pair.ndimension())
            )
        if bbox_pair.size(-1) != 8:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 8, got {}".format(bbox_pair.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox_pair
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxPairList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxPairList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    # def convert_from_boxlist(self, boxes):
    #     # input:
    #     #    boxes: boxlist


    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxPairList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxPairList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxPairList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxPairList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxPairList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxPairList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxPairList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def copy_with_subject(self):
        bbox = BoxList(self.bbox[:, :4], self.size, self.mode)
        return bbox

    def copy_with_object(self):
        bbox = BoxList(self.bbox[:, 4:], self.size, self.mode)
        return bbox

    def copy_with_union(self):
        x1 = self.bbox[:, 0::4].min(dim=1).values.view(-1, 1) # x1
        y1 = self.bbox[:, 1::4].min(dim=1).values.view(-1, 1) # y1
        x2 = self.bbox[:, 2::4].max(dim=1).values.view(-1, 1) # x2
        y2 = self.bbox[:, 3::4].max(dim=1).values.view(-1, 1) # y2
        bbox = BoxList(torch.cat((x1, y1, x2, y2), 1), self.size, self.mode)
        return bbox

    '''separate bounding boxes position'''
    def copy_with_separate(self):
        assert self.bbox.size(-1)==8, \
            f'bounding box pair size should be 8, got {self.bbox.size(-1)}'
        box1 = self.bbox[:, :4] # Nx8 -> Nx4 (obj1)
        box2 = self.bbox[:, 4:] # Nx8 -> Nx4 (obj2)
        boxlist1 = BoxList(box1, self.size, self.mode) # BoxList(Nx4, WxH, 'xyxy')
        boxlist2 = BoxList(box2, self.size, self.mode) # BoxList(Nx4, WxH, 'xyxy')
        return boxlist1, boxlist2

    # def relativity_embedding(self):
    #     #assert type(self.size[0])==float and type(self.size[1])==float, 'type of size should be float'
    #     box1_xyxy = self.bbox[:, :4] # xmin_1, ymin_1, xmax_1, ymax_1
    #     box2_xyxy = self.bbox[:, 4:] # xmin_2, ymin_2, xmax_2, ymax_2
        
    #     # Convert (x_min, y_min, x_max, y_max) to (x_center, y_center, width, height)
    #     box1_w, box1_h = \
    #         box1_xyxy[:, 2::4] - box1_xyxy[:, 0::4], box1_xyxy[:, 3::4] - box1_xyxy[:, 1::4] # Nx1, Nx1
    #     box2_w, box2_h = \
    #         box2_xyxy[:, 2::4] - box2_xyxy[:, 0::4], box2_xyxy[:, 3::4] - box2_xyxy[:, 1::4] # Nx1, Nx1
    #     box1_xc, box1_yc = box1_xyxy[:, 0::4] + box1_w / 2, box1_xyxy[:, 1::4] + box1_h / 2 # Nx1, Nx1
    #     box2_xc, box2_yc = box2_xyxy[:, 0::4] + box2_w / 2, box2_xyxy[:, 1::4] + box2_h / 2 # Nx1, Nx1
    #     box1_xywh = torch.cat((box1_xc, box1_yc, box1_w, box1_h), dim=1) # xc_1, yc_1, w1, h1
    #     box2_xywh = torch.cat((box2_xc, box2_yc, box2_w, box2_h), dim=1) # xc_2, yc_2, w2, h2

    #     '''
    #     bounding box relative position
    #         left top: (xmin_2-xmin_1, ymin_2-ymin_1)
    #         right top: (xmax_2-xmax_1, ymin_2-ymin_1)
    #         left bottom: (xmin_2-xmin_1, ymax_2-y_max_1)
    #         right bottom: (xmax_2-xmax_1, ymax_2-ymax_1)
    #         center: (xc_2-xc_1, yc_2-yc_1)
    #     '''
    #     w, h = float(self.size[0]), float(self.size[1])
    #     xyxy_diff = box2_xyxy - box1_xyxy # Nx4
    #     xmin_diff, ymin_diff, xmax_diff, ymax_diff = \
    #         xyxy_diff[:,0].view(-1, 1), xyxy_diff[:,1].view(-1, 1), \
    #         xyxy_diff[:,2].view(-1, 1), xyxy_diff[:,3].view(-1, 1) # Nx1, Nx1, Nx1, Nx1
     
    #     xywh_diff = box2_xywh - box1_xywh # Nx4
    #     xc_diff, yc_diff = \
    #         xywh_diff[:,0].view(-1, 1), xywh_diff[:,1].view(-1, 1) # Nx1, Nx1
        
    #     lt = torch.cat((xmin_diff/w, ymin_diff/h), dim=1) # Nx2
    #     rt = torch.cat((xmax_diff/w, ymin_diff/h), dim=1) # Nx2
    #     lb = torch.cat((xmin_diff/w, ymax_diff/h), dim=1) # Nx2
    #     rb = torch.cat((xmax_diff/w, ymax_diff/h), dim=1) # Nx2
    #     ctr = torch.cat((xc_diff/w, yc_diff/h), dim=1) # Nx2

    #     '''
    #     bounding box relative area
    #         portion: (box1_area/total_area, box2_area/total_area)
    #     '''
    #     total_area = w * h
    #     # box1_w, box1_h = box1_xywh[:, 2::4], box1_xywh[:, 3::4] # Nx1, Nx1
    #     # box2_w, box2_h = box2_xywh[:, 2::4], box2_xywh[:, 3::4] # Nx1, Nx1
    #     box1_area = box1_w * box1_h # Nx1
    #     box2_area = box2_w * box2_h # Nx1
    #     portion1 = box1_area / total_area # Nx1
    #     portion2 = box2_area / total_area # Nx1
    #     portion = torch.cat((portion1, portion2), dim=1) # Nx2

    #     rel_features = torch.stack((lt, rt, lb, rb, ctr, portion), dim=2) # Nx2x6
    #     #print(rel_features)

    #     # relatvity Embedding
    #     rel_embedding = nn.Sequential(
    #                         nn.Linear(6, 64),
    #                         nn.ReLU(inplace=True),
    #                         nn.Linear(64, 64),
    #                     ).to(rel_features.device) # Nx2x64

    #     out = rel_embedding(rel_features) # Nx2x64
    #     return out # Nx2x64

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    #bbox = BoxPairList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    bbox = BoxPairList([[0, 0, 10, 10, 0, 0, 5, 5], [1, 2, 8, 5, 2, 4, 9, 7]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)


    '''
    t_bbox = bbox.transpose(1)
    print(t_bbox)
    print(t_bbox.bbox)
    '''