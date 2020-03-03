import torch
import torch.nn as nn

class SpatialRelEmbedding(nn.Module):
    '''
    Spatial relation embedding

        Arguments
            box_pair_list(BoxPairList)
    '''
    def __init__(self, xy_hidden=64, spatial_hidden=256, num_dots=8, reduction_ratio=4):
        super(SpatialRelEmbedding, self).__init__()

        self.spatial_hidden = spatial_hidden
        self.num_dots = num_dots

        # embed (x, y)
        self.xy_embedding = nn.Sequential(
                                nn.Linear(2, xy_hidden),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.1),
                                nn.Linear(xy_hidden, 1)
                            )

        # spatial relatvity Embedding
        self.spatial_embedding = nn.Sequential(
                                    nn.Linear(num_dots*num_dots, spatial_hidden//reduction_ratio),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(spatial_hidden//reduction_ratio, spatial_hidden)
                                ) # e.g., Nx256 -> Nxspatial_hidden
        
    def _convert(self, bbox_xyxy):
        '''
        Convert (x_min, y_min, x_max, y_max) to (x_center, y_center, width, height)
        '''
        bbox_w, bbox_h = \
            bbox_xyxy[:, 2::4] - bbox_xyxy[:, 0::4], bbox_xyxy[:, 3::4] - bbox_xyxy[:, 1::4] # Nx1, Nx1
        bbox_xc, bbox_yc = bbox_xyxy[:, 0::4] + bbox_w / 2, bbox_xyxy[:, 1::4] + bbox_h / 2 # Nx1, Nx1
        bbox_xywh = torch.cat((bbox_xc, bbox_yc, bbox_w, bbox_h), dim=1) # Nx4 (xc, yc, w, h)
        return bbox_xywh
        
    def forward(self, proposal_pairs):

        spatial_embeds = []

        for box_pair_list in proposal_pairs:
            '''
            box_pair_list (Nx8): bounding box pair list
            '''
            box1_xyxy = box_pair_list.bbox[:, :4] # xmin_1, ymin_1, xmax_1, ymax_1
            box2_xyxy = box_pair_list.bbox[:, 4:] # xmin_2, ymin_2, xmax_2, ymax_2
            # box1_xywh = self._convert(box1_xyxy) # xc_1, yc_1, w1, h1
            # box2_xywh = self._convert(box2_xyxy) # xc_2, yc_2, w2, h2
            
            # w, h = float(box_pair_list.size[0]), float(box_pair_list.size[1])
            # xyxy_diff = box2_xyxy - box1_xyxy # Nx4
            # xmin_diff, ymin_diff, xmax_diff, ymax_diff = \
            #     xyxy_diff[:,0].view(-1, 1), xyxy_diff[:,1].view(-1, 1), \
            #     xyxy_diff[:,2].view(-1, 1), xyxy_diff[:,3].view(-1, 1) # Nx1, Nx1, Nx1, Nx1
            
            # xywh_diff = box2_xywh - box1_xywh # Nx4
            # xc_diff, yc_diff = \
            #     xywh_diff[:,0].view(-1, 1), xywh_diff[:,1].view(-1, 1) # Nx1, Nx1
            
            '''
            bounding box relative position
                left top: (xmin_2-xmin_1, ymin_2-ymin_1)
                right top: (xmax_2-xmax_1, ymin_2-ymin_1)
                left bottom: (xmin_2-xmin_1, ymax_2-y_max_1)
                right bottom: (xmax_2-xmax_1, ymax_2-ymax_1)
                center: (xc_2-xc_1, yc_2-yc_1)
            '''
            # lt = torch.cat((xmin_diff/w, ymin_diff/h), dim=1) # Nx2
            # rt = torch.cat((xmax_diff/w, ymin_diff/h), dim=1) # Nx2
            # lb = torch.cat((xmin_diff/w, ymax_diff/h), dim=1) # Nx2
            # rb = torch.cat((xmax_diff/w, ymax_diff/h), dim=1) # Nx2
            # ctr = torch.cat((xc_diff/w, yc_diff/h), dim=1) # Nx2

            def box_to_dots(box_xyxy, num_dots):
                box_x = torch.linspace(box_xyxy[0], box_xyxy[2], num_dots)
                box_y = torch.linspace(box_xyxy[1], box_xyxy[3], num_dots)
                box = torch.tensor([[i,j] for i in box_x for j in box_y])
                return box
            
            box1_dots = []
            box2_dots = []

            for i in range(box1_xyxy.size(0)): # N
                box1_dots_i = box_to_dots(box1_xyxy[i], self.num_dots)
                box2_dots_i = box_to_dots(box2_xyxy[i], self.num_dots)
                box1_dots.append(box1_dots_i)
                box2_dots.append(box2_dots_i)

            box1_dots = torch.stack(box1_dots, dim=0) # N x D^2 x 2
            box2_dots = torch.stack(box2_dots, dim=0) # N x D^2 x 2
            box1_to_box2 = box2_dots - box1_dots # N x D^2 x 2

            self.xy_embedding = self.xy_embedding.to(box1_to_box2.device)
            self.spatial_embedding = self.spatial_embedding.to(box1_to_box2.device)

            box1_to_box2 = self.xy_embedding(box1_to_box2) # N x D^2 x 1
            box1_to_box2 = box1_to_box2.squeeze() # N x D^2
            spatial_embed = self.spatial_embedding(box1_to_box2) # N x K

            '''
            bounding box relative area
                fraction: (box1_area/total_area, box2_area/total_area)
            '''
            # total_area = w * h
            # box1_area = box1_xywh[:, 2] * box1_xywh[:, 3] # N
            # box2_area = box2_xywh[:, 2] * box2_xywh[:, 3] # N
            # fraction1 = box1_area.view(-1, 1) / total_area # Nx1
            # fraction2 = box2_area.view(-1, 1) / total_area # Nx1
            # fraction = torch.cat((fraction1, fraction2), dim=1) # Nx2

            spatial_embeds.append(spatial_embed)

        spatial_embeds = torch.cat(spatial_embeds, dim=0)
        return spatial_embeds

def make_spatial_relation_feature_extractor(xy_hidden, spatial_hidden, num_dots, reduction_ratio):
    spatial_embeds = SpatialRelEmbedding(xy_hidden, spatial_hidden, num_dots, reduction_ratio)
    return spatial_embeds