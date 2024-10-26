import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import cv2
import os
from SAM.modeling.mask_decoder import MaskDecoder
from SAM.modeling.prompt_encoder import PromptEncoder
from SAM.modeling.transformer import TwoWayTransformer
from SAM.modeling.common import LayerNorm2d, DownAndUp,LowLevelFeatureExtractor,LowLevelFeatureExtractorMixed
from SAM.modeling.image_encoder import ImageEncoderViT
from SAM.modeling.small_encoder import TinyViT
from functools import partial
from transforms import ResizeLongestSide
from typing import Any, Dict, List, Tuple


def PointGenerator(mask, visual=False):

    # 打印 mask 的类型和形状，确保其正确
    # print(
    #     f"PointGenerator - 类型: {type(mask)}, 形状: {mask.shape}, 数据类型: {mask.dtype}"
    # )

    # 确保 mask 是 numpy 数组
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask 不是一个 NumPy 数组")

    # 确保 mask 是 uint8 类型，并且打印转换前后的类型
    if mask.dtype != np.uint8:
        # print(f"转换前的数据类型: {mask.dtype}")
        mask = mask.astype(np.uint8)
        # print(f"转换后的数据类型: {mask.dtype}")

    # 打印以确保类型和形状正确
    # print(
    #     f"Before connectedComponentsWithStats - 类型: {type(mask)}, 形状: {mask.shape}, 数据类型: {mask.dtype}"
    # )

    # 使用 OpenCV 函数进行处理

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    point_coord = []
    point_class = []
    for i in range(1, num_labels):
        x, y = centroids[i]
        point_coord.append([x, y])
        point_class.append(1)
    if visual:
        #visualize_points(mask,point_coord,point_class)
        for point in point_coord:
            cv2.circle(mask, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
    box_coord = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        box_coord.append([x, y, x + w, y + h])
    mask_shape = mask.shape
    return point_coord, point_class, mask_shape, box_coord

import matplotlib.pyplot as plt
import numpy as np

def visualize_points(image, point_coord, point_class):
    """
    Visualize points on an image.
    
    Parameters:
    - image: A numpy array representing the image.
    - point_coord: A list of point coordinates.
    - point_class: A list of point classes.
    """
    # Convert image to numpy array if it's a torch tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    
    # Create a color map for different classes
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    plt.imshow(image)
    for (x, y), cls in zip(point_coord, point_class):
        plt.scatter(x, y, color=colors[cls % len(colors)], label=f'Class {cls}')
    
    plt.legend()
    plt.show()


import cv2
import torch
import numpy as np

import cv2
import torch
import numpy as np

def calculate_circularity(contours):
    circularity_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # If circularity is close to 1, it is more likely to be a circle
        if 0.7 < circularity < 1.3:
            circularity_count += 1
    return circularity_count

def otsu_threshold(image):
    """Compute Otsu threshold for a single grayscale image tensor."""
    hist = torch.histc(image, bins=256, min=0, max=1)
    hist = hist / hist.sum()  # Normalize histogram

    # Compute cumulative sums of histogram
    cumulative_sum = hist.cumsum(dim=0)
    cumulative_mean = (hist * torch.arange(256, device=image.device)).cumsum(dim=0)
    global_mean = cumulative_mean[-1]

    # Compute between-class variance
    between_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (cumulative_sum * (1 - cumulative_sum) + 1e-8)
    otsu_threshold = between_class_variance.argmax().item() / 255.0

    return otsu_threshold

def generate_otsu_mask(batch_image_tensor, size=(1024, 1024)):
    """
    Generates Otsu thresholded masks from a batch of tensor images.

    Parameters:
    - batch_image_tensor (torch.Tensor): Batch of input image tensors with shape (batch, 3, H, W).
    - size (tuple): Desired size for the output masks.

    Returns:
    - torch.Tensor: Batch of Otsu thresholded masks as tensors with shape (batch, 1, size[0], size[1]).
    """
    # Check if input is a tensor
    if not isinstance(batch_image_tensor, torch.Tensor):
        raise ValueError("Input image must be a torch.Tensor")

    # Check batch dimensions
    if len(batch_image_tensor.shape) != 4 or batch_image_tensor.shape[1] != 3:
        raise ValueError("Expected input tensor with shape (batch, 3, H, W)")

    batch_size, _, height, width = batch_image_tensor.shape
    resized_batch = F.interpolate(batch_image_tensor, size=size, mode='bilinear', align_corners=False)
    gray_batch = 0.2989 * resized_batch[:, 0, :, :] + 0.5870 * resized_batch[:, 1, :, :] + 0.1140 * resized_batch[:, 2, :, :]

    masks = torch.zeros((batch_size, 1, size[0], size[1]), device=batch_image_tensor.device)

    for i in range(batch_size):
        gray_image = gray_batch[i]
        threshold = otsu_threshold(gray_image)
        mask = (gray_image > threshold).float()
        masks[i, 0] = mask

    return masks




class SAMB(nn.Module):

    def __init__(
        self,
        data_path=None,
        img_size=1024,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    ):
        super(SAMB, self).__init__()
        encoder_embed_dim = 768
        encoder_depth = 12
        encoder_num_heads = 12
        encoder_global_attn_indexes = [2, 5, 8, 11]
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        ###################################
        self.image_encoder = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        ###################################
        self.path = data_path
        self.img_size = img_size
        self.pt = ResizeLongestSide(img_size)
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.mask_threshold = 0.0
        self.image_format = "RGB"
        


    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def forward(self, x, mask=None, img_id=None, mode="point"):
        b = x.shape[0]

        image_embeddings = self.image_encoder(x)
        
        otsu_mask = generate_otsu_mask(x).cuda()    
        otsu_mask = F.interpolate(otsu_mask,size=(256,256),mode='bilinear', align_corners=False)
        outputs_mask = []
        for idx in range(b):  # for each batch

            # get point and box
            point_coord, point_class, mask_shape, box_coord = PointGenerator(
                mask[idx].cpu().numpy(),
                visual= True
            )
            point_coord = torch.tensor(point_coord, device=x.device).float().unsqueeze(0)
            point_class = torch.tensor(point_class, device=x.device).long().unsqueeze(0)

            if mode == "point":
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(point_coord, point_class),
                    masks=None,  # 如果需要添加 masks 参数
                    boxes = None
                )

            elif mode == "box":
                box_coord = torch.tensor(box_coord, device=x.device).float()
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    masks=None,  # 如果需要添加 masks 参数
                    boxes=box_coord
                )
            elif mode == "mask":
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    masks=otsu_mask,  # 如果需要添加 masks 参数
                    boxes=None
                )
            
            print(f'dense_embeddings类型: {type(dense_embeddings)}, 形状: {dense_embeddings.shape}, 数据类型: {dense_embeddings.dtype}')
            low_res_masks = self.mask_decoder(
                low_level_feature = None,
                image_embeddings=image_embeddings[idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            masks = F.interpolate(
                low_res_masks[0],
                (self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            outputs_mask.append(masks.squeeze(0))

        return torch.stack(outputs_mask, dim=0)

class TinySAM(nn.Module):
    def __init__(
        self,
        data_path=None,
        img_size=1024,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    ):
        super(TinySAM, self).__init__()
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
            ###################################
        self.image_encoder = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        ###################################
        self.path = data_path
        self.img_size = img_size
        self.pt = ResizeLongestSide(img_size)
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.mask_threshold = 0.0
        self.image_format = "RGB"
        self.conv1_med = LowLevelFeatureExtractorMixed() #b*64*1024*1024
        #self.maxpool = nn.MaxPool2d(8) #b*64*128*128
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def forward(self, x, mask=None, img_id=None, mode="mask"):
        b = x.shape[0]
        #img = img.permute(0, 3, 1, 2)
        x = x.permute(0,3,1,2)
        #print(f'x类型: {type(x)}, 形状: {x.shape}, 数据类型: {x.dtype}')
        image_embeddings = self.image_encoder(x)
        #resized_x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        low_level_feature1, low_level_feature2 = self.conv1_med(x)
        #low_level_feature = self.maxpool(low_level_feature)
        
        outputs_mask = []
        otsu_mask = generate_otsu_mask(x)
        otsu_mask = F.interpolate(otsu_mask,size=(256,256),mode='bilinear', align_corners=False)
        otsu_mask = otsu_mask.cuda()
        #print(f'otsu_mask类型: {type(otsu_mask)}, 形状: {otsu_mask.shape}, 数据类型: {otsu_mask.dtype}')
        for idx in range(b):  # for each batch

            #get point and box
            
            

            if mode == "point":
                point_coord, point_class, mask_shape, box_coord = PointGenerator(
                mask[idx][0].cpu().numpy(),
                visual= False
                )
                point_coord = torch.tensor(point_coord, device=x.device).float().unsqueeze(0)
                point_class = torch.tensor(point_class, device=x.device).long().unsqueeze(0)
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(point_coord, point_class),
                    masks=None,  # 如果需要添加 masks 参数
                    boxes = None
                )

            # elif mode == "box":
            #     box_coord = torch.tensor(box_coord, device=x.device).float()
            #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
            #         points=None,
            #         masks=None,  # 如果需要添加 masks 参数
            #         boxes=box_coord
            #     )
            if mode == "mask":
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    #masks = None,
                    masks=otsu_mask[idx].unsqueeze(0),  # 如果需要添加 masks 参数
                    boxes=None
                )
            
            low_res_masks = self.mask_decoder(
                #low_level_feature = None,
                low_level_feature = [low_level_feature1[idx].unsqueeze(0), low_level_feature2[idx].unsqueeze(0)],
                image_embeddings=image_embeddings[idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks[0],
                (self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            outputs_mask.append(masks.squeeze(0))

        return torch.stack(outputs_mask, dim=0)
