import escnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as esnn
from escnn.nn import FieldType
from torchvision.models import resnet18, ResNet18_Weights

from diffplan.utils import helpers_esnn as h
from diffplan.utils.tensor_transform import vis_to_conv2d, flatten_repr_channel


# from .pos_encod import PositionalEncoding


class NavMapper(nn.Module):
    def __init__(
        self,
        map_height,
        map_width,
        num_views,
        img_height,
        img_width,
        workspace_size=None,
    ):
        super().__init__()

        assert num_views == 4
        assert img_height == img_width

        self.map_height, self.map_width = map_height, map_width
        self.num_views = num_views
        self.img_height, self.img_width = img_height, img_width
        self.img_rgb = 3

        self.hidden_dim = [32, 64]
        self.img_embed_dim = 256
        self.map_dim = 1

        # step 1: use regular CNN to embed 4 images
        # input = [batch, map_height, map_width, (4, img_height, img_width, RGB)]
        # output = [batch, map_height, map_width, (4, img_embedding)]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(
                self.img_rgb, self.hidden_dim[0], kernel_size=10, stride=4, padding=4
            ),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dim[0],
                self.hidden_dim[1],
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(self.hidden_dim[1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.img_embed_dim, kernel_size=4, stride=1),
            nn.BatchNorm2d(self.img_embed_dim),
            nn.ReLU(),
        )

        self.map_kernel_sizes = []

        # step 2: use regular CNN to process 4 embeddings again
        # input = [batch, map_height, map_width, (4, img_embedding)]
        # output = [batch, map_height, map_width, (obstacle_embedding)]
        self.map_encoder = nn.Sequential(
            nn.Conv2d(
                self.img_embed_dim * self.num_views,
                self.hidden_dim[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dim[0], self.map_dim, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x):
        (
            batch_size,
            map_height,
            map_width,
            num_views,
            img_height,
            img_width,
            img_rgb,
        ) = x.size()
        assert (map_height, map_width, num_views, img_height, img_width, img_rgb) == (
            self.map_height,
            self.map_width,
            self.num_views,
            self.img_height,
            self.img_width,
            3,
        )

        # > flatten map dimensions, process individual images, shared across batch/locations/views
        x = vis_to_conv2d(x)
        # > size = (0) (batch_size * map_width * map_height * |C_4|) x [(1) RGB x (2) image_width x (3) image_height]

        # > step 1
        x = self.img_encoder(x)
        # size = (0) (batch_size * map_width * map_height) x (1) |C_4| x (2) image_embedding

        # > reshape for processing over map, in two steps
        x = x.view(batch_size, num_views, map_height, map_width, self.img_embed_dim)
        x = x.view(batch_size, num_views * self.img_embed_dim, map_height, map_width)

        # > step 2
        x = self.map_encoder(x)
        x = x.view(batch_size, self.map_dim, map_height, map_width)

        return x


class GraphNavMapper(nn.Module):
    def __init__(self, num_views, img_height, img_width, workspace_size=None):
        super().__init__()

        assert num_views == 4
        assert img_height == img_width

        self.num_views = num_views
        self.img_height, self.img_width = img_height, img_width
        self.img_rgb = 3

        self.hidden_dim = [32, 64]
        self.img_embed_dim = 256
        self.map_dim = 1

        # step 1: use regular CNN to embed 4 images
        # input = [node_num, (4, img_height, img_width, RGB)]
        # output = [node_num, (4, img_embedding)]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(
                self.img_rgb, self.hidden_dim[0], kernel_size=10, stride=4, padding=4
            ),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dim[0],
                self.hidden_dim[1],
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(self.hidden_dim[1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.img_embed_dim, kernel_size=4, stride=1),
            nn.BatchNorm2d(self.img_embed_dim),
            nn.ReLU(),
        )

        self.map_kernel_sizes = []

        # step 2: use regular CNN to process 4 embeddings again
        # input = [batch, map_height, map_width, (4, img_embedding)]
        # output = [batch, map_height, map_width, (obstacle_embedding)]
        # self.map_encoder = nn.Sequential(
        #     nn.Conv2d(self.img_embed_dim * self.num_views, self.hidden_dim[0],
        #               kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.hidden_dim[0]),
        #     nn.ReLU(),
        #     nn.Conv2d(self.hidden_dim[0], self.map_dim,
        #               kernel_size=3, stride=1, padding=1),
        # )

        # LHY: for graph case, try MLP
        self.map_encoder = nn.Sequential(
            nn.Linear(self.img_embed_dim * self.num_views, self.hidden_dim[0]),
            nn.BatchNorm1d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dim[0], self.map_dim),
        )

    def forward(self, x):
        node_num, num_views, img_height, img_width, img_rgb = x.shape

        # > flatten map dimensions, process individual images, shared across batch/locations/views
        x = vis_to_conv2d(x)
        # > size = (0) (node_num * |C_4|) x [(1) RGB x (2) image_width x (3) image_height]

        # > step 1
        x = self.img_encoder(x)
        # size = (0) (batch_size * map_width * map_height) x (1) |C_4| x (2) image_embedding

        # > reshape for processing over map, in two steps
        x = x.view(node_num, num_views * self.img_embed_dim)

        # > step 2
        x = self.map_encoder(x)

        return x


class SymNavMapper(nn.Module):
    def __init__(
        self,
        map_height,
        map_width,
        num_views,
        img_height,
        img_width,
        workspace_size=None,
        geo_output=False,
    ):
        super().__init__()

        assert num_views == 4
        assert img_height == img_width

        self.map_height, self.map_width = map_height, map_width
        self.num_views = num_views
        self.img_height, self.img_width = img_height, img_width
        self.img_rgb = 3
        self.geo_output = geo_output

        self.hidden_dim = [32, 64]
        self.img_embed_dim = 256
        self.map_dim = 1

        # Step 1: use regular CNN to embed 4 images
        # input = [batch, map_height, map_width, (4, img_height, img_width, RGB)]
        # output = [batch, map_height, map_width, (4, img_embedding)]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(
                self.img_rgb, self.hidden_dim[0], kernel_size=10, stride=4, padding=4
            ),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dim[0],
                self.hidden_dim[1],
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(self.hidden_dim[1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.img_embed_dim, kernel_size=4, stride=1),
            nn.BatchNorm2d(self.img_embed_dim),
            nn.ReLU(),
        )

        # Step 2: use E(2)-CNN to process 4 embeddings again
        # Note that dim 1 is group/repr dim, dim -2 & -1 are base space dims
        # input = [batch, 4, img_embedding, (map_height, map_width)]
        # output = [batch, obstacle_embedding, (map_height, map_width)]
        # NOTE: 4 images -> C4, for other number, need to use quotient repr (or induced repr?)
        self.r2_space = gspaces.Rot2dOnR2(4)
        self.spatial_in_type = FieldType(
            self.r2_space, self.img_embed_dim * [self.r2_space.regular_repr]
        )
        self.spatial_hid_type = FieldType(
            self.r2_space, self.hidden_dim[0] * [self.r2_space.regular_repr]
        )
        self.spatial_out_type = FieldType(
            self.r2_space, 1 * [self.r2_space.trivial_repr]
        )
        self.group_size = self.r2_space.regular_repr.size

        self.map_encoder = esnn.SequentialModule(
            esnn.R2Conv(
                self.spatial_in_type,
                self.spatial_hid_type,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            esnn.InnerBatchNorm(self.spatial_hid_type),
            esnn.ReLU(self.spatial_hid_type),
            esnn.R2Conv(
                self.spatial_hid_type,
                self.spatial_out_type,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        (
            batch_size,
            map_height,
            map_width,
            num_views,
            img_height,
            img_width,
            img_rgb,
        ) = x.size()
        assert (map_height, map_width, num_views, img_height, img_width, img_rgb) == (
            self.map_height,
            self.map_width,
            self.num_views,
            self.img_height,
            self.img_width,
            3,
        )

        # > flatten map dimensions, process individual images, shared across batch/locations/views
        x = vis_to_conv2d(x)
        # size = (0) (batch_size * map_width * map_height * |C_4|) x [(1) RGB x (2) image_width x (3) image_height]

        # > step 1
        x = self.img_encoder(x)
        # size = (0) (batch_size * map_width * map_height) x (1) |C_4| x (2) image_embedding

        # > reshape for processing over map, in two steps
        x = x.view(batch_size, num_views, map_height, map_width, self.img_embed_dim)
        x = flatten_repr_channel(x, group_size=self.group_size)

        # > step 2
        x_geo = esnn.GeometricTensor(x, self.spatial_in_type)
        x_geo = self.map_encoder(x_geo)

        # > return
        if self.geo_output:
            return x_geo
        else:
            x_out = x_geo.tensor
            x_out = x_out.view(batch_size, self.map_dim, map_height, map_width)
            return x_out


class InducedSymNavMapper(nn.Module):
    def __init__(
        self,
        map_height,
        map_width,
        img_height,
        img_width,
        out_type=None,
        out_group=None,
        out_dim=1,
        num_views=4,
        wrap_output=False,
    ):
        super().__init__()

        assert num_views == 4
        assert img_height == img_width

        self.map_height, self.map_width = map_height, map_width
        self.num_views = num_views
        self.img_height, self.img_width = img_height, img_width
        self.img_channel = 3
        self.out_dim = out_dim
        self.geo_output = wrap_output

        self.hidden_dim = [32, 64]
        self.img_embed_dim = 256
        self.map_dim = 1

        # Step 1: use regular CNN to embed 4 images
        # input = [batch, map_height, map_width, (4, img_height, img_width, RGB)]
        # output = [batch, map_height, map_width, (4, img_embedding)]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(
                self.img_channel,
                self.hidden_dim[0],
                kernel_size=10,
                stride=4,
                padding=4,
            ),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dim[0],
                self.hidden_dim[1],
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(self.hidden_dim[1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.img_embed_dim, kernel_size=4, stride=1),
            nn.BatchNorm2d(self.img_embed_dim),
            nn.ReLU(),
        )

        # Step 2: learnable mapping to convert to regular repr
        # Input is 4-direction 90-degree images -> hardcoded to be C4
        in_group = escnn.group.cyclic_group(num_views)
        # Only cyclically permuting the input images (facing different directions), so no base space
        in_space = escnn.gspaces.no_base_space(in_group)
        # NOTE: hardcode
        self.in_type = in_space.type(
            *[in_group.regular_representation] * self.img_embed_dim
        )

        # Note: output tensor is used in planner
        if out_type is not None:
            self.out_type = out_type
            self.out_group = str(out_type.group)

            out_space = out_type.gspace
            out_group = out_space.fibergroup

            out_fiber_repr = out_type.representation
        else:
            self.out_group = h.get_group(out_group)

            out_space = gspaces.no_base_space(self.out_group)
            # NOTE: hardcode dim; should be number of image features - out_dim?
            out_fiber_repr = [out_group.regular_representation] * self.hidden_dim[0]
            self.out_type = out_space.type(*out_fiber_repr)

        if str(self.out_group) == "C4":
            # For C4 output, use C4 regular repr directly
            restrict_repr = out_fiber_repr
        elif str(self.out_group).startswith("C"):
            # Restrict from C_n to C4 subgroup (e.g., C8 -> C4)
            restrict_repr = self.out_group.restrict_representation(
                id=4,  # reduce rotation component from C_n to C4 subgroup
                repr=self.out_type.representation,  # use e.g. regular repr in C_n (no quotient)
            )
        elif str(self.out_group).startswith("D"):
            # Restrict directly from D_n to C4 subgroup (e.g., D8 -> C4)
            # (Or:) Quotient D_n by reflection (e.g., D8 -> C8), then restrict to C4 subgroup (e.g., C8 -> C4)
            restrict_repr = self.out_group.restrict_representation(
                id=(None, 4),  # reduce rotation component from D_n to C4 subgroup
                # repr=out_group.quotient_representation((0, 1)),  # quotient out reflection in D_n
                repr=self.out_type.representation,  # use e.g. regular repr in D_n (no quotient)
            )
        else:
            raise ValueError
        self.restrict_out_type = in_space.type(restrict_repr)

        # NOTE: change to conv? check SymNavMapper
        self.fc_induced = esnn.Linear(
            in_type=self.in_type, out_type=self.restrict_out_type
        )

        # Step 3: map encoder - for graph case, only use linear
        self.spatial_hid_type = out_space.type(
            *[out_space.regular_repr] * self.hidden_dim[0]
        )
        self.spatial_out_type = out_space.type(*[out_space.trivial_repr] * 1)
        # NOTE: output hardcode to be trivial, but can be any based on what planners need

        self.map_encoder = esnn.SequentialModule(
            esnn.R2Conv(
                self.spatial_in_type,
                self.spatial_hid_type,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            esnn.InnerBatchNorm(self.spatial_hid_type),
            esnn.ReLU(self.spatial_hid_type),
            esnn.R2Conv(
                self.spatial_hid_type,
                self.spatial_out_type,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        (
            batch_size,
            map_height,
            map_width,
            num_views,
            img_height,
            img_width,
            img_rgb,
        ) = x.size()
        assert (map_height, map_width, num_views, img_height, img_width, img_rgb) == (
            self.map_height,
            self.map_width,
            self.num_views,
            self.img_height,
            self.img_width,
            3,
        )

        # > flatten map dimensions, process individual images, shared across batch/locations/views
        x = vis_to_conv2d(x)
        # size = (0) (batch_size * map_width * map_height * |C_4|) x [(1) RGB x (2) image_width x (3) image_height]

        # Step 1
        x = self.img_encoder(x)
        # size = (0) (batch_size * map_width * map_height) x (1) |C_4| x (2) image_embedding

        # > reshape for processing over map, in two steps
        x = x.view(batch_size, num_views, map_height, map_width, self.img_embed_dim)
        # NOTE: should be equivalent to `view`
        x_flat = flatten_repr_channel(x, group_size=self.group_size)

        # Step 2 - convert to regular representation
        # Go through a learned induction (FC) layer that outputs *restricted* out_repr
        x_in = self.in_type(x_flat)

        x_induced = self.fc_induced(x_in)
        x_unwrapped = x_induced.tensor

        x_unwrapped = torch.relu(x_unwrapped)
        x_wrapped = self.out_type(x_unwrapped)

        x_geo = self.map_encoder(x_wrapped)

        # > return
        if self.geo_output:
            return x_geo
        else:
            x_out = x_geo.tensor
            x_out = x_out.view(batch_size, self.map_dim, map_height, map_width)
            return x_out


def flatten_fiber_channel(img_feat, group_size=4):
    """
    Flatten fiber channel for regular representation of cyclic group (with `group_size` = number of views)
    based on `flatten_repr_channel` in  `tensor_transform.py`
    - input: (node num, feat size, 4)
    - output: (node num, feat size * 4) - compatible with regular repr
    """

    batch_size, img_embed_dim, num_views = img_feat.size()
    assert group_size == num_views

    """
    out_tensor = torch.empty(batch_size, num_views * img_embed_dim).to(img_feat.device)

    # > compute indices
    base_indices = torch.arange(start=0, end=num_views * img_embed_dim, step=num_views)
    view2indices = {v: (base_indices + v) for v in range(num_views)}

    for i in range(group_size):
        repr_indices = view2indices[i]
        repr_tensor = img_feat[:, :, i].view(batch_size, img_embed_dim)
        out_tensor[:, repr_indices] = repr_tensor
    """

    # NOTE: should be equivalent
    flat_feat = img_feat.view(batch_size, img_embed_dim * num_views)

    return flat_feat


class InducedGraphNavMapper(nn.Module):
    def __init__(
        self,
        img_height,
        img_width,
        out_type=None,
        out_group=None,
        out_dim=None,
        num_views=4,
        wrap_output=False,
    ):
        super().__init__()

        self.num_views = num_views
        self.img_height, self.img_width = img_height, img_width
        self.img_channel = 3
        self.out_dim = out_dim
        self.geo_output = wrap_output

        self.hidden_dim = [32, 64]
        self.img_embed_dim = 256  # NOTE: hardcode, check
        self.map_dim = 1

        # Step 1: use regular CNN to embed 4 images
        # input = [batch, map_height, map_width, (4, img_height, img_width, RGB)]
        # output = [batch, map_height, map_width, (4, img_embedding)]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(
                self.img_channel,
                self.hidden_dim[0],
                kernel_size=10,
                stride=4,
                padding=4,
            ),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dim[0],
                self.hidden_dim[1],
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(self.hidden_dim[1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.img_embed_dim, kernel_size=4, stride=1),
            nn.BatchNorm2d(self.img_embed_dim),
            nn.ReLU(),
        )

        # Step 2: learnable mapping to convert to regular repr
        # Input is 4-direction 90-degree images -> hardcoded to be C4
        in_group = escnn.group.cyclic_group(num_views)
        # Only cyclically permuting the input images (facing different directions), so no base space
        in_space = escnn.gspaces.no_base_space(in_group)
        # NOTE: hardcode dim
        self.in_type = in_space.type(
            *[in_group.regular_representation] * self.img_embed_dim
        )

        # NOTE: output tensor is used in planner
        if out_type is not None:
            self.out_type = out_type
            self.out_group = str(out_type.group)

            out_space = out_type.gspace
            out_group = out_space.fibergroup

            out_fiber_repr = out_type.representation
        else:
            self.out_group = h.get_group(out_group)

            out_space = gspaces.no_base_space(self.out_group)
            # NOTE: hardcode dim; should be number of image features - out_dim?
            out_fiber_repr = [out_group.regular_representation] * self.hidden_dim[0]
            self.out_type = out_space.type(*out_fiber_repr)

        if str(self.out_group) == "C4":
            # For C4 output, use C4 regular repr directly
            restrict_repr = out_fiber_repr
        elif str(self.out_group).startswith("C"):
            # Restrict from C_n to C4 subgroup (e.g., C8 -> C4)
            restrict_repr = self.out_group.restrict_representation(
                id=4,  # reduce rotation component from C_n to C4 subgroup
                repr=self.out_type.representation,  # use e.g. regular repr in C_n (no quotient)
            )
        elif str(self.out_group).startswith("D"):
            # Restrict directly from D_n to C4 subgroup (e.g., D8 -> C4)
            # (Or:) Quotient D_n by reflection (e.g., D8 -> C8), then restrict to C4 subgroup (e.g., C8 -> C4)
            restrict_repr = self.out_group.restrict_representation(
                id=(None, 4),  # reduce rotation component from D_n to C4 subgroup
                # repr=out_group.quotient_representation((0, 1)),  # quotient out reflection in D_n
                repr=self.out_type.representation,  # use e.g. regular repr in D_n (no quotient)
            )
        else:
            raise ValueError
        self.restrict_out_type = in_space.type(restrict_repr)

        self.fc_induced = esnn.Linear(
            in_type=self.in_type, out_type=self.restrict_out_type
        )

        # Step 3: map encoder - for graph case, only use linear
        self.spatial_hid_type = out_space.type(
            *[out_space.regular_repr] * self.hidden_dim[0]
        )
        self.spatial_out_type = out_space.type(*[out_space.trivial_repr] * 1)
        # NOTE: output hardcode to be trivial, but can be any based on what planners need

        self.map_encoder = esnn.SequentialModule(
            # NOTE: take input of out_type of the induced layer
            esnn.Linear(self.out_type, self.spatial_hid_type),
            # NOTE: use 1D batch norm
            # esnn.InnerBatchNorm(self.spatial_hid_type),
            esnn.ReLU(self.spatial_hid_type),
            esnn.Linear(self.spatial_hid_type, self.spatial_out_type),
        )

    def forward(self, x, wrap_out=False):
        node_num, num_views, img_height, img_width, img_rgb = x.shape

        # > flatten map dimensions, process individual images, shared across batch/locations/views
        x = vis_to_conv2d(x)
        # > size = (0) (node_num * |C_4|) x [(1) RGB x (2) image_width x (3) image_height]

        # Step 1
        x = self.img_encoder(x)
        # > size = (0) (batch_size * map_width * map_height) x (1) |C_4| x (2) image_embedding
        # NOTE: check

        # > reshape for processing over map, in two steps
        x_flat = x.view(node_num, num_views * self.img_embed_dim)

        # Step 2 - convert to regular representation
        # Go through a learned induction (FC) layer that outputs *restricted* out_repr
        x_in = self.in_type(x_flat)

        x_induced = self.fc_induced(x_in)
        x_unwrapped = x_induced.tensor

        x_unwrapped = torch.relu(x_unwrapped)
        x_wrapped = self.out_type(x_unwrapped)

        x_out = self.map_encoder(x_wrapped)

        # Obtain the raw tensor, transformed in *given* out_repr
        x_out = x_out.tensor
        # NOTE: hardcode to unwrap tensor; just trivial repr

        return x_out


class GraphImgEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc_induced = nn.Linear(512, out_dim)

    def forward(self, sample):
        img_0 = ResNet18_Weights.IMAGENET1K_V1.transforms()(
            sample.img_0[..., :3].permute(0, 3, 1, 2)
        )
        img_1 = ResNet18_Weights.IMAGENET1K_V1.transforms()(
            sample.img_1[..., :3].permute(0, 3, 1, 2)
        )
        img_2 = ResNet18_Weights.IMAGENET1K_V1.transforms()(
            sample.img_2[..., :3].permute(0, 3, 1, 2)
        )
        img_3 = ResNet18_Weights.IMAGENET1K_V1.transforms()(
            sample.img_3[..., :3].permute(0, 3, 1, 2)
        )
        # img_combined = torch.cat([img_0, img_1, img_3, img_2])
        img_combined = torch.cat([img_0, img_2, img_3, img_1])
        N = img_0.shape[0]
        img_feat = self.resnet(img_combined)
        img_feat = F.dropout(img_feat, 0.2)
        # Transform into following format
        # dimension: (125 * 4), 512; after flatten = 500, 512

        img_feat_cat = torch.stack(
            [
                img_feat[:N, :],
                img_feat[N : 2 * N, :],
                img_feat[2 * N : 3 * N, :],
                img_feat[3 * N :, :],
            ],
            dim=-1,
        )
        # NOTE: check image feature dim, no idea what they look like
        img_feat_flat = flatten_fiber_channel(img_feat_cat)
        # Expect size = (batch/node size, feat size * 4), in regular repr order

        return img_feat_flat


class SymGraphImgEncoder(nn.Module):
    """
    Steps:
    1 -- use pretrained ResNet
    2 -- we can choose output, should be regular repr (if we want to keep equivariant features)
    """

    def __init__(self, out_type=None, out_group=None, out_dim=None, in_views=4):
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # NOTE: Use a dummy layer, later use an "induction" layer
        self.resnet.fc_induced = nn.Identity()
        # self.resnet.fc = nn.Linear(512, visual_feat_dim)
        # NOTE: hardcode
        self.resnet.fc_induced = nn.Linear(512, 128)

        # Input is 4-direction 90-degree images -> hardcoded to be C4
        in_group = escnn.group.cyclic_group(in_views)
        # Only cyclically permuting the input images (facing different directions), so no base space
        in_space = escnn.gspaces.no_base_space(in_group)
        # NOTE: hardcode dim
        self.in_type = in_space.type(*[in_group.regular_representation] * 128)

        # NOTE: output tensor is used in planner
        if out_type is not None:
            self.out_type = out_type
            self.out_group = str(out_type.group)

            out_space = out_type.gspace
            out_group = out_space.fibergroup

            out_fiber_repr = out_type.representation
        else:
            self.out_group = h.get_group(out_group)

            out_space = gspaces.no_base_space(self.out_group)
            out_fiber_repr = [self.out_group.regular_representation] * out_dim
            # = number of image features
            # out_fiber_repr = [out_group.regular_representation] * visual_feat_dim
            self.out_type = out_space.type(*out_fiber_repr)

        if str(self.out_group) == "C4":
            # For C4 output, use C4 regular repr directly
            restrict_repr = out_fiber_repr
        elif str(self.out_group).startswith("C"):
            # Restrict from C_n to C4 subgroup (e.g., C8 -> C4)
            restrict_repr = self.out_group.restrict_representation(
                id=4,  # reduce rotation component from C_n to C4 subgroup
                repr=self.out_type.representation,  # use e.g. regular repr in C_n (no quotient)
            )
        elif str(self.out_group).startswith("D"):
            # Restrict directly from D_n to C4 subgroup (e.g., D8 -> C4)
            # (Or:) Quotient D_n by reflection (e.g., D8 -> C8), then restrict to C4 subgroup (e.g., C8 -> C4)
            restrict_repr = self.out_group.restrict_representation(
                id=(None, 4),  # reduce rotation component from D_n to C4 subgroup
                # repr=out_group.quotient_representation((0, 1)),  # quotient out reflection in D_n
                repr=self.out_type.representation,  # use e.g. regular repr in D_n (no quotient)
            )
        else:
            raise ValueError
        self.restrict_out_type = in_space.type(restrict_repr)

        self.fc = esnn.Linear(in_type=self.in_type, out_type=self.restrict_out_type)

    def forward(self, sample, wrap_out=False):
        img_0 = ResNet18_Weights.IMAGENET1K_V1.transforms()(
            sample.img_0[..., :3].permute(0, 3, 1, 2)
        )
        img_1 = ResNet18_Weights.IMAGENET1K_V1.transforms()(
            sample.img_1[..., :3].permute(0, 3, 1, 2)
        )
        img_2 = ResNet18_Weights.IMAGENET1K_V1.transforms()(
            sample.img_2[..., :3].permute(0, 3, 1, 2)
        )
        img_3 = ResNet18_Weights.IMAGENET1K_V1.transforms()(
            sample.img_3[..., :3].permute(0, 3, 1, 2)
        )

        # NOTE: need to be counter-clockwise, aligning with the generator of cyclic groups in escnn
        img_combined = torch.cat([img_0, img_2, img_3, img_1])
        N = img_0.shape[0]
        img_feat = self.resnet(img_combined)
        img_feat = F.dropout(img_feat, 0.2)
        # Transform into following format
        # NOTE: (125 * 4), 512; after flatten = 500, 512

        img_feat_cat = torch.stack(
            [
                img_feat[:N, :],
                img_feat[N : 2 * N, :],
                img_feat[2 * N : 3 * N, :],
                img_feat[3 * N :, :],
            ],
            dim=-1,
        )

        # NOTE: hardcode a ReLU
        img_feat_cat = torch.relu(img_feat_cat)

        # NOTE: check image feature dim, check what they look like
        img_feat_flat = flatten_fiber_channel(img_feat_cat)
        # Expect size: (batch/node size, feat size * 4), in regular repr order

        # Go through a learned induction (FC) layer that outputs *restricted* out_repr
        img_feat_wrapped = self.in_type(img_feat_flat)
        out = self.fc(img_feat_wrapped)
        # Obtain the raw tensor, transformed in *given* out_repr
        out = out.tensor
        # Optionally wrap the tensor in a *given* out_repr
        if wrap_out:
            out = self.out_type(out)

        return out
