import torch


def vis_to_e2cnn_order(tensor: torch.Tensor, contiguous=True):
    """
    E2CNN channels: (1) dim 0 = batch size, (2) dim 1 = group/repr dim, (3) last 2 dims = base space dim
    E.g. for C4, group action is to: 1. cyclically permute for (2) dim 1, and 2. spatially rotate for (3) last 2 dims
    in: (0) batch_size x [(1) map_width x (2) map_height] x [(3) |C_4| x (4) image_width x (5) image_height x (6) RGB]
    out: (0) batch_size x [(1) |C_4| x (2) image_width x (3) image_height x (4) RGB] x [(5) map_width x (6) map_height]
    """
    # without batch
    if tensor.ndim == 6:
        tensor = tensor.unsqueeze(0)
        print("(augment with batch dim)")
    assert tensor.ndim == 7

    tensor = tensor.permute(0, 3, 4, -2, -1, 1, 2)
    if contiguous:
        tensor = tensor.contiguous()
    return tensor


def e2cnn_to_vis_order(tensor: torch.Tensor, to_vis=True, contiguous=True):
    """
    Reverse operation of last one, for easier visualization
    in: batch_size x [|C_4| x image_width x image_height x RGB] x [map_width x map_height]
    out: batch_size x [map_width x map_height] x [|C_4| x image_width x image_height x RGB]
    """
    if tensor.ndim == 6:
        tensor = tensor.unsqueeze(0)
        print("(augment with batch dim)")
    assert tensor.ndim == 7

    tensor = tensor.permute(0, -2, -1, 1, 2, 3, 4)
    if contiguous:
        tensor = tensor.contiguous()
    return tensor.to(dtype=torch.uint8) if to_vis else tensor


def vis_to_conv2d(tensor: torch.Tensor):
    """
    in: (1) node_num x [(2) |C_4| x (3) image_width x (4) image_height x (5) RGB]
    out: (0) (node_num * |C_4|) x [(1) RGB x (2) image_width x (3) image_height]
    """
    if tensor.ndim == 5:
        node_num, num_views, img_height, img_width, img_rgb = tensor.shape
        tensor = tensor.view(node_num * num_views, img_rgb, img_height, img_width)

    elif tensor.ndim == 7:
        (
            batch_size,
            map_height,
            map_width,
            num_views,
            img_height,
            img_width,
            img_rgb,
        ) = tensor.size()
        tensor = tensor.view(
            batch_size * map_height * map_width * num_views,
            img_rgb,
            img_height,
            img_width,
        )

    else:
        raise Exception("Unknown batch dimension")

    return tensor


def flatten_repr_channel(tensor: torch.Tensor, group_size: int):
    """
    in: batch_size x num_views x map_height x map_width x img_embed_dim
    """

    batch_size, num_views, map_height, map_width, img_embed_dim = tensor.size()
    assert group_size == num_views

    out_tensor = torch.empty(
        batch_size, num_views * img_embed_dim, map_height, map_width
    ).to(tensor.device)

    # > compute indices
    base_indices = torch.arange(start=0, end=num_views * img_embed_dim, step=num_views)
    view2indices = {v: (base_indices + v) for v in range(num_views)}

    for i in range(group_size):
        repr_indices = view2indices[i]
        repr_tensor = tensor[:, i, :, :, :].view(
            batch_size, img_embed_dim, map_height, map_width
        )
        out_tensor[:, repr_indices, :, :] = repr_tensor

    return out_tensor
