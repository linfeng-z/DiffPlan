import torch


class FullObsTensor:
    def __init__(self, in_tensor: torch.Tensor, in_type, reprs):
        self.in_tensor = in_tensor
        self.in_type = in_type
        self.reprs = reprs

        assert in_tensor.dtype == torch.uint8
        assert all([repr_ == reprs[0] for repr_ in reprs])
        self.repr_size = reprs[0].size

        (
            self.batch_size,
            self.map_wi,
            self.map_hi,
            self.num_views,
            self.img_wi,
            self.img_hi,
            self.img_channel,
        ) = in_tensor.size()
        assert self.num_views == self.repr_size
        self.img_size = self.img_wi * self.img_hi * self.img_channel

        self.flatten = self.flatten_repr_grouped()

    def _get_input_indices(self):
        base_indices = torch.arange(
            start=0, end=self.num_views * self.img_size, step=self.num_views
        )
        view2indices = {v: (base_indices + v) for v in range(self.num_views)}
        return view2indices

    def _get_output_indices(self):
        """
        Invert the above function `_get_input_indices`
        """
        pass

    def flatten_view_grouped(self) -> torch.Tensor:
        """
        (Deprecated)
        Flatten along the group channel, grouped by *views*
        Example: ([view1 repr1, ..., view1 reprN], ..., [view4 repr1, ..., view4 reprN])
        It is grouped by 4 groups of size N = (|C_4| * image_width * image_height * RGB)

        in shape = batch_size x [map_width x map_height] x [|C_4| x image_width x image_height x RGB]
        out shape = batch_size x (|C_4| * image_width * image_height * RGB) x map_width x map_height
        """

        out_tensor = torch.empty(
            self.batch_size, self.num_views * self.img_size, self.map_wi, self.map_hi
        )

        # > the group channel, the images are contiguous
        for i in range(self.repr_size):
            id_start = i * self.img_size
            id_end = (i + 1) * self.img_size
            reshaped_tensor = self.in_tensor[:, :, :, i, :, :, :].reshape(
                self.batch_size, self.img_size, self.map_wi, self.map_hi
            )
            out_tensor[:, id_start:id_end, :, :] = reshaped_tensor

        return out_tensor

    def flatten_repr_grouped(self) -> torch.Tensor:
        """
        Flatten along the group channel, grouped by *representations*
        Example: ([repr1 view1, ..., repr1 view4], [repr2 view1, ..., repr2 view4], ...)
        It is grouped by N groups of size 4, where N = (|C_4| * image_width * image_height * RGB)

        in shape = batch_size x [map_width x map_height] x [|C_4| x image_width x image_height x RGB]
        out shape = batch_size x (|C_4| * image_width * image_height * RGB) x map_width x map_height
        """

        out_tensor = torch.empty(
            self.batch_size,
            self.num_views * self.img_size,
            self.map_wi,
            self.map_hi,
            dtype=torch.uint8,
        )
        view2indices = self._get_input_indices()

        # > key: keep num_views x img_size dimensions ready
        repr_grouped_tensor = self.in_tensor.reshape(
            self.batch_size, self.num_views, self.img_size, self.map_wi, self.map_hi
        )

        # > the group channel, the images are contiguous
        for i in range(self.repr_size):
            repr_tensor = repr_grouped_tensor[:, i, ...]
            repr_indices = view2indices[i]
            out_tensor[:, repr_indices, :, :] = repr_tensor

        return out_tensor

    def unflatten(self, flattened_in=None):
        """
        Inverse operation of flatten - to visualize the group transformation is compatible and meaningful
        in shape = batch_size x (|C_4| * image_width * image_height * RGB) x map_width x map_height
        out shape = batch_size x [map_width x map_height] x [|C_4| x image_width x image_height x RGB]
        """

        if flattened_in is None:
            flattened_tensor = self.flatten
        else:
            flattened_tensor = flattened_in

        # > intermediate out tensor
        repr_grouped_out_tensor = torch.empty(
            self.batch_size,
            self.num_views,
            self.img_size,
            self.map_wi,
            self.map_hi,
            dtype=torch.uint8,
        )
        view2indices = self._get_input_indices()

        # > the group channel, the images are contiguous
        for i in range(self.repr_size):
            repr_indices = view2indices[i]
            repr_grouped_out_tensor[:, i, ...] = flattened_tensor[:, repr_indices, :, :]

            print(flattened_tensor[:, repr_indices, :, :].size())

        out_tensor = repr_grouped_out_tensor.reshape(self.in_tensor.size())

        return out_tensor

    def test_compatibility(self):
        """
        Test compatibility with e2cnn group transformation (base + fiber)
        needed shape = batch_size x (|C_4| * image_width * image_height * RGB) x map_width x map_height
        """
        pass
