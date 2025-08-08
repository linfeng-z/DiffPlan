from typing import Tuple, List

import escnn
import torch
from escnn import nn as esnn
from escnn.group import Representation
from escnn.gspaces import GSpace


class GroupReducedMaxOperation(esnn.EquivariantModule):
    def __init__(
        self,
        g_space: GSpace,
        in_repr: List[Representation],
        out_repr: List[Representation] = None,
        out_mode: str = None,
        no_base: bool = False,
        **kwargs,
    ):
        assert isinstance(g_space, escnn.gspaces.GSpace)
        super().__init__()

        self.no_base = no_base
        self.space = g_space

        in_type = g_space.type(*in_repr)
        self.in_type = in_type

        # > assert all representations are the same
        self.in_repr_0 = in_repr[0]
        # self.in_repr_0 = in_type.representations[0]
        self.num_repr = len(in_type.representations)
        print(f"> type of: {self.in_repr_0}, # of repr: {self.num_repr}")
        for _repr in in_type.representations:
            assert self.in_repr_0 == _repr

        # > build the output representation with only a regular/trivial repr
        self.out_mode = out_mode

        if out_mode is None:
            assert out_repr is not None
            self.out_repr_0 = out_repr[0]

            # > decide mode
            if self.out_repr_0 == self.in_repr_0:
                self.out_mode = "keep"
            elif self.out_repr_0 == in_type.gspace.trivial_repr:
                self.out_mode = "reduce"
            else:
                raise ValueError

        elif out_mode == "keep":
            # > mode 1: keep the repr (usually regular)
            self.out_repr_0 = self.in_repr_0

        elif out_mode == "reduce":
            # > mode 2: also do group pooling (just trivial repr)
            self.out_repr_0 = in_type.gspace.trivial_repr

        else:
            raise ValueError

        self.out_type = esnn.FieldType(self.space, 1 * [self.out_repr_0])

    def forward(self, input: esnn.GeometricTensor) -> esnn.GeometricTensor:
        assert input.type == self.in_type

        input = input.tensor
        if self.no_base:
            b, c = input.shape
        else:
            b, c, h, w = input.shape

        if self.out_mode == "keep":
            # > split the field along the group channel dim into 2 dimensions:
            if self.no_base:
                output = input.view(b, self.num_repr, self.in_repr_0.size)
            else:
                output = input.view(b, self.num_repr, self.in_repr_0.size, h, w)  # noqa
        elif self.out_mode == "reduce":
            if self.no_base:
                output = input.view(b, self.num_repr * self.in_repr_0.size)
            else:
                output = input.view(
                    b, self.num_repr * self.in_repr_0.size, 1, h, w
                )  # noqa
        else:
            raise ValueError

        # > max along the repr channel
        output, _ = torch.max(output, 1, keepdim=(self.out_mode == "reduce"))

        # > check result
        assert self.evaluate_output_shape(input.shape) == output.shape

        # > wrap the result in a GeometricTensor
        return esnn.GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.no_base:
            assert len(input_shape) == 2
        else:
            assert len(input_shape) == 4

        # > the group channel will be maxed out and pushed to 1 of regular/trivial repr
        if self.no_base:
            b, c = input_shape
            assert c == self.in_type.size
            return b, self.out_repr_0.size
        else:
            b, c, hi, wi = input_shape
            assert c == self.in_type.size
            return b, self.out_repr_0.size, hi, wi

    def export(self):
        # use MaxPoolChannels
        raise NotImplementedError
