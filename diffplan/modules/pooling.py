from typing import Tuple

import torch
from escnn.group import Representation
# from escnn.gspaces import GeneralOnR2
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.nn.modules.equivariant_module import EquivariantModule


class GroupReducedMaxPooling(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_mode: str = None,
        out_repr: Representation = None,
        **kwargs,
    ):
        # assert isinstance(in_type.gspace, GeneralOnR2)
        super().__init__()

        self.space = in_type.gspace
        self.in_type = in_type

        # > assert all representations are the same
        self.in_repr = in_type.representations[0]
        self.num_repr = len(in_type.representations)
        print(f"> type of: {self.in_repr}, # of repr: {self.num_repr}")
        for _repr in in_type.representations:
            assert self.in_repr == _repr

        # > build the output representation with only a regular/trivial repr
        self.out_mode = out_mode

        if out_mode is None:
            assert out_repr is not None
            self.out_repr = out_repr

            # > decide mode
            if self.out_repr == self.in_repr:
                self.out_mode = "keep"
            elif self.out_repr == in_type.gspace.trivial_repr:
                self.out_mode = "reduce"
            else:
                raise ValueError

        elif out_mode == "keep":
            # > mode 1: keep the repr (usually regular)
            self.out_repr = self.in_repr

        elif out_mode == "reduce":
            # > mode 2: also do group pooling (just trivial repr)
            self.out_repr = in_type.gspace.trivial_repr

        else:
            raise ValueError

        self.out_type = FieldType(self.space, 1 * [self.out_repr])

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type

        input = input.tensor
        b, c, h, w = input.shape

        if self.out_mode == "keep":
            # > split the field along the group channel dim into 2 dimensions:
            output = input.view(b, self.num_repr, self.in_repr.size, h, w)
        elif self.out_mode == "reduce":
            output = input.view(b, self.num_repr * self.in_repr.size, 1, h, w)
        else:
            raise ValueError

        # > max along the repr channel
        output, _ = torch.max(output, 1)

        # > check result
        assert self.evaluate_output_shape(input.shape) == output.shape

        # > wrap the result in a GeometricTensor
        return GeometricTensor(output, self.out_type)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) == 4

        b, c, hi, wi = input_shape
        assert c == self.in_type.size

        # > the group channel will be maxed out and pushed to 1 of regular/trivial repr
        return b, self.out_repr.size, hi, wi

    def export(self):
        # use MaxPoolChannels
        raise NotImplementedError
