import torch
import torch.nn as nn
import torch.nn.functional as F

from diffplan.modules.helpers import NormalDebugReturn, StandardReturn


class Planner(nn.Module):
    """
    Implementation of the Fully Convolutional version of GPPN.
    (based on VIN and Conv-LSTM)
    """

    def __init__(self, num_orient, num_actions, args):
        super(Planner, self).__init__()

        self.num_orient = num_orient
        self.num_actions = num_actions

        self.l_q = args.l_q
        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f
        self.padding_mode = "circular" if "wrap" in args.mechanism else "zeros"

        self.h = nn.Conv2d(
            in_channels=(num_orient + 1),  # maze map + goal location
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.r = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=num_orient,  # reward per orientation
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.q = nn.Conv2d(
            in_channels=num_orient,
            out_channels=self.l_q * num_orient,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=False,
        )

        self.policy = nn.Conv2d(
            in_channels=self.l_q * num_orient,
            out_channels=num_actions * num_orient,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.LSTM_cnn = ConvLSTM(args)

        self.sm = nn.Softmax2d()

    def forward(self, map_design, goal_map, debug=False):
        maze_size = map_design.size()[-1]
        X = torch.cat([map_design, goal_map], 1)

        h = self.h(X)  # [150, 15, 15]
        r = self.r(h)  # [1, 15, 15]
        q = self.q(r)  # [600, 15, 15]

        q_rnn = q
        q_cell = q

        for _ in range(0, self.k - 1):
            q_rnn, q_cell = self.LSTM_cnn(r, q_rnn, q_cell)

        q_rnn, q_cell = self.LSTM_cnn(r, q_rnn, q_cell)
        # q_rnn, q_cell = self.LSTM_cnn(r, q_rnn, q_rnn)  # mistake, which one is good?

        logits = self.policy(q_rnn)

        # Normalize over actions
        logits = logits.view(-1, self.num_actions, maze_size, maze_size)
        probs = self.sm(logits)

        # Reshape to output dimensions
        logits = logits.view(
            -1, self.num_orient, self.num_actions, maze_size, maze_size
        )
        probs = probs.view(-1, self.num_orient, self.num_actions, maze_size, maze_size)
        logits = torch.transpose(logits, 1, 2).contiguous()
        probs = torch.transpose(probs, 1, 2).contiguous()

        if not debug:
            return StandardReturn(logits, probs)
        else:
            return NormalDebugReturn(logits, probs, q_rnn, None, r)


class ConvLSTM(nn.Module):
    """
    Using pytorch, implement a customized LSTM
    """

    def __init__(self, args):
        super(ConvLSTM, self).__init__()

        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f
        self.l_q = args.l_q
        self.padding_mode = "circular" if "wrap" in args.mechanism else "zeros"

        # input content
        self.cnn_q_hid_tanh = nn.Conv2d(
            in_channels=self.l_q,
            out_channels=self.l_q,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        # fixme see if bias matters? hidden(q) bias is false?
        self.cnn_r_hid_tanh = nn.Conv2d(
            in_channels=1,
            out_channels=self.l_q,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.tanh_input = nn.Tanh()

        # input gate
        self.cnn_q_hid_sig = nn.Conv2d(
            in_channels=self.l_q,
            out_channels=self.l_q,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.cnn_r_sig = nn.Conv2d(
            in_channels=1,
            out_channels=self.l_q,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.sigmoid_input = nn.Sigmoid()

        # forget gate
        self.cnn_q_hid_forget = nn.Conv2d(
            in_channels=self.l_q,
            out_channels=self.l_q,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.cnn_r_forget = nn.Conv2d(
            in_channels=1,
            out_channels=self.l_q,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.sigmoid_forget = nn.Sigmoid()

        # output gate
        self.cnn_q_hid_out = nn.Conv2d(
            in_channels=self.l_q,
            out_channels=self.l_q,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.cnn_r_out = nn.Conv2d(
            in_channels=1,
            out_channels=self.l_q,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            padding_mode=self.padding_mode,
            bias=True,
        )

        self.sigmoid_output = nn.Sigmoid()

        self.tanh_output = nn.Tanh()

    def cat_pad(self, r, q, padding=0):
        out = torch.cat([r, q], 1)
        padding_mode = "circular" if self.padding_mode == "circular" else "constant"
        if padding != 0:
            out = F.pad(out, (padding,) * 4, mode=padding_mode)
        return out

    def input_content(self, r, q):
        q_rnn = F.conv2d(
            self.cat_pad(r, q, padding=int((self.f - 1.0) / 2)),
            torch.cat([self.cnn_r_hid_tanh.weight, self.cnn_q_hid_tanh.weight], 1),
            stride=1,
        )

        q_rnn = self.tanh_input(q_rnn)
        return q_rnn

    def input_gate(self, r, q):
        q_rnn = F.conv2d(
            self.cat_pad(r, q, padding=int((self.f - 1.0) / 2)),
            torch.cat([self.cnn_r_sig.weight, self.cnn_q_hid_sig.weight], 1),
            stride=1,
        )

        q_rnn = self.sigmoid_input(q_rnn)

        return q_rnn

    def forget(self, r, q):
        q_rnn = F.conv2d(
            self.cat_pad(r, q, padding=int((self.f - 1.0) / 2)),
            torch.cat([self.cnn_r_forget.weight, self.cnn_q_hid_forget.weight], 1),
            stride=1,
        )

        q_rnn = self.sigmoid_forget(q_rnn)

        return q_rnn

    def output_gate(self, r, q):
        q_rnn = F.conv2d(
            self.cat_pad(r, q, padding=int((self.f - 1.0) / 2)),
            torch.cat([self.cnn_r_out.weight, self.cnn_q_hid_out.weight], 1),
            stride=1,
        )

        q_rnn = self.sigmoid_output(q_rnn)

        return q_rnn

    def cell_memory_update(self, q_cell, forget_gate, input_cont, input_gate):
        cell_new = forget_gate * q_cell + input_cont * input_gate

        return cell_new

    def forward(self, r, q, q_cell):
        """
        input: r: [1, maze_size, maze_size]
                q: [hid_dim, maze_size, maze_size]
                q_cell: [hid_dim, maze_size, maze_size]
        output: same as q: [hid_dim, maze_size, maze_size]
        """

        # equation 1: forget gate
        forget_gate = self.forget(r, q)

        # equation 2: input gate
        input_cont = self.input_content(r, q)
        input_gate = self.input_gate(r, q)

        # equation 3: cell update
        q_cell_new = self.cell_memory_update(
            q_cell, forget_gate, input_cont, input_gate
        )

        # equation 4: output gate
        output_g = self.output_gate(r, q)

        # equation 5: hidden state q
        q_new = output_g * self.tanh_output(q_cell_new)

        return q_new, q_cell_new
