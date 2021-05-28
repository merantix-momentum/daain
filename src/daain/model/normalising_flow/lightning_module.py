from typing import Union

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import pytorch_lightning as pl
import numpy as np
import torch

from daain.model.normalising_flow.coupling_blocks.subnet_constructors import CouplingBlockType, subnet_conv_1x1, \
    subnet_fc, subnet_linear
from daain.utils.evaluation_utils import kullback_leibler_divergence_to_multivariate_standard_normal


def _create_node(previous_node, subnet_constructor=None, node_id=None, coupling_block_type=Fm.GLOWCouplingBlock):
    if subnet_constructor is not None:
        return Ff.Node(
            previous_node,
            coupling_block_type,
            {"subnet_constructor": subnet_constructor, "clamp": 2.0},
            name=f"cpl_{node_id}",
        )
    else:
        return Ff.Node(previous_node, coupling_block_type, {}, name=f"cpl_{node_id}")


class OODDetectionFlow(pl.LightningModule):
    def __init__(
            self,
            input_shape,
            coupling_block_type: Union[CouplingBlockType, str],
            num_coupling_blocks=8,
    ):
        super().__init__()

        if isinstance(coupling_block_type, str):
            coupling_block_type = getattr(CouplingBlockType, coupling_block_type)

        self._str_ = f"{coupling_block_type.name}-flow_steps_{num_coupling_blocks}"

        if coupling_block_type in (CouplingBlockType.GLOW_LINEAR, CouplingBlockType.GIN_LINEAR):
            inn = Ff.SequenceINN(*input_shape)

            for k in range(num_coupling_blocks):
                inn.append(Fm.AllInOneBlock,
                           subnet_constructor=subnet_fc,
                           gin_block=coupling_block_type == CouplingBlockType.GIN_LINEAR)

            self.model = inn
        elif coupling_block_type in (CouplingBlockType.GLOW_1x1_CONV, CouplingBlockType.GLOW_1x1_CONV_GIN):
            inn = Ff.SequenceINN(*input_shape)

            for k in range(num_coupling_blocks):
                inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_conv_1x1,
                           gin_block=coupling_block_type == CouplingBlockType.GLOW_1x1_CONV_GIN)

            self.model = inn
        elif coupling_block_type == CouplingBlockType.CONDITIONAL:
            # TODO move this into subnet_constructors
            nodes = [Ff.InputNode(*input_shape, name='input')]
            ndim_x = np.prod(input_shape)

            # Higher resolution convolutional part
            for k in range(4):
                nodes.append(Ff.Node(nodes[-1],
                                     Fm.GLOWCouplingBlock,
                                     {'subnet_constructor': subnet_linear, 'clamp': 1.2},
                                     name=F'conv_high_res_{k}'))
                nodes.append(Ff.Node(nodes[-1],
                                     Fm.PermuteRandom,
                                     {'seed': k},
                                     name=F'permute_high_res_{k}'))

            nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

            # Lower resolution convolutional part
            for k in range(12):
                if k % 2 == 0:
                    subnet = subnet_conv_1x1
                else:
                    subnet = subnet_linear

                nodes.append(Ff.Node(nodes[-1],
                                     Fm.GLOWCouplingBlock,
                                     {'subnet_constructor': subnet, 'clamp': 1.2},
                                     name=F'conv_low_res_{k}'))
                nodes.append(Ff.Node(nodes[-1],
                                     Fm.PermuteRandom,
                                     {'seed': k},
                                     name=F'permute_low_res_{k}'))

            # Make the outputs into a vector, then split off 1/4 of the outputs for the
            # fully connected part
            nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
            split_node = Ff.Node(nodes[-1],
                                 Fm.Split,
                                 {'section_sizes': (ndim_x // 4, 3 * ndim_x // 4), 'dim': 0},
                                 name='split')
            nodes.append(split_node)

            # Fully connected part
            for k in range(12):
                nodes.append(Ff.Node(nodes[-1],
                                     Fm.GLOWCouplingBlock,
                                     {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                                     name=F'fully_connected_{k}'))
                nodes.append(Ff.Node(nodes[-1],
                                     Fm.PermuteRandom,
                                     {'seed': k},
                                     name=F'permute_{k}'))

            # Concatenate the fully connected part and the skip connection to get a single output
            nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
                                 Fm.Concat1d, {'dim': 0}, name='concat'))
            nodes.append(Ff.OutputNode(nodes[-1], name='output'))

            self.model = Ff.GraphINN(nodes)
        else:
            raise NotImplementedError(f"no such coupling block type: {coupling_block_type}")

        # used for the attention layers with fixed positional encoding (the keys are passed along)
        self.split_output = False

    def forward(self, x, jac=False):
        """

        Args:
            x:

        Returns:

        """

        if self.split_output:
            out, jacobian = self.model(x, jac=True)
            # select only the first channel, the rest are the positions
            out = torch.split(out, [1, out.shape[1] - 1], dim=1)[0].view(out.shape[0], -1)
            if jac:
                return out, jacobian
            else:
                return out
        else:
            out, jacobian = self.model(x, jac=True)
            if jac:
                return out, jacobian
            else:
                return out

    def training_step(self, batch, batch_idx):
        # print(f"batch shape {batch.shape}")
        z, jacobian = self.forward(batch, jac=True)
        loss = self._loss(z, jacobian)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        z, jacobian = self.forward(batch, jac=True)
        loss = self._loss(z, jacobian)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log(
            "kl_loss", kullback_leibler_divergence_to_multivariate_standard_normal(z), on_epoch=True, on_step=False
        )

        return loss

    def configure_optimizers(self, learning_rate=2e-4, betas=None, eps=1e-04, weight_decay=1e-5):
        if betas is None:
            betas = (0.8, 0.8)
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

    def _loss(self, z, jac):
        if len(z.shape) == 4:
            return torch.mean(0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) - jac) / z.shape[1]
        else:
            return torch.mean(0.5 * torch.sum(z ** 2, dim=(1)) - jac) / z.shape[1]

    def eval(self, cuda=False):
        t = super(OODDetectionFlow, self).eval()

        if cuda:
            return t.cuda()
        else:
            return t

    def __str__(self):
        return self._str_

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        new_state_dict = self.model.state_dict()
        saved_state_dict = ckpt["state_dict"]

        for k in saved_state_dict.keys():
            new_state_dict[k[len('model.'):]] = saved_state_dict[k]

        self.model.load_state_dict(new_state_dict)

        return self
