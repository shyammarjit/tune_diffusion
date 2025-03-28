# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch.nn.functional as F
from torch import nn
import torch

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


class KronALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=(64, 4), network_alpha=None, device=None, dtype=None):
        super().__init__()
        
        self.a1 = rank[0]
        self.a2 = rank[1]
        """ 
        This is krona implementation
            A: (a1 * a2)
            B: (b1 * b2)
            a1 * b1 = d_out = out_features
            a2 * b2 = d_in = in_features
        
        Note: Supported a1 and a2 must be in multiplier of 2
        """
        assert in_features%self.a2==0 and out_features%self.a1==0
        self.b2 = int(in_features/self.a2); self.b1 = int(out_features/self.a1)
        self.down = nn.Linear(self.a1, self.a2, bias=False, device=device, dtype=dtype) # A
        self.up = nn.Linear(self.b2, self.b1, bias=False, device=device, dtype=dtype) # B
        # self.down = nn.Parameter(torch.FloatTensor(self.a2, self.a1).to(dtype).to(device), requires_grad=True)
        # self.up = nn.Parameter(torch.FloatTensor(self.b1, self.b2).to(dtype).to(device), requires_grad=True)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha

        nn.init.normal_(self.down.weight, std=1 / rank[0])
        nn.init.zeros_(self.up.weight)
        # nn.init.normal_(self.down, std=1 / rank[0])
        # nn.init.zeros_(self.up)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        # dtype = self.down.dtype
        
        
        # print(self.a1, self.a2, self.b1, self.b2)
        # exit()
        if len(hidden_states.shape) == 3:
            B1, C, D = hidden_states.size() # get the matrix shape
            hidden_states = hidden_states.view(-1, self.b2, self.a2).contiguous().view(-1, self.a2, self.b2).transpose(1, 2)
            
            up_hidden_states = self.up.weight@(hidden_states.to(dtype)@self.down.weight)
            # up_hidden_states = @(hidden_states.to(dtype)@self.down)
            # up_hidden_states = torch.matmul(self.up, torch.matmul(hidden_states.to(dtype), self.down))
            
            up_hidden_states = up_hidden_states.view(B1, C, self.a1*self.b1)

        else: 
            B1, C = hidden_states.size() # get the matrix shape
            hidden_states = hidden_states.view(B1, self.b2, self.a2)
            hidden_states = hidden_states.view(B1*self.b2, self.a2)
            up_hidden_states = self.up.weight@(self.down(hidden_states.to(dtype)))
            up_hidden_states = up_hidden_states.view(B1, self.b1 * self.a1)

        # exit()
        """new"""
        # hidden_states_reshape = hidden_states.view(self.b2, self.a2).contiguous().view(self.a2, self.b2).t()
        # down_hidden_states = self.down(hidden_states_reshape.to(dtype))
        # up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None, lphm=None):
        super().__init__()
        # print(kamal)
        # exit()
        self.lphm = lphm
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        
        if lphm:
            # self.down_in = nn.Parameter(torch.FloatTensor(in_features, 1).to(dtype).to(device), requires_grad=True)
            self.down_in = nn.Linear(1, in_features, bias=False, device=device, dtype=dtype)
            # self.down_out = nn.Parameter(torch.FloatTensor(1, rank).to(dtype).to(device), requires_grad=True)
            self.down_out = nn.Linear(rank, 1, bias=False, device=device, dtype=dtype)

            # self.up_in = nn.Parameter(torch.FloatTensor(rank, 1).to(dtype).to(device), requires_grad=True)
            self.up_in = nn.Linear(1, rank, bias=False, device=device, dtype=dtype)
            # self.up_out = nn.Parameter(torch.FloatTensor(1, out_features).to(dtype).to(device), requires_grad=True)
            self.up_out = nn.Linear(out_features, 1, bias=False, device=device, dtype=dtype)

            nn.init.normal_(self.down_in.weight, std=1 / rank)
            nn.init.normal_(self.down_out.weight, std=1 / rank)
            nn.init.zeros_(self.up_in.weight)
            nn.init.zeros_(self.up_out.weight)
            # self.down = torch.kron(self.down_in.weight, self.down_out.weight)#.to(dtype).to(device)
            # self.up = torch.kron(self.up_in.weight, self.up_out.weight)#.to(dtype).to(device)
            
        else: 
            self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
            self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
            
            nn.init.normal_(self.down.weight, std=1 / rank)
            nn.init.zeros_(self.up.weight)

        
        

    def forward(self, hidden_states):
        # print('in')
        # print(self.down_in.weight.device)
        # exit()
        orig_dtype = hidden_states.dtype

        if self.lphm:
            dtype = self.down_in.weight.dtype
            device = self.down_in.weight.device
            # print(hidden_states.shape, self.down.shape)
            # exit()
            self.down = torch.kron(self.down_in.weight, self.down_out.weight)#.to(dtype).to(device)
            self.up = torch.kron(self.up_in.weight, self.up_out.weight)#.to(dtype).to(device)
            down_hidden_states = hidden_states.to(dtype)@self.down.to(device)
            up_hidden_states = down_hidden_states@self.up.to(device)
        else:
            dtype = self.down.weight.dtype
            down_hidden_states = self.down(hidden_states.to(dtype))
            up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRAConv2dLayer(nn.Module):
    def __init__(
        self, in_features, out_features, rank=4, kernel_size=(1, 1), stride=(1, 1), padding=0, network_alpha=None
    ):
        super().__init__()

        self.down = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def forward(self, x):
        """It's None."""
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super().forward(x) + self.lora_layer(x)


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, adapter_low_rank=None, **kwargs):
        super().__init__(*args, **kwargs)
        # print('shyam', args, kwargs)
        # print(kamal)
        self.lora_layer = lora_layer
        self.args = args
        self.kwargs = kwargs
    
    def get_config(self):
        return self.args, self.kwargs

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        """ Note: here rank paramters are already set so no need to set them again. """
        self.lora_layer = lora_layer

    def forward(self, x):
        if self.lora_layer is None:
            return super().forward(x)
        else:
            return super().forward(x) + self.lora_layer(x)
