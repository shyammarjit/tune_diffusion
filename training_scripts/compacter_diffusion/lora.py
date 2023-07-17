import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


import torch
import torch.nn as nn
from typing import Union, Optional
import torch.nn.functional as F

from .hypercomplex .inits import glorot_uniform, glorot_normal  #phm_init
from .hypercomplex.kronecker import kronecker_product, kronecker_product_einsum_batched
from .adapter_utils import Activations
from .hypercomplex.layers import PHMLinear

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


class LoraInjectedLinear(nn.Module):
    # def __init__(
    #     self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    # ):

    #     super().__init__()
    #     if r > min(in_features, out_features):
    #         raise ValueError(
    #             f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
    #         )
    #     """
    #     in_features: int 
    #     out_features: int
    #     r: int
    #     """

    #     self.r = r
    #     self.linear = nn.Linear(in_features, out_features, bias)
    #     self.dh = in_features
    #     self.a1 = int(self.dh/ r ) # 256
    #     self.b1 = r # 4
    #     self.a2 = r # 4
    #     self.b2 = int(out_features / self.a2) 
    #     self.lora_down = nn.Linear(self.a1, self.a2, bias=False) # A
    #     self.dropout = nn.Dropout(dropout_p)
    #     self.lora_up = nn.Linear(self.b1, self.b2, bias=False) # B
    #     self.scale = scale
    #     self.selector = nn.Identity()

    #     nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
    #     nn.init.zeros_(self.lora_up.weight)

    # def forward(self, input):
    #     """
    #     Input has variable shapes (1, #channels, in_features)
    #     Note: It can also be of shape (1, in_features)
    #     We only require to change in_features.
    #     W * X + Y (B N(x)A`)
    #     """
    #     # get the original weight matrix
    #     W0X = self.linear(input)
    #     if len(input.shape) == 3:
    #         B1, C, D = input.size() # get the matrix shape
    #         input = input.view(-1, self.a2, self.a1)
    #         B2, _, _ = input.size() # get the matrix shape
    #         W_ = self.lora_up(self.selector(self.lora_down(input)))
    #         W_= W_.view(B1, C, self.b2*self.b1)
    #     else: 
    #         B1, C = input.size() # get the matrix shape
    #         input = input.view(B1, self.a2, self.a1)
    #         input = input.view(B1*self.a2, self.a1)
    #         W_ = self.lora_up(self.selector(self.lora_down(input)))
    #         W_= W_.view(B1, self.b1* self.b2)

    #     return (W0X + self.dropout(W_) * self.scale)

    # def realize_as_lora(self):
    #     return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    # def set_selector_from_diag(self, diag: torch.Tensor):
    #     # diag is a 1D tensor of size (r,)
    #     assert diag.shape == (self.r,)
    #     self.selector = nn.Linear(self.r, self.r, bias=False)
    #     self.selector.weight.data = torch.diag(diag)
    #     self.selector.weight.data = self.selector.weight.data.to(
    #         self.lora_up.weight.device
    #     ).to(self.lora_up.weight.dtype)

    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""
    # def __init__(self,
    #              in_features: int,
    #              out_features: int,
    #              r: int,
    #              bias: bool = True,
    #              phm_rule: Union[None, torch.Tensor] = None,
    #              w_init: str = "phm",
    #              c_init: str = "random",
    #              learn_phm: bool = True,
    #              shared_phm_rule=False,
    #              factorized_phm=False,
    #              shared_W_phm=False,
    #              factorized_phm_rule=False,
    #              phm_rank = 1,
    #              phm_init_range=0.0001,
    #              kronecker_prod=False) -> None:
    #     super(LoraInjectedLinear, self).__init__()
    #     assert w_init in ["phm", "glorot-normal", "glorot-uniform", "normal"]
    #     assert c_init in ["normal", "uniform"]
    #     assert in_features % r == 0, f"Argument `in_features`={in_features} is not divisble be `r`{r}"
    #     assert out_features % r == 0, f"Argument `out_features`={out_features} is not divisble be `r`{r}"
    #     self.in_features = in_features
    #     self.out_features = out_features
    #     self.learn_phm = learn_phm
    #     self.r = r
    #     self._in_feats_per_axis = in_features // r
    #     self._out_feats_per_axis = out_features // r
    #     self.phm_rank = phm_rank
    #     self.phm_init_range = phm_init_range
    #     self.kronecker_prod=kronecker_prod
    #     self.shared_phm_rule = shared_phm_rule
    #     self.factorized_phm_rule = factorized_phm_rule 
    #     if not self.shared_phm_rule:
    #         if self.factorized_phm_rule:
    #             self.phm_rule_left = nn.Parameter(torch.FloatTensor(r, r, 1),
    #                    requires_grad=learn_phm)
    #             self.phm_rule_right = nn.Parameter(torch.FloatTensor(r, 1, r),
    #                    requires_grad=learn_phm)
    #         else:
    #             self.phm_rule = nn.Parameter(torch.FloatTensor(r, r, r), 
    #                    requires_grad=learn_phm)
    #     self.bias_flag = bias
    #     self.w_init = w_init
    #     self.c_init = c_init
    #     self.shared_W_phm = shared_W_phm 
    #     self.factorized_phm = factorized_phm
    #     if not self.shared_W_phm:
    #         if self.factorized_phm:
    #             self.W_left = nn.Parameter(torch.Tensor(size=(r, self._in_feats_per_axis, self.phm_rank)),
    #                           requires_grad=True)
    #             self.W_right = nn.Parameter(torch.Tensor(size=(r, self.phm_rank, self._out_feats_per_axis)),
    #                           requires_grad=True)
    #         else:
    #             self.W = nn.Parameter(torch.Tensor(size=(r, self._in_feats_per_axis, self._out_feats_per_axis)),
    #                           requires_grad=True)
    #     if self.bias_flag:
    #         self.b = nn.Parameter(torch.Tensor(out_features))
    #     else:
    #         self.register_parameter("b", None)
    #     self.reset_parameters()

    # def init_W(self):
    #     if self.w_init == "glorot-normal":
    #         if self.factorized_phm:
    #             for i in range(self.r):
    #                 self.W_left.data[i] = glorot_normal(self.W_left.data[i])
    #                 self.W_right.data[i] = glorot_normal(self.W_right.data[i])
    #         else:
    #             for i in range(self.r):
    #                 self.W.data[i] = glorot_normal(self.W.data[i])
    #     elif self.w_init == "glorot-uniform":
    #         if self.factorized_phm:
    #             for i in range(self.r):
    #                 self.W_left.data[i] = glorot_uniform(self.W_left.data[i])
    #                 self.W_right.data[i] = glorot_uniform(self.W_right.data[i])
    #         else:
    #             for i in range(self.r):
    #                 self.W.data[i] = glorot_uniform(self.W.data[i])
    #     elif self.w_init == "normal":
    #         if self.factorized_phm:
    #             for i in range(self.r):
    #                 self.W_left.data[i].normal_(mean=0, std=self.phm_init_range)
    #                 self.W_right.data[i].normal_(mean=0, std=self.phm_init_range)
    #         else:
    #             for i in range(self.r):
    #                 self.W.data[i].normal_(mean=0, std=self.phm_init_range)
    #     else:
    #         raise ValueError

    # def reset_parameters(self):
    #     if not self.shared_W_phm:
    #        self.init_W()

    #     if self.bias_flag:
    #         self.b.data = torch.zeros_like(self.b.data)

    #     if not self.shared_phm_rule:
    #         if self.factorized_phm_rule:
    #             if self.c_init == "uniform":
    #                self.phm_rule_left.data.uniform_(-0.01, 0.01)
    #                self.phm_rule_right.data.uniform_(-0.01, 0.01)
    #             elif self.c_init == "normal":
    #                self.phm_rule_left.data.normal_(std=self.phm_init_range)
    #                self.phm_rule_right.data.normal_(std=self.phm_init_range)
    #             else:
    #                raise NotImplementedError
    #         else:
    #             if self.c_init == "uniform":
    #                self.phm_rule.data.uniform_(-0.01, 0.01)
    #             elif self.c_init == "normal":
    #                self.phm_rule.data.normal_(mean=0, std=self.phm_init_range)
    #             else:
    #                raise NotImplementedError

    # def set_phm_rule(self, phm_rule=None, phm_rule_left=None, phm_rule_right=None):
    #     """If factorized_phm_rules is set, phm_rule is a tuple, showing the left and right
    #     phm rules, and if this is not set, this is showing  the phm_rule."""
    #     if self.factorized_phm_rule:
    #         self.phm_rule_left = phm_rule_left
    #         self.phm_rule_right = phm_rule_right 
    #     else:
    #         self.phm_rule = phm_rule 
 
    # def set_W(self, W=None, W_left=None, W_right=None):
    #     if self.factorized_phm:
    #         self.W_left = W_left
    #         self.W_right = W_right
    #     else:
    #         self.W = W

    # def forward(self, x: torch.Tensor, phm_rule: Union[None, nn.ParameterList] = None) -> torch.Tensor:
    #     if self.factorized_phm:
    #        W = torch.bmm(self.W_left, self.W_right)
    #     if self.factorized_phm_rule:
    #        phm_rule = torch.bmm(self.phm_rule_left, self.phm_rule_right)
    #     return matvec_product(
    #            W=W if self.factorized_phm else self.W,
    #            x=x,
    #            bias=self.b,
    #            phm_rule=phm_rule if self.factorized_phm_rule else self.phm_rule, 
    #            kronecker_prod=self.kronecker_prod)



    # def __init__(self, input_dim: int, output_dim: int,  bias: bool = True,r: int = 1,
    #             dropout_p=0.1, scale=1.0):


    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, in_features, out_features, bias=False, r=2, dropout_p=0.1, scale=1.0):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.linear = nn.Linear(in_features, out_features, bias)
        self.r = r
        self.bias = bias
        self.down_sample_size = self.input_dim // self.r
        self.lora_down = nn.Linear(in_features,self.down_sample_size , bias=False)
        self.lora_up = nn.Linear(self.down_sample_size , out_features, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)
        self.lora_down = PHMLinear(in_features=self.input_dim,
                                      out_features=self.down_sample_size,
                                      bias=self.bias,
                                      phm_dim=self.r,
                                      dropout_p=0.1, scale=1.0,
                                      )
        
        self.lora_up = PHMLinear(in_features=self.down_sample_size,
                                      out_features=self.output_dim,
                                      bias=self.bias,
                                      phm_dim=self.r,
                                      dropout_p=0.1, scale=1.0,
                                      )
        
    def forward(self, x):
        z = self.lora_down(x)
        z = self.activation(z)
        return  self.linear(input) + self.lora_up(z)



    
class LoraInjectedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}"
            )
        self.r = r
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        
        self.dh = in_channels
        self.a1 = int(self.dh/ r ) # 256
        self.b1 = r # 256
        # a1 * b1 = dh = in_features
        
        self.a2 = r # 4
        self.b2 = int(out_channels / self.a2)
        # a2 * b2 = dh = out_features
        

        self.lora_down = nn.Conv2d(
            in_channels = self.a1,
            out_channels = self.a2,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv2d(
            in_channels = self.b1,
            out_channels = self.b2,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False,
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        """
        input has the shape of B, C, H, W
        """
        # original weight matrix multiplication 
        W0X = self.conv(input)
        
        B, C, H, W = input.size() # shape([1, 320, 64, 64])
        input = input.view(B, self.a2, self.a1, H, W)
        W_X = self.dropout(self.lora_up(self.selector(self.lora_down(input.view(-1, self.a1, H, W)))))
        W_X = W_X.view(B, self.b1 * self.b2, H, W)
        # print("Huehue")
        try:
            return (W0X + W_X * self.scale)
        except:
            print(B, C, H, W)
            print(W0X.shape)
            print(W_X.shape)
            exit()

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Conv2d(
            in_channels=self.r,
            out_channels=self.r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag)

        # same device + dtype as lora_up
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}

UNET_EXTENDED_TARGET_REPLACE = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}

TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

TEXT_ENCODER_EXTENDED_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

EMBED_FLAG = "<embed>"


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    Find all modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """
    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module


def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedLinear,
        LoraInjectedConv2d,
    ],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def _find_modules_old(
    model,
    ancestor_class: Set[str] = DEFAULT_TARGET_REPLACE,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [LoraInjectedLinear],
):
    ret = []
    for _module in model.modules():
        if _module.__class__.__name__ in ancestor_class:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__ in search_class:
                    ret.append((_module, name, _child_module))
    print(ret)
    return ret


_find_modules = _find_modules_v2


def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)
        _tmp = LoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)

        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        names.append(name)

    return require_grad_params, names


def inject_trainable_lora_extended(
    model: nn.Module,
    target_replace_module: Set[str] = UNET_EXTENDED_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
    **kwargs,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, nn.Conv2d]
    ):
        if _child_module.__class__ == nn.Linear:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedLinear(
                _child_module.in_features,
                _child_module.out_features,
                _child_module.bias is not None,
                r=r,
            )
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias
        elif _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedConv2d(
                _child_module.in_channels,
                _child_module.out_channels,
                _child_module.kernel_size,
                _child_module.stride,
                _child_module.padding,
                _child_module.dilation,
                _child_module.groups,
                _child_module.bias is not None,
                r=r,
            )

            _tmp.conv.weight = weight
            if bias is not None:
                _tmp.conv.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        if bias is not None:
            _tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)

        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        names.append(name)

    return require_grad_params, names


def extract_lora_ups_down(model, target_replace_module=DEFAULT_TARGET_REPLACE):

    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        loras.append((_child_module.lora_up, _child_module.lora_down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras


def extract_lora_as_tensor(
    model, target_replace_module=DEFAULT_TARGET_REPLACE, as_fp16=True
):

    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        up, down = _child_module.realize_as_lora()
        if as_fp16:
            up = up.to(torch.float16)
            down = down.to(torch.float16)

        loras.append((up, down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras


def save_lora_weight(
    model,
    path="./lora.pt",
    target_replace_module=DEFAULT_TARGET_REPLACE,
):
    weights = []
    for _up, _down in extract_lora_ups_down(
        model, target_replace_module=target_replace_module
    ):
        weights.append(_up.weight.to("cpu").to(torch.float16))
        weights.append(_down.weight.to("cpu").to(torch.float16))

    torch.save(weights, path)


def save_lora_as_json(model, path="./lora.json"):
    weights = []
    for _up, _down in extract_lora_ups_down(model):
        weights.append(_up.weight.detach().cpu().numpy().tolist())
        weights.append(_down.weight.detach().cpu().numpy().tolist())

    import json

    with open(path, "w") as f:
        json.dump(weights, f)


def save_safeloras_with_embeds(
    modelmap: Dict[str, Tuple[nn.Module, Set[str]]] = {},
    embeds: Dict[str, torch.Tensor] = {},
    outpath="./lora.safetensors",
):
    """
    Saves the Lora from multiple modules in a single safetensor file.

    modelmap is a dictionary of {
        "module name": (module, target_replace_module)
    }
    """
    weights = {}
    metadata = {}

    for name, (model, target_replace_module) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        for i, (_up, _down) in enumerate(
            extract_lora_as_tensor(model, target_replace_module)
        ):
            rank = _down.shape[0]

            metadata[f"{name}:{i}:rank"] = str(rank)
            weights[f"{name}:{i}:up"] = _up
            weights[f"{name}:{i}:down"] = _down

    for token, tensor in embeds.items():
        metadata[token] = EMBED_FLAG
        weights[token] = tensor

    print(f"Saving weights to {outpath}")
    safe_save(weights, outpath, metadata)


def save_safeloras(
    modelmap: Dict[str, Tuple[nn.Module, Set[str]]] = {},
    outpath="./lora.safetensors",
):
    return save_safeloras_with_embeds(modelmap=modelmap, outpath=outpath)


def convert_loras_to_safeloras_with_embeds(
    modelmap: Dict[str, Tuple[str, Set[str], int]] = {},
    embeds: Dict[str, torch.Tensor] = {},
    outpath="./lora.safetensors",
):
    """
    Converts the Lora from multiple pytorch .pt files into a single safetensor file.

    modelmap is a dictionary of {
        "module name": (pytorch_model_path, target_replace_module, rank)
    }
    """

    weights = {}
    metadata = {}

    for name, (path, target_replace_module, r) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        lora = torch.load(path)
        for i, weight in enumerate(lora):
            is_up = i % 2 == 0
            i = i // 2

            if is_up:
                metadata[f"{name}:{i}:rank"] = str(r)
                weights[f"{name}:{i}:up"] = weight
            else:
                weights[f"{name}:{i}:down"] = weight

    for token, tensor in embeds.items():
        metadata[token] = EMBED_FLAG
        weights[token] = tensor

    print(f"Saving weights to {outpath}")
    safe_save(weights, outpath, metadata)


def convert_loras_to_safeloras(
    modelmap: Dict[str, Tuple[str, Set[str], int]] = {},
    outpath="./lora.safetensors",
):
    convert_loras_to_safeloras_with_embeds(modelmap=modelmap, outpath=outpath)


def parse_safeloras(
    safeloras,
) -> Dict[str, Tuple[List[nn.parameter.Parameter], List[int], List[str]]]:
    """
    Converts a loaded safetensor file that contains a set of module Loras
    into Parameters and other information

    Output is a dictionary of {
        "module name": (
            [list of weights],
            [list of ranks],
            target_replacement_modules
        )
    }
    """
    loras = {}
    metadata = safeloras.metadata()

    get_name = lambda k: k.split(":")[0]

    keys = list(safeloras.keys())
    keys.sort(key=get_name)

    for name, module_keys in groupby(keys, get_name):
        info = metadata.get(name)

        if not info:
            raise ValueError(
                f"Tensor {name} has no metadata - is this a Lora safetensor?"
            )

        # Skip Textual Inversion embeds
        if info == EMBED_FLAG:
            continue

        # Handle Loras
        # Extract the targets
        target = json.loads(info)

        # Build the result lists - Python needs us to preallocate lists to insert into them
        module_keys = list(module_keys)
        ranks = [4] * (len(module_keys) // 2)
        weights = [None] * len(module_keys)

        for key in module_keys:
            # Split the model name and index out of the key
            _, idx, direction = key.split(":")
            idx = int(idx)

            # Add the rank
            ranks[idx] = int(metadata[f"{name}:{idx}:rank"])

            # Insert the weight into the list
            idx = idx * 2 + (1 if direction == "down" else 0)
            weights[idx] = nn.parameter.Parameter(safeloras.get_tensor(key))

        loras[name] = (weights, ranks, target)

    return loras


def parse_safeloras_embeds(
    safeloras,
) -> Dict[str, torch.Tensor]:
    """
    Converts a loaded safetensor file that contains Textual Inversion embeds into
    a dictionary of embed_token: Tensor
    """
    embeds = {}
    metadata = safeloras.metadata()

    for key in safeloras.keys():
        # Only handle Textual Inversion embeds
        meta = metadata.get(key)
        if not meta or meta != EMBED_FLAG:
            continue

        embeds[key] = safeloras.get_tensor(key)

    return embeds


def load_safeloras(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras(safeloras)


def load_safeloras_embeds(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras_embeds(safeloras)


def load_safeloras_both(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras(safeloras), parse_safeloras_embeds(safeloras)


def collapse_lora(model, alpha=1.0):

    for _module, name, _child_module in _find_modules(
        model,
        UNET_EXTENDED_TARGET_REPLACE | TEXT_ENCODER_EXTENDED_TARGET_REPLACE,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):

        if isinstance(_child_module, LoraInjectedLinear):
            print("Collapsing Lin Lora in", name)

            _child_module.linear.weight = nn.Parameter(
                _child_module.linear.weight.data
                + alpha
                * (
                    _child_module.lora_up.weight.data
                    @ _child_module.lora_down.weight.data
                )
                .type(_child_module.linear.weight.dtype)
                .to(_child_module.linear.weight.device)
            )

        else:
            print("Collapsing Conv Lora in", name)
            _child_module.conv.weight = nn.Parameter(
                _child_module.conv.weight.data
                + alpha
                * (
                    _child_module.lora_up.weight.data.flatten(start_dim=1)
                    @ _child_module.lora_down.weight.data.flatten(start_dim=1)
                )
                .reshape(_child_module.conv.weight.data.shape)
                .type(_child_module.conv.weight.dtype)
                .to(_child_module.conv.weight.device)
            )


def monkeypatch_or_replace_lora(
    model,
    loras,
    target_replace_module=DEFAULT_TARGET_REPLACE,
    r: Union[int, List[int]] = 4,
):
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, LoraInjectedLinear]
    ):
        _source = (
            _child_module.linear
            if isinstance(_child_module, LoraInjectedLinear)
            else _child_module
        )

        weight = _source.weight
        bias = _source.bias
        _tmp = LoraInjectedLinear(
            _source.in_features,
            _source.out_features,
            _source.bias is not None,
            r=r.pop(0) if isinstance(r, list) else r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )

        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_lora_extended(
    model,
    loras,
    target_replace_module=DEFAULT_TARGET_REPLACE,
    r: Union[int, List[int]] = 4,
):
    
    for _module, name, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[nn.Linear, LoraInjectedLinear, nn.Conv2d, LoraInjectedConv2d],
    ):

        if (_child_module.__class__ == nn.Linear) or (
            _child_module.__class__ == LoraInjectedLinear
        ):
            if len(loras[0].shape) != 2:
                continue

            _source = (
                _child_module.linear
                if isinstance(_child_module, LoraInjectedLinear)
                else _child_module
            )

            weight = _source.weight
            bias = _source.bias
            _tmp = LoraInjectedLinear(
                _source.in_features,
                _source.out_features,
                _source.bias is not None,
                r=r.pop(0) if isinstance(r, list) else r,
            )
            _tmp.linear.weight = weight

            if bias is not None:
                _tmp.linear.bias = bias

        elif (_child_module.__class__ == nn.Conv2d) or (
            _child_module.__class__ == LoraInjectedConv2d
        ):
            if len(loras[0].shape) != 4:
                continue
            _source = (
                _child_module.conv
                if isinstance(_child_module, LoraInjectedConv2d)
                else _child_module
            )

            weight = _source.weight
            bias = _source.bias
            _tmp = LoraInjectedConv2d(
                _source.in_channels,
                _source.out_channels,
                _source.kernel_size,
                _source.stride,
                _source.padding,
                _source.dilation,
                _source.groups,
                _source.bias is not None,
                r=r.pop(0) if isinstance(r, list) else r,
            )

            _tmp.conv.weight = weight

            if bias is not None:
                _tmp.conv.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )

        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_safeloras(models, safeloras):
    loras = parse_safeloras(safeloras)

    for name, (lora, ranks, target) in loras.items():
        model = getattr(models, name, None)

        if not model:
            print(f"No model provided for {name}, contained in Lora")
            continue

        monkeypatch_or_replace_lora_extended(model, lora, target, ranks)


def monkeypatch_remove_lora(model):
    for _module, name, _child_module in _find_modules(
        model, search_class=[LoraInjectedLinear, LoraInjectedConv2d]
    ):
        if isinstance(_child_module, LoraInjectedLinear):
            _source = _child_module.linear
            weight, bias = _source.weight, _source.bias

            _tmp = nn.Linear(
                _source.in_features, _source.out_features, bias is not None
            )

            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias

        else:
            _source = _child_module.conv
            weight, bias = _source.weight, _source.bias

            _tmp = nn.Conv2d(
                in_channels=_source.in_channels,
                out_channels=_source.out_channels,
                kernel_size=_source.kernel_size,
                stride=_source.stride,
                padding=_source.padding,
                dilation=_source.dilation,
                groups=_source.groups,
                bias=bias is not None,
            )

            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias

        _module._modules[name] = _tmp


def monkeypatch_add_lora(
    model,
    loras,
    target_replace_module=DEFAULT_TARGET_REPLACE,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedLinear]
    ):
        weight = _child_module.linear.weight

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_up.weight.to(weight.device) * beta
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_down.weight.to(weight.device) * beta
        )

        _module._modules[name].to(weight.device)


def tune_lora_scale(model, alpha: float = 1.0):
    for _module in model.modules():
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            _module.scale = alpha


def set_lora_diag(model, diag: torch.Tensor):
    for _module in model.modules():
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            _module.set_selector_from_diag(diag)


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def _ti_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["ti", "pt"])


def apply_learned_embed_in_clip(
    learned_embeds,
    text_encoder,
    tokenizer,
    token: Optional[Union[str, List[str]]] = None,
    idempotent=False,
):
    if isinstance(token, str):
        trained_tokens = [token]
    elif isinstance(token, list):
        assert len(learned_embeds.keys()) == len(
            token
        ), "The number of tokens and the number of embeds should be the same"
        trained_tokens = token
    else:
        trained_tokens = list(learned_embeds.keys())

    for token in trained_tokens:
        print(token)
        embeds = learned_embeds[token]

        # cast to dtype of text_encoder
        dtype = text_encoder.get_input_embeddings().weight.dtype
        num_added_tokens = tokenizer.add_tokens(token)

        i = 1
        if not idempotent:
            while num_added_tokens == 0:
                print(f"The tokenizer already contains the token {token}.")
                token = f"{token[:-1]}-{i}>"
                print(f"Attempting to add the token {token}.")
                num_added_tokens = tokenizer.add_tokens(token)
                i += 1
        elif num_added_tokens == 0 and idempotent:
            print(f"The tokenizer already contains the token {token}.")
            print(f"Replacing {token} embedding.")

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token


def load_learned_embed_in_clip(
    learned_embeds_path,
    text_encoder,
    tokenizer,
    token: Optional[Union[str, List[str]]] = None,
    idempotent=False,
):
    learned_embeds = torch.load(learned_embeds_path)
    apply_learned_embed_in_clip(
        learned_embeds, text_encoder, tokenizer, token, idempotent
    )

def patch_pipe(
    pipe,
    maybe_unet_path,
    token: Optional[str] = None,
    r: int = 4,
    patch_unet=True,
    patch_text=True,
    patch_ti=True,
    idempotent_token=True,
    unet_target_replace_module=DEFAULT_TARGET_REPLACE,
    text_target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
):
    if maybe_unet_path.endswith(".pt"):
        # torch format

        if maybe_unet_path.endswith(".ti.pt"):
            unet_path = maybe_unet_path[:-6] + ".pt"
        elif maybe_unet_path.endswith(".text_encoder.pt"):
            unet_path = maybe_unet_path[:-16] + ".pt"
        else:
            unet_path = maybe_unet_path

        ti_path = _ti_lora_path(unet_path)
        text_path = _text_lora_path(unet_path)

        if patch_unet:
            print("LoRA : Patching Unet")
            monkeypatch_or_replace_lora(
                pipe.unet,
                torch.load(unet_path),
                r=r,
                target_replace_module=unet_target_replace_module,
            )

        if patch_text:
            print("LoRA : Patching text encoder")
            monkeypatch_or_replace_lora(
                pipe.text_encoder,
                torch.load(text_path),
                target_replace_module=text_target_replace_module,
                r=r,
            )
        if patch_ti:
            print("LoRA : Patching token input")
            token = load_learned_embed_in_clip(
                ti_path,
                pipe.text_encoder,
                pipe.tokenizer,
                token=token,
                idempotent=idempotent_token,
            )

    elif maybe_unet_path.endswith(".safetensors"):
        safeloras = safe_open(maybe_unet_path, framework="pt", device="cpu")
        monkeypatch_or_replace_safeloras(pipe, safeloras)
        tok_dict = parse_safeloras_embeds(safeloras)
        if patch_ti:
            apply_learned_embed_in_clip(
                tok_dict,
                pipe.text_encoder,
                pipe.tokenizer,
                token=token,
                idempotent=idempotent_token,
            )
        return tok_dict


@torch.no_grad()
def inspect_lora(model):
    moved = {}

    for name, _module in model.named_modules():
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            ups = _module.lora_up.weight.data.clone()
            downs = _module.lora_down.weight.data.clone()

            wght: torch.Tensor = ups.flatten(1) @ downs.flatten(1)

            dist = wght.flatten().abs().mean().item()
            if name in moved:
                moved[name].append(dist)
            else:
                moved[name] = [dist]

    return moved


def save_all(
    unet,
    text_encoder,
    save_path,
    placeholder_token_ids=None,
    placeholder_tokens=None,
    save_lora=True,
    save_ti=True,
    target_replace_module_text=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    target_replace_module_unet=DEFAULT_TARGET_REPLACE,
    safe_form=True,
):
    if not safe_form:
        # save ti
        if save_ti:
            ti_path = _ti_lora_path(save_path)
            learned_embeds_dict = {}
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                print(
                    f"Current Learned Embeddings for {tok}:, id {tok_id} ",
                    learned_embeds[:4],
                )
                learned_embeds_dict[tok] = learned_embeds.detach().cpu()

            torch.save(learned_embeds_dict, ti_path)
            print("Ti saved to ", ti_path)

        # save text encoder
        if save_lora:

            save_lora_weight(
                unet, save_path, target_replace_module=target_replace_module_unet
            )
            print("Unet saved to ", save_path)

            save_lora_weight(
                text_encoder,
                _text_lora_path(save_path),
                target_replace_module=target_replace_module_text,
            )
            print("Text Encoder saved to ", _text_lora_path(save_path))

    else:
        assert save_path.endswith(
            ".safetensors"
        ), f"Save path : {save_path} should end with .safetensors"

        loras = {}
        embeds = {}

        if save_lora:

            loras["unet"] = (unet, target_replace_module_unet)
            loras["text_encoder"] = (text_encoder, target_replace_module_text)

        if save_ti:
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                print(
                    f"Current Learned Embeddings for {tok}:, id {tok_id} ",
                    learned_embeds[:4],
                )
                embeds[tok] = learned_embeds.detach().cpu()

        save_safeloras_with_embeds(loras, embeds, save_path)

