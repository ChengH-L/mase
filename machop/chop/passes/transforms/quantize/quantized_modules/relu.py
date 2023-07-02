from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from ..quantizers import (
    block_fp_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
)
from .utils import extract_required_config


class _ReLUBase(torch.nn.ReLU):
    bypass = None
    _required_config_keys = None
    _optional_config_keys = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.relu(x)
        else:
            x = self.x_quantizer(x)
            return F.relu(x, self.inplace)

    def get_quantized_output(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        return {"x": x}


class ReLUInteger(_ReLUBase):
    bypass = None
    _required_config_keys = ("name", "data_in_width", "data_in_frac_width")
    _optional_config_keys = ("bypass",)

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)
        # establish quantizers
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width, is_signed=False
        )
        self.config = self.construct_essential_config(config)
        self.x_width = x_width
        self.x_frac_width = x_frac_width

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        o_config["data_out_width"] = r_config["data_in_width"]
        return r_config | o_config

    def get_output_bitwidth(self) -> dict:
        return {
            "data_out_width": self.config["data_in_width"],
            "data_out_frac_width": self.config["data_in_frac_width"],
        }


class ReLUMinifloatDenorm(_ReLUBase):
    bypass = None
    _required_config_keys = (
        "name",
        "data_in_width",
        "data_in_exponent_width",
        "data_in_exponent_bias",
    )
    _optional_config_keys = ("bypass",)

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        self.x_quantizer = partial(
            minifloat_denorm_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )
        self.config = self.construct_essential_config()

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config


class ReLUMinifloatIEEE(_ReLUBase):
    bypass = None
    _required_config_keys = (
        "name",
        "data_in_width",
        "data_in_exponent_width",
        "data_in_exponent_bias",
    )
    _optional_config_keys = ("bypass",)

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        x_width, x_exponent_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
        )
        self.x_quantizer = partial(
            minifloat_ieee_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
        )
        self.config = self.construct_essential_config(config)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config


class ReLULog(_ReLUBase):
    bypass = None
    _required_config_keys = (
        "name",
        "data_in_width",
        "data_in_exponent_bias",
    )
    _optional_config_keys = ("bypass",)

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        x_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_bias"],
        )
        self.x_quantizer = partial(
            log_quantizer,
            width=x_width,
            exponent_bias=x_exponent_bias,
        )
        self.config = self.construct_essential_config(config)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config


class ReLULog(_ReLUBase):
    bypass = None
    _required_config_keys = (
        "name",
        "data_in_width",
        "data_in_exponent_bias",
    )
    _optional_config_keys = ("bypass",)

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        x_width, x_exponent_bias = (
            config["data_in_width"],
            config["data_in_exponent_bias"],
        )
        self.x_quantizer = partial(
            log_quantizer,
            width=x_width,
            exponent_bias=x_exponent_bias,
        )
        self.config = self.construct_essential_config(config)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config


class ReLUBlockFP(_ReLUBase):
    bypass = None
    _required_config_keys = (
        "data_in_width",
        "data_in_exponent_width",
        "data_in_exponent_bias",
        "data_in_block_size",
    )

    _optional_config_keys = ("bypass",)

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        x_width, x_exponent_width, x_exponent_bias, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias"],
            config["data_in_block_size"],
        )
        self.x_quantizer = partial(
            block_fp_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias=x_exponent_bias,
            block_size=x_block_size,
            skip_first_dim=True,
        )
        self.config = self.construct_essential_config(config)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.relu(x)
        else:
            x_shape = [i for i in x.shape]
            if x.ndim > 2:
                x = torch.flatten(x, 0, -3)
            x = self.x_quantizer(x)
            x = torch.reshape(x, x_shape)
            return F.relu(x, self.inplace)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config


class ReLUBlockMinifloat(_ReLUBase):
    bypass = None
    _required_config_keys = (
        "data_in_width",
        "data_in_exponent_width",
        "data_in_exponent_bias_width",
        "data_in_block_size",
    )

    _optional_config_keys = ("bypass",)

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.bypass = config.get("bypass", False)

        x_width, x_exponent_width, x_exponent_bias_width, x_block_size = (
            config["data_in_width"],
            config["data_in_exponent_width"],
            config["data_in_exponent_bias_width"],
            config["data_in_block_size"],
        )
        self.x_quantizer = partial(
            block_minifloat_quantizer,
            width=x_width,
            exponent_width=x_exponent_width,
            exponent_bias_width=x_exponent_bias_width,
            block_size=x_block_size,
            skip_first_dim=True,
        )
        self.config = self.construct_essential_config(config)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.relu(x)
        else:
            x_shape = [i for i in x.shape]
            if x.ndim > 2:
                x = torch.flatten(x, 0, -3)
            x = self.x_quantizer(x)
            x = torch.reshape(x, x_shape)
            return F.relu(x, self.inplace)

    def construct_essential_config(self, config):
        r_config = extract_required_config(self, config)
        o_config = {}
        o_config["bypass"] = config.get("bypass", False)
        return r_config | o_config