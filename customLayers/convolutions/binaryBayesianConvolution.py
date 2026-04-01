from jax.numpy import expand_dims, squeeze, float32, ones
from jax.lax import tanh, logistic, conv_general_dilated, log
from jax.random import bernoulli
from typing import Union, Optional, Sequence
from jaxtyping import PRNGKeyArray, Array
from equinox import field
from jax.random import split, uniform
from equinox.nn import Conv


class BaseBinaryBayesianConv(Conv):
    """General N-dimensional convolution."""

    num_spatial_dims: int = field(static=True)
    weight: Array
    bias: Optional[Array]
    in_channels: int = field(static=True)
    out_channels: int = field(static=True)
    kernel_size: tuple[int, ...] = field(static=True)
    stride: tuple[int, ...] = field(static=True)
    padding: Union[str, tuple[tuple[int, int], ...]] = field(static=True)
    dilation: tuple[int, ...] = field(static=True)
    groups: int = field(static=True)
    use_bias: bool = field(static=True)
    padding_mode: str = field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )

    def __call__(self, x: Array, *, key: PRNGKeyArray, temperature: float = 1) -> Array:
        wkey, bkey = split(key, 2)
        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )

        if self.padding_mode != "ZEROS":
            x = self._nonzero_pad(x)
            padding = "VALID"
        else:
            padding = self.padding

        epsilon = uniform(wkey, self.weight.shape,
                          minval=1e-10, maxval=1 - 1e-10)
        logit_epsilon = log(epsilon) - log(1 - epsilon)
        weights = tanh((self.weight + 0.5 * logit_epsilon) / temperature)

        x = expand_dims(x, axis=0)
        x = conv_general_dilated(
            lhs=x,
            rhs=weights,
            window_strides=self.stride,
            padding=padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = squeeze(x, axis=0)
        return x

    def sample(self, x: Array, *, key: PRNGKeyArray, temperature: float = 1) -> Array:
        wkey, bkey = split(key, 2)
        p = logistic(2 * self.weight)
        weights = 2 * bernoulli(wkey, p).astype(float32) - 1

        x = expand_dims(x, axis=0)
        x = conv_general_dilated(
            lhs=x,
            rhs=weights,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = squeeze(x, axis=0)

        if self.use_bias:
            p_bias = logistic(2 * self.bias)
            biases = 2 * bernoulli(bkey, p_bias).astype(float32) - 1
            x = x + biases
        return x


class BinaryBayesianConv1D(BaseBinaryBayesianConv):
    """Performs a 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )


class BinaryBayesianConv2D(BaseBinaryBayesianConv):
    """Performs a 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )
