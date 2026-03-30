from jax.numpy import expand_dims, squeeze
from jax.lax import conv_general_dilated
from typing import Union, Optional, Sequence
from jaxtyping import PRNGKeyArray, Array
from equinox import field
from equinox.nn import Conv

class BaseRealConv(Conv):
    """General N-dimensional real-valued convolution."""

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

    def __call__(self, x: Array) -> Array:
        # Ensure input has correct rank
        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to Conv needs to have rank {unbatched_rank}, but input has shape {x.shape}."
            )

        if self.padding_mode != "ZEROS":
            raise NotImplementedError("Non-zero padding not implemented for real conv.")
        padding = self.padding

        # Add batch dimension for conv_general_dilated
        x = expand_dims(x, axis=0)
        x = conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.stride,
            padding=padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = squeeze(x, axis=0)

        if self.use_bias:
            x = x + self.bias
        return x


class RealConv1D(BaseRealConv):
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


class RealConv2D(BaseRealConv):
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
