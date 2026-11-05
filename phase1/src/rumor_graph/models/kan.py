
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineKANLayer(nn.Module):
    """
    A spline-based KAN layer.

    Each input-output connection has:
    - a learnable base weight
    - a learnable set of B-spline coefficients
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_degree: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ):
        super().__init__()

        if in_features < 1 or out_features < 1:
            raise ValueError(
                "Feature dimensions must be positive"
            )

        if grid_size < 1 or spline_degree < 1:
            raise ValueError(
                "grid_size and spline_degree must be positive"
            )

        low, high = grid_range

        if not (
            math.isfinite(low)
            and math.isfinite(high)
            and low < high
        ):
            raise ValueError("Invalid grid_range")

        self.in_features = in_features
        self.out_features = out_features
        self.spline_degree = spline_degree

        # 1. Define the spline grid.
        step = (high - low) / grid_size

        knots = torch.linspace(
            low - spline_degree * step,
            high + spline_degree * step,
            steps=grid_size + 2 * spline_degree + 1,
        )

        # A buffer moves with the model to CPU/GPU
        # but is not learned by the optimizer.
        self.register_buffer("knots", knots)

        # Number of B-spline basis functions.
        self.num_basis = (
            knots.numel() - spline_degree - 1
        )

        # 2. Learnable parameters.
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.spline_coeff = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                self.num_basis,
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # 3. Initialise the parameters.
        nn.init.kaiming_uniform_(
            self.base_weight,
            a=math.sqrt(5),
        )

        nn.init.normal_(
            self.spline_coeff,
            mean=0.0,
            std=0.02 / math.sqrt(in_features),
        )

    def bspline_basis(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate B-spline basis functions.

        Input shape:
            [batch_size, in_features]

        Output shape:
            [batch_size, in_features, num_basis]
        """

        if x.ndim != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected input shape [batch, {self.in_features}]"
            )

        if not x.is_floating_point():
            raise TypeError(
                "KAN input must be floating point"
            )

        x = x.unsqueeze(-1)

        knots = self.knots

        # Degree-0 basis: 1 inside its knot interval,
        # otherwise 0.
        basis = (
            (x >= knots[:-1])
            & (x < knots[1:])
        ).to(x.dtype)

        # Cox-de Boor recursion:
        # build higher-degree spline bases.
        for degree in range(1, self.spline_degree + 1):

            left = (
                (x - knots[:-degree - 1])
                / (
                    knots[degree:-1]
                    - knots[:-degree - 1]
                )
                * basis[..., :-1]
            )

            right = (
                (knots[degree + 1:] - x)
                / (
                    knots[degree + 1:]
                    - knots[1:-degree]
                )
                * basis[..., 1:]
            )

            basis = left + right

        return basis

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Calculate the spline basis values.
        basis = self.bspline_basis(x)

        # Base transformation using SiLU.
        base_output = F.linear(
            F.silu(x),
            self.base_weight,
        )

        # Weighted sum of spline basis functions.
        spline_output = torch.einsum(
            "bik,oik->bo",
            basis,
            self.spline_coeff,
        )

        # Combine base and spline contributions.
        output = (
            base_output
            + spline_output
            + self.bias
        )

        return output



class SplineKAN(nn.Module):
    """
    Small two-layer KAN network.

    This network operates on feature vectors.
    It is not yet a graph neural network.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_degree: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ):
        super().__init__()

        self.layer1 = SplineKANLayer(
            in_features=in_features,
            out_features=hidden_features,
            grid_size=grid_size,
            spline_degree=spline_degree,
            grid_range=grid_range,
        )

        self.layer2 = SplineKANLayer(
            in_features=hidden_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_degree=spline_degree,
            grid_range=grid_range,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layer1(x)

        x = self.layer2(x)

        return x
