
import pytest
import torch

from rumor_graph.models.kan import (
    SplineKANLayer,
    SplineKAN,
)


def test_layer_output_shape():
    layer = SplineKANLayer(4, 6)

    x = torch.randn(7, 4)
    y = layer(x)

    assert y.shape == (7, 6)
    assert torch.isfinite(y).all()


def test_basis_shape_and_partition():
    layer = SplineKANLayer(
        2, 3,
        grid_size=5,
        spline_degree=3,
    )

    x = torch.tensor([
        [-1.0, 0.0],
        [0.75, 1.5],
    ])

    basis = layer.bspline_basis(x)

    assert basis.shape == (2, 2, 8)

    # On the interior grid, the basis functions
    # should sum to approximately 1.
    assert torch.allclose(
        basis.sum(dim=-1),
        torch.ones_like(x),
        atol=1e-5,
    )


def test_backward_gradients():
    torch.manual_seed(3)

    layer = SplineKANLayer(2, 3)

    x = torch.tensor(
        [
            [0.25, -0.5],
            [0.5, 0.75],
        ],
        requires_grad=True,
    )

    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert layer.base_weight.grad is not None
    assert torch.isfinite(
        layer.base_weight.grad
    ).all()

    assert layer.spline_coeff.grad is not None
    assert torch.isfinite(
        layer.spline_coeff.grad
    ).all()

    assert layer.spline_coeff.grad.abs().sum() > 0


def test_input_gradient_can_train_coefficients():
    """
    Check gradients through input derivatives.

    This capability will be useful later for
    interpretability-aware training.
    """

    torch.manual_seed(3)

    layer = SplineKANLayer(2, 1)

    x = torch.tensor(
        [
            [0.25, -0.5],
            [0.5, 0.75],
        ],
        requires_grad=True,
    )

    output = layer(x)

    input_gradient = torch.autograd.grad(
        output.sum(),
        x,
        create_graph=True,
    )[0]

    penalty = input_gradient.square().mean()
    penalty.backward()

    assert layer.spline_coeff.grad is not None

    assert torch.isfinite(
        layer.spline_coeff.grad
    ).all()

    assert layer.spline_coeff.grad.abs().sum() > 0


def test_optimizer_changes_coefficients():
    torch.manual_seed(1)

    layer = SplineKANLayer(1, 1)

    optimizer = torch.optim.Adam(
        layer.parameters(),
        lr=0.05,
    )

    x = torch.tensor([
        [-1.0],
        [0.2],
        [0.8],
    ])

    target = x.square()

    old_coefficients = (
        layer.spline_coeff.detach().clone()
    )

    optimizer.zero_grad()

    loss = (layer(x) - target).square().mean()

    loss.backward()
    optimizer.step()

    assert not torch.allclose(
        layer.spline_coeff,
        old_coefficients,
    )


def test_two_layer_network():
    model = SplineKAN(
        in_features=4,
        hidden_features=16,
        out_features=4,
    )

    x = torch.randn(5, 4)

    output = model(x)

    assert output.shape == (5, 4)


def test_outside_grid_is_finite():
    layer = SplineKANLayer(2, 3)

    x = torch.tensor([
        [-20.0, 20.0],
    ])

    output = layer(x)

    assert torch.isfinite(output).all()

    # Both inputs are outside the complete
    # extended knot support.
    assert torch.count_nonzero(
        layer.bspline_basis(x)
    ) == 0


def test_invalid_configuration():
    with pytest.raises(
        ValueError,
        match="grid_range",
    ):
        SplineKANLayer(
            2, 3,
            grid_range=(2.0, -2.0),
        )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        SplineKANLayer(0, 3)


def test_learns_simple_nonlinear_function():
    torch.manual_seed(7)

    model = SplineKANLayer(1, 1)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.03,
    )

    x = torch.linspace(
        -1.5, 1.5, 60
    ).unsqueeze(1)

    target = x.square()

    with torch.no_grad():
        initial_loss = (
            (model(x) - target).square().mean().item()
        )

    # Train on synthetic data.
    for _ in range(120):
        optimizer.zero_grad()

        prediction = model(x)

        loss = (
            (prediction - target).square().mean()
        )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = (
            (model(x) - target).square().mean().item()
        )

    assert final_loss < 0.1 * initial_loss
