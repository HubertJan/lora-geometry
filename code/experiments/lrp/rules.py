"""LRP rules for custom modules (e.g. ElementwiseProduct).

(migrated from SRC/src/glad/lrp/rules.py)
"""

import torch
from zennit.core import BasicHook, Hook


class PassThrough(Hook):
    """LRP rule for element-wise activations.

    Passes relevance through unchanged (R_input = R_output), preserving
    conservation across activations such as ReLU, LeakyReLU, or GELU.
    Without this rule, the activation's gradient derivative (e.g. alpha=0.01
    for negative LeakyReLU inputs) would scale the relevance, breaking the
    global sum-conservation property.
    """

    def backward(self, module, grad_input, grad_output):
        return (grad_output[0],)


def compute_mat_mul_relevance(
    A: torch.Tensor,
    V: torch.Tensor,
    R_O: torch.Tensor,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute LRP relevances R_A and R_V for matrix multiplication O = A @ V using AttnLRP rule.

    Args:
    A: torch.Tensor (J, I) - left matrix
    V: torch.Tensor (I, P) - right matrix
    R_O: torch.Tensor (J, P) - relevance at O
    eps: float - epsilon for stability

    Returns:
    tuple (R_A, R_V) both torch.Tensors
    """
    # Ensure tensors
    A = torch.as_tensor(A, dtype=torch.float32)
    V = torch.as_tensor(V, dtype=torch.float32)
    R_O = torch.as_tensor(R_O, dtype=torch.float32)

    O = A @ V  # Computed for verification/stability

    # Stabilized denominator (element-wise)
    denom = 2 * (O + eps * (O >= 0).float() - eps * (O < 0).float())

    # R_O / denom (element-wise)
    R_div_denom = R_O / denom

    # R_A: sum_p A_ji * V_ip * (R_jp / denom_jp)
    # Equivalent to A @ (V * R_div_denom.T).T efficient way
    R_A = torch.einsum("ji,ip,jp->ji", A, V, R_div_denom)

    # R_V: sum_j A_ji * V_ip * (R_jp / denom_jp)
    R_V = torch.einsum("ji,ip,jp->ip", A, V, R_div_denom)

    return R_A, R_V


class MatMulEpsilon(Hook):
    def forward(self, module, args, kwargs, output):
        """Forward hook to save module in-/outputs.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        args: tuple of :py:obj:`torch.Tensor`
            The input tensors passed to ``module.forward``.
        kwargs: tuple of object
            The keyword arguments passed to ``module.forward``.
        output: :py:obj:`torch.Tensor`
            The output tensor.
        """
        self.stored_tensors["input"] = args
        self.stored_tensors["kwargs"] = kwargs

    def backward(self, module, grad_input, grad_output):
        """Backward hook to compute LRP based on the class attributes.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        grad_input: :py:obj:`torch.Tensor`
            The input gradient tensor.
        grad_output: :py:obj:`torch.Tensor`
            The output gradient tensor.

        Returns
        -------
        tuple of :py:obj:`torch.nn.Module`
            The modified input gradient tensors.
        """
        original_input, *original_args = self.stored_tensors["input"]
        original_input = original_input.clone()
        original_kwargs = self.stored_tensors["kwargs"]
        inputs = []
        outputs = []

        R_As = []
        R_Vs = []
        for v_input, args, output in zip(
            original_input,
            original_args[0],
            grad_output[0],
        ):
            R_A, R_V = compute_mat_mul_relevance(
                v_input,
                args,
                output,
                eps=0.0,
            )
            R_As.append(R_A)
            R_Vs.append(R_V)
        relevances = (
            torch.stack(R_As, dim=0),
            torch.stack(R_Vs, dim=0),
        )
        assert isinstance(relevances, tuple)
        assert len(relevances) == len(grad_input)
        for original, relevance in zip(grad_input, relevances):
            assert original.shape == relevance.shape

        for i in range(len(grad_output[0])):
            relevance = torch.stack([
                relevances[0][i],
                torch.transpose(relevances[1][i], -1, -2),
            ])  # Stacking is, in theory, not needed as we sum up later, but we have to stack here beacuse we can't stack tensors with different shapes
            output = grad_output[0][i]
            assert torch.allclose(torch.sum(relevance), torch.sum(output), atol=1e-3), (
                f"Relevance sum {torch.sum(relevance)} does not match output relevance sum {torch.sum(output)}"
            )

        return relevances


def _equivariant_mapper(out_grad, outputs, eps: float = 0.0):
    C = outputs[0]
    R = out_grad
    C_stable = C + eps * torch.where(C >= 0, torch.ones_like(C), -torch.ones_like(C))
    return R / C_stable


def _equivariant_reducer(inputs, gradients):
    B = inputs[0]
    weighted_sum = gradients[0]
    return B * weighted_sum


class EpsilonNoBias(BasicHook):
    """LRP Epsilon rule for nn.Linear that zeros the bias in the backward pass.

    Standard LRP Epsilon with biases does not conserve relevance sums because
    the denominator is W@x + b while the numerator only sums W@x. This rule
    zeros the bias during the LRP backward computation so the denominator
    becomes W@x, ensuring sum(R_input) = sum(R_output) for each layer.
    """

    def __init__(self, epsilon: float = 0.0):
        eps = epsilon
        super().__init__(
            input_modifiers=[lambda x: x],
            param_modifiers=[
                lambda param, name: torch.zeros_like(param) if "bias" in name else param
            ],
            output_modifiers=[lambda output: output],
            gradient_mapper=lambda out_grad, outputs: (
                out_grad
                / (
                    outputs[0]
                    + eps * (outputs[0] >= 0).float()
                    - eps * (outputs[0] < 0).float()
                )
            ),
            reducer=lambda inputs, gradients: inputs[0] * gradients[0],
        )


class EquivariantRule(BasicHook):
    """LRP rule for torch.matmul(A, V): AttnLRP bilinear decomposition."""

    def __init__(self, epsilon: float = 0.0):
        eps = epsilon
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param, _: param],
            output_modifiers=[lambda output: output],
            gradient_mapper=lambda out_grad, outputs: _equivariant_mapper(
                out_grad, outputs, eps=eps
            ),
            reducer=_equivariant_reducer,
        )
