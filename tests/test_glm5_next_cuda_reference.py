"""Numerical reference checks for GLM-5.3 CUDA kernels.

This test is intentionally opt-in because it needs a CUDA GPU and the PTX
produced by Krasis's build script.  Run it with, for example::

    KRASIS_CUDA_REFERENCE_PTX=/path/to/decode_kernels.ptx \
      python -m unittest tests.test_glm5_next_cuda_reference

The calculations on the reference side follow the released Transformers
implementation, including its BF16 boundaries.
"""

from __future__ import annotations

import ctypes
import os
import unittest


PTX_ENV = "KRASIS_CUDA_REFERENCE_PTX"


@unittest.skipUnless(os.environ.get(PTX_ENV), f"set {PTX_ENV} to run CUDA checks")
class Glm5NextCudaReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        from cuda.bindings import driver

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")
        torch.cuda.init()
        cls.torch = torch
        cls.driver = driver
        cls.stream = int(torch.cuda.current_stream().cuda_stream)
        cls.module = cls._checked(driver.cuModuleLoad(os.environ[PTX_ENV].encode()))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "module"):
            cls._checked(cls.driver.cuModuleUnload(cls.module))

    @staticmethod
    def _checked(result):
        status, *values = result
        if int(status) != 0:
            raise RuntimeError(f"CUDA driver call failed: {status}")
        if not values:
            return None
        return values[0] if len(values) == 1 else tuple(values)

    @classmethod
    def _function(cls, name: str):
        return cls._checked(cls.driver.cuModuleGetFunction(cls.module, name.encode()))

    @classmethod
    def _launch(
        cls,
        function,
        values: tuple[object, ...],
        types: tuple[object, ...],
        *,
        grid: int = 1,
        block: int = 256,
        shared_bytes: int = 0,
    ) -> None:
        cls._checked(
            cls.driver.cuLaunchKernel(
                function,
                grid,
                1,
                1,
                block,
                1,
                1,
                shared_bytes,
                cls.stream,
                (values, types),
                0,
            )
        )
        cls.torch.cuda.synchronize()

    def test_kpool_matches_released_transformers_bf16_contract(self) -> None:
        torch = self.torch
        torch.manual_seed(53)
        device = "cuda"
        head_dim = 128
        pool_size = 4
        token_count = 11
        eps = 1.0e-6

        raw_keys = torch.randn(token_count, head_dim, device=device, dtype=torch.bfloat16)
        raw_gates = torch.randn(token_count, head_dim, device=device, dtype=torch.bfloat16)
        norm_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
        norm_bias = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
        learned_ape = torch.randn(pool_size, head_dim, device=device, dtype=torch.bfloat16)

        pool_capacity = (token_count + pool_size - 1) // pool_size
        pooled = torch.full(
            (pool_capacity, head_dim),
            float("nan"),
            device=device,
            dtype=torch.bfloat16,
        )
        key_tail = torch.empty(pool_size, head_dim, device=device, dtype=torch.bfloat16)
        gate_tail = torch.empty_like(key_tail)
        function = self._function("glm5_next_kpool_update_kernel")
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        scalar = ctypes.c_float

        for position in range(token_count):
            self._launch(
                function,
                (
                    pooled.data_ptr(),
                    key_tail.data_ptr(),
                    gate_tail.data_ptr(),
                    raw_keys[position].data_ptr(),
                    raw_gates[position].data_ptr(),
                    norm_weight.data_ptr(),
                    norm_bias.data_ptr(),
                    learned_ape.data_ptr(),
                    position,
                    head_dim,
                    pool_size,
                    eps,
                ),
                (
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    integer,
                    integer,
                    integer,
                    scalar,
                ),
                shared_bytes=64 * ctypes.sizeof(ctypes.c_float),
            )

        normalized = torch.nn.functional.layer_norm(
            raw_keys,
            (head_dim,),
            norm_weight,
            norm_bias,
            eps,
        )
        expected_pools = []
        for start in range(0, token_count - pool_size + 1, pool_size):
            gate_group = raw_gates[start : start + pool_size]
            probabilities = (gate_group.float() + learned_ape.float()).softmax(dim=0).to(
                torch.bfloat16
            )
            expected_pools.append(
                (probabilities * normalized[start : start + pool_size]).sum(dim=0)
            )
        expected = torch.stack(expected_pools)
        actual = pooled[: len(expected_pools)]

        difference = (actual.float() - expected.float()).abs()
        maximum = float(difference.max())
        mean = float(difference.mean())
        self.assertLessEqual(maximum, 0.015625, f"max={maximum}, mean={mean}")
        self.assertLessEqual(mean, 0.001, f"max={maximum}, mean={mean}")

        tail_start = token_count - token_count % pool_size
        expected_tail = normalized[tail_start:]
        self.assertTrue(torch.equal(key_tail[: len(expected_tail)], expected_tail))
        self.assertTrue(torch.equal(gate_tail[: len(expected_tail)], raw_gates[tail_start:]))

    def test_kpool_index_expansion_appends_incomplete_tail(self) -> None:
        torch = self.torch
        device = "cuda"
        function = self._function("glm5_next_kpool_expand_indices_kernel")
        pointer = ctypes.c_void_p
        integer = ctypes.c_int

        def expand(selected: list[int], position: int, capacity: int) -> list[int]:
            selected_tensor = torch.tensor(selected, device=device, dtype=torch.int32)
            output = torch.empty(capacity, device=device, dtype=torch.int32)
            self._launch(
                function,
                (
                    output.data_ptr(),
                    selected_tensor.data_ptr(),
                    len(selected),
                    position,
                    4,
                    capacity,
                ),
                (pointer, pointer, integer, integer, integer, integer),
                grid=(capacity + 255) // 256,
            )
            return output.cpu().tolist()

        self.assertEqual(expand([1, 0], position=10, capacity=11), list(range(4, 8)) + list(range(4)) + [8, 9, 10])
        self.assertEqual(expand([], position=2, capacity=6), [0, 1, 2, -1, -1, -1])

    def test_dense_swiglu_uses_glm_raw_input_clamp(self) -> None:
        torch = self.torch
        torch.manual_seed(5301)
        device = "cuda"
        intermediate = 257
        limit = 10.0
        gate_up = (torch.randn(2, intermediate, device=device) * 18).to(
            torch.bfloat16
        )
        output = torch.empty(intermediate, device=device, dtype=torch.bfloat16)
        self._launch(
            self._function("deepseek_v4_swiglu_bf16"),
            (output.data_ptr(), gate_up.data_ptr(), intermediate, limit),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float),
            block=256,
        )

        gate = gate_up[0].float().clamp(max=limit)
        up = gate_up[1].float().clamp(min=-limit, max=limit)
        expected = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        self.assertTrue(torch.equal(output, expected))

    def test_kda_recurrence_matches_released_bf16_boundary(self) -> None:
        torch = self.torch
        torch.manual_seed(5302)
        device = "cuda"
        num_heads = 2
        head_dim = 8
        width = num_heads * head_dim
        query_scale = head_dim**-0.5
        gate_lower_bound = -5.0
        norm_eps = 1.0e-6

        convolved = torch.randn(3, num_heads, head_dim, device=device).to(
            torch.bfloat16
        )
        forget_projection = torch.randn(num_heads, head_dim, device=device).to(
            torch.bfloat16
        )
        beta_logits = torch.randn(num_heads, device=device).to(torch.bfloat16)
        output_gate = torch.randn(num_heads, head_dim, device=device).to(
            torch.bfloat16
        )
        a_log = torch.randn(num_heads, device=device, dtype=torch.float32) * 0.25
        dt_bias = torch.randn(num_heads, head_dim, device=device, dtype=torch.float32)
        output_norm = torch.randn(head_dim, device=device).to(torch.bfloat16)
        recurrent_state = torch.randn(
            num_heads, head_dim, head_dim, device=device, dtype=torch.float32
        ) * 0.1
        reference_state = recurrent_state.clone()
        output = torch.empty(num_heads, head_dim, device=device, dtype=torch.bfloat16)

        self._launch(
            self._function("glm5_next_kda_recurrent_decode_kernel"),
            (
                output.data_ptr(),
                convolved.data_ptr(),
                forget_projection.data_ptr(),
                beta_logits.data_ptr(),
                output_gate.data_ptr(),
                a_log.data_ptr(),
                dt_bias.data_ptr(),
                output_norm.data_ptr(),
                recurrent_state.data_ptr(),
                num_heads,
                head_dim,
                query_scale,
                gate_lower_bound,
                norm_eps,
            ),
            (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ),
            grid=num_heads,
            block=head_dim,
            shared_bytes=4 * head_dim * ctypes.sizeof(ctypes.c_float),
        )

        query, key, value = convolved.float()
        query = query / torch.sqrt(query.square().sum(-1, keepdim=True) + 1.0e-6)
        query = query * query_scale
        key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1.0e-6)
        forget = gate_lower_bound * torch.sigmoid(
            torch.exp(a_log)[:, None] * (forget_projection.float() + dt_bias)
        )
        reference_state = reference_state * torch.exp(forget)[..., None]
        memory = (reference_state * key[..., None]).sum(dim=-2)
        delta = (value - memory) * torch.sigmoid(beta_logits.float())[:, None]
        reference_state = reference_state + key[..., None] * delta[:, None, :]
        core = (reference_state * query[..., None]).sum(dim=-2).to(torch.bfloat16)
        inv_rms = torch.rsqrt(core.float().square().mean(-1, keepdim=True) + norm_eps)
        expected = (
            core.float()
            * inv_rms
            * output_norm.float()[None]
            * torch.sigmoid(output_gate.float())
        ).to(torch.bfloat16)

        output_error = (output.float() - expected.float()).abs()
        state_error = (recurrent_state - reference_state).abs()
        self.assertLessEqual(float(output_error.max()), 0.015625)
        self.assertLessEqual(float(output_error.mean()), 0.002)
        self.assertLessEqual(float(state_error.max()), 2.0e-5)


if __name__ == "__main__":
    unittest.main()
