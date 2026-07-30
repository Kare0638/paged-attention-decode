import pytest
import torch

from src.reference import paged_attention_decode_reference


def _valid_inputs():
    batch, num_q_heads, num_kv_heads, head_dim = 2, 8, 2, 32
    page_size, num_pages = 16, 4
    return dict(
        q=torch.randn(batch, num_q_heads, head_dim, dtype=torch.float32),
        k_cache=torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.float32),
        v_cache=torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.float32),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([20, 20], dtype=torch.int32),
    )


def _call(**overrides):
    kwargs = _valid_inputs()
    kwargs.update(overrides)
    return paged_attention_decode_reference(**kwargs)


def test_valid_inputs_do_not_raise():
    _call()


@pytest.mark.parametrize(
    "overrides,exc_type",
    [
        pytest.param(
            dict(q=torch.randn(2, 8, 32, 1, dtype=torch.float32)), ValueError, id="q_wrong_ndim"
        ),
        pytest.param(
            dict(k_cache=torch.randn(4, 16, 2, dtype=torch.float32)),
            ValueError,
            id="k_cache_wrong_ndim",
        ),
        pytest.param(
            dict(v_cache=torch.randn(4, 16, 2, 16, dtype=torch.float32)),
            ValueError,
            id="v_cache_shape_mismatch",
        ),
        pytest.param(
            dict(block_table=torch.zeros(2, 2, 1, dtype=torch.int32)),
            ValueError,
            id="block_table_wrong_ndim",
        ),
        pytest.param(
            dict(seq_lens=torch.zeros(2, 1, dtype=torch.int32)),
            ValueError,
            id="seq_lens_wrong_ndim",
        ),
        pytest.param(
            dict(q=torch.randn(2, 8, 32, dtype=torch.float16)), TypeError, id="q_wrong_dtype"
        ),
        pytest.param(
            dict(k_cache=torch.randn(4, 16, 2, 32, dtype=torch.float16)),
            TypeError,
            id="k_cache_wrong_dtype",
        ),
        pytest.param(
            dict(v_cache=torch.randn(4, 16, 2, 32, dtype=torch.float16)),
            TypeError,
            id="v_cache_wrong_dtype",
        ),
        pytest.param(
            dict(block_table=torch.zeros(2, 2, dtype=torch.float32)),
            TypeError,
            id="block_table_wrong_dtype",
        ),
        pytest.param(
            dict(seq_lens=torch.zeros(2, dtype=torch.float32)),
            TypeError,
            id="seq_lens_wrong_dtype",
        ),
        pytest.param(
            dict(block_table=torch.zeros(3, 2, dtype=torch.int32)),
            ValueError,
            id="block_table_batch_mismatch",
        ),
        pytest.param(
            dict(seq_lens=torch.zeros(3, dtype=torch.int32)),
            ValueError,
            id="seq_lens_batch_mismatch",
        ),
        pytest.param(
            dict(q=torch.randn(2, 8, 16, dtype=torch.float32)), ValueError, id="head_dim_mismatch"
        ),
        pytest.param(
            dict(
                k_cache=torch.randn(4, 0, 2, 32, dtype=torch.float32),
                v_cache=torch.randn(4, 0, 2, 32, dtype=torch.float32),
            ),
            ValueError,
            id="page_size_zero",
        ),
        pytest.param(
            dict(
                k_cache=torch.randn(4, 16, 0, 32, dtype=torch.float32),
                v_cache=torch.randn(4, 16, 0, 32, dtype=torch.float32),
            ),
            ValueError,
            id="num_kv_heads_zero",
        ),
        pytest.param(
            dict(q=torch.randn(2, 5, 32, dtype=torch.float32)), ValueError, id="gqa_not_divisible"
        ),
        pytest.param(
            dict(seq_lens=torch.tensor([-1, 20], dtype=torch.int32)),
            ValueError,
            id="negative_seq_len",
        ),
        pytest.param(
            dict(seq_lens=torch.tensor([999, 20], dtype=torch.int32)),
            ValueError,
            id="seq_len_exceeds_block_table_capacity",
        ),
        pytest.param(
            dict(block_table=torch.tensor([[-1, 1], [2, 3]], dtype=torch.int32)),
            ValueError,
            id="negative_physical_page_id",
        ),
        pytest.param(
            dict(block_table=torch.tensor([[99, 1], [2, 3]], dtype=torch.int32)),
            ValueError,
            id="physical_page_id_out_of_range",
        ),
    ],
)
def test_invalid_inputs_raise(overrides, exc_type):
    with pytest.raises(exc_type):
        _call(**overrides)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device to mismatch against")
def test_device_mismatch_raises():
    with pytest.raises(ValueError):
        _call(k_cache=_valid_inputs()["k_cache"].cuda())
