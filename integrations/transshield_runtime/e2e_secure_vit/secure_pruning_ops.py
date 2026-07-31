"""JAX primitives for exact-count secure token pruning.

The functions in this module use fixed compare-and-swap schedules so their
control flow is public and can be lowered by SPU.  They intentionally avoid
data-dependent indexing.
"""


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < int(value):
        result *= 2
    return result


def bitonic_sort_desc(values):
    """Sort ``[batch, token]`` values descending with one comparison per pair."""
    import jax.numpy as jnp

    token_count = int(values.shape[1])
    x = values
    stage_size = 2
    while stage_size <= token_count:
        stride = stage_size // 2
        while stride >= 1:
            positions = jnp.arange(token_count, dtype=jnp.int32)
            partners = positions ^ stride
            partner_values = x[:, partners]
            take_current_as_high = x >= partner_values
            high = jnp.where(take_current_as_high, x, partner_values)
            low = jnp.where(take_current_as_high, partner_values, x)

            is_left = positions < partners
            left_positions = jnp.where(is_left, positions, partners)
            descending_pair = (left_positions & stage_size) == 0
            left_value = jnp.where(descending_pair, high, low)
            right_value = jnp.where(descending_pair, low, high)
            x = jnp.where(is_left, left_value, right_value)
            stride //= 2
        stage_size *= 2
    return x


def exact_topk_keep_mask(score, active_mask, keep_count: int):
    """Return an exact-size top-k mask with lowest-index boundary tie breaking."""
    import jax.numpy as jnp

    token_count = int(score.shape[1])
    keep_count = int(keep_count)
    if keep_count <= 0 or keep_count > token_count:
        raise ValueError(f"keep_count must be within [1, {token_count}], got {keep_count}")

    active = jnp.asarray(active_mask).squeeze(-1) > 0
    masked_score = jnp.where(active, score, jnp.asarray(-1.0e6, dtype=score.dtype))
    padded_count = _next_power_of_two(token_count)
    if padded_count > token_count:
        padding = jnp.full(
            (int(score.shape[0]), padded_count - token_count),
            -1.0e6,
            dtype=score.dtype,
        )
        masked_score = jnp.concatenate([masked_score, padding], axis=1)

    sorted_score = bitonic_sort_desc(masked_score)
    threshold = sorted_score[:, keep_count - 1 : keep_count]
    greater = (score > threshold) & active
    equal = (score == threshold) & active
    remaining = keep_count - jnp.sum(greater.astype(jnp.int32), axis=1, keepdims=True)
    equal_rank = jnp.cumsum(equal.astype(jnp.int32), axis=1)
    selected_equal = equal & (equal_rank <= remaining)
    return (greater | selected_equal)[:, :, None]


def logical_uniform_mean(value, dropped_zero_value, logical_token_count: int):
    """Mean over a compact payload plus omitted logical zero-token outputs."""
    import jax.numpy as jnp

    physical_token_count = int(value.shape[2])
    logical_token_count = int(logical_token_count)
    if logical_token_count < physical_token_count:
        raise ValueError(
            f"logical_token_count={logical_token_count} is smaller than physical_token_count={physical_token_count}"
        )
    value_sum = jnp.sum(value, axis=2, keepdims=True)
    omitted_count = logical_token_count - physical_token_count
    if omitted_count:
        if dropped_zero_value is None:
            raise ValueError("dropped_zero_value is required when logical tokens were omitted")
        value_sum = value_sum + omitted_count * dropped_zero_value
    return value_sum / logical_token_count


def compact_topk_tokens(
    score,
    spatial_tokens,
    original_indices,
    keep_count: int,
    *,
    fxp_fraction_bits: int,
    original_token_count: int,
):
    """Sort scores and their token payload, then return the first ``keep_count``.

    Scores are packed with the original token index before comparison.  SPU
    fixed-point outputs lie on a ``2**-fxp_fraction_bits`` grid; multiplying the
    score by ``original_token_count + 1`` leaves enough representable space for
    an exact lowest-original-index tie breaker without reordering distinct
    fixed-point scores.
    """
    import jax.numpy as jnp

    token_count = int(score.shape[1])
    keep_count = int(keep_count)
    original_token_count = int(original_token_count)
    if keep_count <= 0 or keep_count > token_count:
        raise ValueError(f"keep_count must be within [1, {token_count}], got {keep_count}")
    if tuple(spatial_tokens.shape[:2]) != tuple(score.shape):
        raise ValueError(
            f"spatial token shape {spatial_tokens.shape} does not match score shape {score.shape}"
        )
    if tuple(original_indices.shape) != tuple(score.shape):
        raise ValueError(
            f"original index shape {original_indices.shape} does not match score shape {score.shape}"
        )

    tie_unit = 2.0 ** (-int(fxp_fraction_bits))
    packed_key = (
        score * float(original_token_count + 1)
        + (float(original_token_count) - original_indices.astype(score.dtype)) * tie_unit
    )

    padded_count = _next_power_of_two(token_count)
    if padded_count > token_count:
        pad_width = padded_count - token_count
        packed_key = jnp.concatenate(
            [
                packed_key,
                jnp.full((int(score.shape[0]), pad_width), -1.0e6, dtype=score.dtype),
            ],
            axis=1,
        )
        spatial_tokens = jnp.concatenate(
            [
                spatial_tokens,
                jnp.zeros(
                    (int(spatial_tokens.shape[0]), pad_width, int(spatial_tokens.shape[2])),
                    dtype=spatial_tokens.dtype,
                ),
            ],
            axis=1,
        )
        original_indices = jnp.concatenate(
            [
                original_indices,
                jnp.full(
                    (int(original_indices.shape[0]), pad_width),
                    original_token_count,
                    dtype=original_indices.dtype,
                ),
            ],
            axis=1,
        )

    keys = packed_key
    tokens = spatial_tokens
    indices = original_indices
    stage_size = 2
    while stage_size <= padded_count:
        stride = stage_size // 2
        while stride >= 1:
            positions = jnp.arange(padded_count, dtype=jnp.int32)
            partners = positions ^ stride

            partner_keys = keys[:, partners]
            partner_tokens = tokens[:, partners, :]
            partner_indices = indices[:, partners]
            take_current_as_high = keys >= partner_keys

            high_keys = jnp.where(take_current_as_high, keys, partner_keys)
            low_keys = jnp.where(take_current_as_high, partner_keys, keys)
            payload_condition = take_current_as_high[:, :, None]
            high_tokens = jnp.where(payload_condition, tokens, partner_tokens)
            low_tokens = jnp.where(payload_condition, partner_tokens, tokens)
            high_indices = jnp.where(take_current_as_high, indices, partner_indices)
            low_indices = jnp.where(take_current_as_high, partner_indices, indices)

            is_left = positions < partners
            left_positions = jnp.where(is_left, positions, partners)
            descending_pair = (left_positions & stage_size) == 0
            left_keys = jnp.where(descending_pair, high_keys, low_keys)
            right_keys = jnp.where(descending_pair, low_keys, high_keys)
            direction_condition = descending_pair[None, :, None]
            left_tokens = jnp.where(direction_condition, high_tokens, low_tokens)
            right_tokens = jnp.where(direction_condition, low_tokens, high_tokens)
            left_indices = jnp.where(descending_pair, high_indices, low_indices)
            right_indices = jnp.where(descending_pair, low_indices, high_indices)

            keys = jnp.where(is_left, left_keys, right_keys)
            side_condition = is_left[None, :, None]
            tokens = jnp.where(side_condition, left_tokens, right_tokens)
            indices = jnp.where(is_left, left_indices, right_indices)
            stride //= 2
        stage_size *= 2

    return tokens[:, :keep_count, :], indices[:, :keep_count]
