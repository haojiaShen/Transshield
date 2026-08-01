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


def _bitonic_pair_schedule(token_count: int, stage_size: int, stride: int):
    """Return public pair indices and the permutation back to token order."""
    left = tuple(
        position
        for position in range(int(token_count))
        if position < (position ^ int(stride))
    )
    right = tuple(position ^ int(stride) for position in left)
    paired_order = left + right
    paired_offset = {position: offset for offset, position in enumerate(paired_order)}
    restore = tuple(paired_offset[position] for position in range(int(token_count)))
    descending = tuple((position & int(stage_size)) == 0 for position in left)
    return left, right, restore, descending


def normalize_pruning_schedule(
    pruning_loc,
    token_keep_counts,
    *,
    depth: int,
    max_token_count=None,
):
    """Validate and normalize the public cumulative pruning schedule."""
    depth = int(depth)
    locations = tuple(int(value) for value in pruning_loc)
    keep_counts = tuple(int(value) for value in token_keep_counts)
    if len(locations) != len(keep_counts):
        raise ValueError(
            "pruning location/count length mismatch: "
            f"{len(locations)} locations vs {len(keep_counts)} counts"
        )
    if any(location < 0 or location >= depth for location in locations):
        raise ValueError(f"pruning locations must be within executed depth={depth}: {locations}")
    if any(left >= right for left, right in zip(locations, locations[1:])):
        raise ValueError(f"pruning locations must be strictly increasing: {locations}")
    if any(count <= 0 for count in keep_counts):
        raise ValueError(f"token keep counts must be positive: {keep_counts}")
    if max_token_count is not None and any(count > int(max_token_count) for count in keep_counts):
        raise ValueError(
            f"token keep counts cannot exceed {int(max_token_count)}: {keep_counts}"
        )
    if any(left < right for left, right in zip(keep_counts, keep_counts[1:])):
        raise ValueError(f"cumulative token keep counts must be non-increasing: {keep_counts}")
    return locations, keep_counts


def bitonic_sort_desc(values):
    """Sort ``[batch, token]`` values descending with one comparison per pair."""
    import jax.numpy as jnp

    token_count = int(values.shape[1])
    x = values
    stage_size = 2
    while stage_size <= token_count:
        stride = stage_size // 2
        while stride >= 1:
            left, right, restore, descending = _bitonic_pair_schedule(
                token_count,
                stage_size,
                stride,
            )
            left_value = x[:, left]
            right_value = x[:, right]
            take_left_as_high = left_value >= right_value
            high = jnp.where(take_left_as_high, left_value, right_value)
            low = jnp.where(take_left_as_high, right_value, left_value)
            descending_pair = jnp.asarray(descending, dtype=jnp.bool_)[None, :]
            sorted_left = jnp.where(descending_pair, high, low)
            sorted_right = jnp.where(descending_pair, low, high)
            x = jnp.concatenate([sorted_left, sorted_right], axis=1)[:, restore]
            stride //= 2
        stage_size *= 2
    return x


def exact_topk_keep_mask(score, active_mask, keep_count: int, *, unique_keys: bool = False):
    """Return an exact-size top-k mask with deterministic boundary handling.

    ``unique_keys`` is a public graph option for callers that have already
    packed a unique tie-breaker into every active score.  In that case the
    threshold comparison alone selects exactly ``keep_count`` entries and the
    generic secret equality/rank path is unnecessary.
    """
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
    if unique_keys:
        return ((score >= threshold) & active)[:, :, None]
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


def pack_topk_key(
    score,
    original_indices,
    *,
    fxp_fraction_bits: int,
    original_token_count: int,
):
    """Pack a fixed-point score with the public lowest-original-index tie rule."""
    import jax.numpy as jnp

    original_token_count = int(original_token_count)
    if tuple(original_indices.shape) != tuple(score.shape):
        raise ValueError(
            f"original index shape {original_indices.shape} does not match score shape {score.shape}"
        )
    tie_unit = 2.0 ** (-int(fxp_fraction_bits))
    return (
        score * float(original_token_count + 1)
        + (float(original_token_count) - original_indices.astype(jnp.asarray(score).dtype)) * tie_unit
    )


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

    packed_key = pack_topk_key(
        score,
        original_indices,
        fxp_fraction_bits=fxp_fraction_bits,
        original_token_count=original_token_count,
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
            left, right, restore, descending = _bitonic_pair_schedule(
                padded_count,
                stage_size,
                stride,
            )
            left_keys = keys[:, left]
            right_keys = keys[:, right]
            left_tokens = tokens[:, left, :]
            right_tokens = tokens[:, right, :]
            left_indices = indices[:, left]
            right_indices = indices[:, right]
            take_left_as_high = left_keys >= right_keys

            high_keys = jnp.where(take_left_as_high, left_keys, right_keys)
            low_keys = jnp.where(take_left_as_high, right_keys, left_keys)
            payload_condition = take_left_as_high[:, :, None]
            high_tokens = jnp.where(payload_condition, left_tokens, right_tokens)
            low_tokens = jnp.where(payload_condition, right_tokens, left_tokens)
            high_indices = jnp.where(take_left_as_high, left_indices, right_indices)
            low_indices = jnp.where(take_left_as_high, right_indices, left_indices)

            descending_pair = jnp.asarray(descending, dtype=jnp.bool_)[None, :]
            left_keys = jnp.where(descending_pair, high_keys, low_keys)
            right_keys = jnp.where(descending_pair, low_keys, high_keys)
            direction_condition = descending_pair[:, :, None]
            left_tokens = jnp.where(direction_condition, high_tokens, low_tokens)
            right_tokens = jnp.where(direction_condition, low_tokens, high_tokens)
            left_indices = jnp.where(descending_pair, high_indices, low_indices)
            right_indices = jnp.where(descending_pair, low_indices, high_indices)

            keys = jnp.concatenate([left_keys, right_keys], axis=1)[:, restore]
            tokens = jnp.concatenate([left_tokens, right_tokens], axis=1)[:, restore, :]
            indices = jnp.concatenate([left_indices, right_indices], axis=1)[:, restore]
            stride //= 2
        stage_size *= 2

    return tokens[:, :keep_count, :], indices[:, :keep_count]
