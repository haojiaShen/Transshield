"""JAX primitives for exact-count secure token pruning.

The functions in this module use fixed compare-and-swap schedules so their
control flow is public and can be lowered by SPU.  They intentionally avoid
data-dependent indexing.
"""

from functools import lru_cache


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


@lru_cache(maxsize=None)
def _bitonic_full_schedule(token_count: int):
    """Return every public compare layer in a bitonic sorting network."""
    token_count = int(token_count)
    if token_count <= 0 or token_count & (token_count - 1):
        raise ValueError(f"bitonic token_count must be a positive power of two, got {token_count}")
    layers = []
    stage_size = 2
    while stage_size <= token_count:
        stride = stage_size // 2
        while stride >= 1:
            left, right, restore, descending = _bitonic_pair_schedule(
                token_count,
                stage_size,
                stride,
            )
            layers.append((left, right, restore, descending, ()))
            stride //= 2
        stage_size *= 2
    return tuple(layers)


def _parallel_network_layers(left_layers, right_layers):
    """Overlay two comparator-layer sequences that operate on disjoint wires."""
    return tuple(
        (left_layers[index] if index < len(left_layers) else ())
        + (right_layers[index] if index < len(right_layers) else ())
        for index in range(max(len(left_layers), len(right_layers)))
    )


def _greatest_power_of_two_less_than(value: int) -> int:
    return 1 << ((int(value) - 1).bit_length() - 1)


@lru_cache(maxsize=None)
def _bitonic_arbitrary_merge_layers(start: int, count: int, descending: bool):
    """Build Batcher's bitonic merge network for an arbitrary wire count."""
    start = int(start)
    count = int(count)
    descending = bool(descending)
    if count <= 1:
        return ()
    stride = _greatest_power_of_two_less_than(count)
    first_layer = tuple(
        (index, index + stride, descending)
        for index in range(start, start + count - stride)
    )
    tail = _parallel_network_layers(
        _bitonic_arbitrary_merge_layers(start, stride, descending),
        _bitonic_arbitrary_merge_layers(start + stride, count - stride, descending),
    )
    return (first_layer,) + tail


@lru_cache(maxsize=None)
def _bitonic_arbitrary_sort_layers(start: int, count: int, descending: bool):
    """Build an exact arbitrary-length bitonic sorting network."""
    start = int(start)
    count = int(count)
    descending = bool(descending)
    if count <= 1:
        return ()
    left_count = count // 2
    bitonic_input = _parallel_network_layers(
        _bitonic_arbitrary_sort_layers(start, left_count, not descending),
        _bitonic_arbitrary_sort_layers(
            start + left_count,
            count - left_count,
            descending,
        ),
    )
    return bitonic_input + _bitonic_arbitrary_merge_layers(start, count, descending)


def _network_layer_schedule(token_count: int, pairs):
    """Convert public comparator triples into gather/restore indices."""
    left = tuple(pair[0] for pair in pairs)
    right = tuple(pair[1] for pair in pairs)
    descending = tuple(pair[2] for pair in pairs)
    compared = frozenset(left + right)
    untouched = tuple(index for index in range(int(token_count)) if index not in compared)
    gathered_order = left + right + untouched
    gathered_offset = {position: offset for offset, position in enumerate(gathered_order)}
    restore = tuple(gathered_offset[position] for position in range(int(token_count)))
    return left, right, restore, descending, untouched


@lru_cache(maxsize=None)
def _bitonic_arbitrary_full_schedule(token_count: int):
    """Return an exact descending sorting network without power-of-two padding."""
    token_count = int(token_count)
    if token_count <= 0:
        raise ValueError(f"bitonic token_count must be positive, got {token_count}")
    return tuple(
        _network_layer_schedule(token_count, layer)
        for layer in _bitonic_arbitrary_sort_layers(0, token_count, True)
    )


def _slice_network_schedule(token_count: int, full_layers, output_indices):
    """Backward-slice an arbitrary sorting network to requested output wires."""
    outputs = tuple(int(index) for index in output_indices)
    if not outputs:
        raise ValueError("bitonic selection requires at least one output index")
    if any(index < 0 or index >= int(token_count) for index in outputs):
        raise ValueError(
            f"bitonic output indices must be within [0, {int(token_count)}), got {outputs}"
        )

    needed = set(outputs)
    selected_pair_offsets = []
    for left, right, _, _, _ in reversed(full_layers):
        offsets = []
        for offset, (left_index, right_index) in enumerate(zip(left, right)):
            if left_index in needed or right_index in needed:
                offsets.append(offset)
                needed.add(left_index)
                needed.add(right_index)
        selected_pair_offsets.append(tuple(offsets))
    selected_pair_offsets.reverse()

    layers = []
    for (full_left, full_right, _, full_descending, _), offsets in zip(
        full_layers,
        selected_pair_offsets,
    ):
        pairs = tuple(
            (
                full_left[offset],
                full_right[offset],
                full_descending[offset],
            )
            for offset in offsets
        )
        layers.append(_network_layer_schedule(token_count, pairs))
    return tuple(layers)


@lru_cache(maxsize=None)
def _bitonic_selection_schedule(token_count: int, output_indices):
    """Slice away comparators that cannot affect the requested sorted outputs.

    The schedule is obtained by walking the complete sorting network backwards.
    A compare-and-swap is retained exactly when either of its output wires is
    still needed.  Both input wires then become dependencies.  The resulting
    public, fixed-topology network produces the requested outputs exactly while
    avoiding comparisons and payload muxes outside their dependency cone.
    """
    token_count = int(token_count)
    full_layers = _bitonic_full_schedule(token_count)
    return _slice_network_schedule(token_count, full_layers, output_indices)


@lru_cache(maxsize=None)
def _bitonic_unpadded_selection_schedule(token_count: int, output_indices):
    """Return an exact output-sliced network over the actual token count."""
    token_count = int(token_count)
    full_layers = _bitonic_arbitrary_full_schedule(token_count)
    return _slice_network_schedule(token_count, full_layers, output_indices)


@lru_cache(maxsize=None)
def _odd_even_merge_sequence(start: int, count: int, stride: int):
    """Return Batcher odd-even merge comparators for a power-of-two range."""
    start = int(start)
    count = int(count)
    stride = int(stride)
    doubled_stride = stride * 2
    if doubled_stride < count:
        pairs = list(_odd_even_merge_sequence(start, count, doubled_stride))
        pairs.extend(
            _odd_even_merge_sequence(start + stride, count, doubled_stride)
        )
        pairs.extend(
            (index, index + stride)
            for index in range(
                start + stride,
                start + count - stride,
                doubled_stride,
            )
        )
        return tuple(pairs)
    return ((start, start + stride),)


@lru_cache(maxsize=None)
def _odd_even_sort_sequence(start: int, count: int):
    """Return a descending odd-even merge-sort comparator sequence."""
    start = int(start)
    count = int(count)
    if count <= 1:
        return ()
    half = count // 2
    return (
        _odd_even_sort_sequence(start, half)
        + _odd_even_sort_sequence(start + half, half)
        + _odd_even_merge_sequence(start, count, 1)
    )


@lru_cache(maxsize=None)
def _odd_even_padded_layers(token_count: int):
    """Layerize an odd-even sorting sequence at the earliest safe wire depth."""
    token_count = int(token_count)
    if token_count <= 0 or token_count & (token_count - 1):
        raise ValueError(
            f"odd-even token_count must be a positive power of two, got {token_count}"
        )
    wire_depth = [-1] * token_count
    layers = []
    for left, right in _odd_even_sort_sequence(0, token_count):
        depth = max(wire_depth[left], wire_depth[right]) + 1
        while len(layers) <= depth:
            layers.append([])
        layers[depth].append((left, right))
        wire_depth[left] = depth
        wire_depth[right] = depth
    return tuple(tuple(layer) for layer in layers)


@lru_cache(maxsize=None)
def _odd_even_unpadded_full_schedule(token_count: int):
    """Remove public ``-inf`` padding wires from an odd-even sorting network.

    Real values always beat the conceptual padding sentinel in descending
    comparisons.  Those comparisons therefore become public routes instead of
    secret compare-and-swaps.  ``restore`` records the resulting compact wire
    permutation for each layer.
    """
    token_count = int(token_count)
    if token_count <= 0:
        raise ValueError(f"odd-even token_count must be positive, got {token_count}")
    padded_count = _next_power_of_two(token_count)
    slots = list(range(token_count)) + [None] * (padded_count - token_count)
    schedule = []

    for padded_pairs in _odd_even_padded_layers(padded_count):
        output_descriptors = [None] * padded_count
        compared_positions = set()
        left = []
        right = []
        for padded_left, padded_right in padded_pairs:
            compared_positions.add(padded_left)
            compared_positions.add(padded_right)
            left_source = slots[padded_left]
            right_source = slots[padded_right]
            if left_source is not None and right_source is not None:
                pair_offset = len(left)
                left.append(left_source)
                right.append(right_source)
                output_descriptors[padded_left] = ("pair_high", pair_offset)
                output_descriptors[padded_right] = ("pair_low", pair_offset)
            elif left_source is not None:
                output_descriptors[padded_left] = ("pass", left_source)
            elif right_source is not None:
                output_descriptors[padded_left] = ("pass", right_source)

        for padded_index, source in enumerate(slots):
            if padded_index not in compared_positions and source is not None:
                output_descriptors[padded_index] = ("pass", source)

        compared_inputs = frozenset(left + right)
        untouched = tuple(
            index for index in range(token_count) if index not in compared_inputs
        )
        untouched_offset = {source: offset for offset, source in enumerate(untouched)}
        pair_count = len(left)
        restore = []
        next_slots = [None] * padded_count
        compact_output_index = 0
        for padded_index, descriptor in enumerate(output_descriptors):
            if descriptor is None:
                continue
            kind, offset = descriptor
            if kind == "pair_high":
                restore.append(offset)
            elif kind == "pair_low":
                restore.append(pair_count + offset)
            else:
                restore.append(2 * pair_count + untouched_offset[offset])
            next_slots[padded_index] = compact_output_index
            compact_output_index += 1
        if compact_output_index != token_count:
            raise AssertionError(
                f"odd-even public routing lost wires: {compact_output_index} vs {token_count}"
            )
        schedule.append(
            (
                tuple(left),
                tuple(right),
                tuple(restore),
                (True,) * pair_count,
                untouched,
            )
        )
        slots = next_slots
    return tuple(schedule)


def _slice_routed_network_schedule(token_count: int, full_layers, output_indices):
    """Backward-slice a network whose public routes may permute compact wires."""
    token_count = int(token_count)
    outputs = tuple(int(index) for index in output_indices)
    if not outputs:
        raise ValueError("odd-even selection requires at least one output index")
    if any(index < 0 or index >= token_count for index in outputs):
        raise ValueError(
            f"odd-even output indices must be within [0, {token_count}), got {outputs}"
        )

    needed = set(outputs)
    selected_offsets_by_layer = []
    for left, right, restore, _, untouched in reversed(full_layers):
        pair_count = len(left)
        selected_offsets = set()
        upstream_needed = set()
        for output_index in needed:
            producer = restore[output_index]
            if producer < pair_count:
                selected_offsets.add(producer)
            elif producer < 2 * pair_count:
                selected_offsets.add(producer - pair_count)
            else:
                upstream_needed.add(untouched[producer - 2 * pair_count])
        for offset in selected_offsets:
            upstream_needed.add(left[offset])
            upstream_needed.add(right[offset])
        selected_offsets_by_layer.append(tuple(sorted(selected_offsets)))
        needed = upstream_needed
    selected_offsets_by_layer.reverse()

    sliced_layers = []
    for (left, right, restore, descending, untouched), selected_offsets in zip(
        full_layers,
        selected_offsets_by_layer,
    ):
        pair_count = len(left)
        selected_offset_map = {
            old_offset: new_offset
            for new_offset, old_offset in enumerate(selected_offsets)
        }
        sliced_left = tuple(left[offset] for offset in selected_offsets)
        sliced_right = tuple(right[offset] for offset in selected_offsets)
        sliced_descending = tuple(descending[offset] for offset in selected_offsets)
        compared_inputs = frozenset(sliced_left + sliced_right)
        sliced_untouched = tuple(
            index for index in range(token_count) if index not in compared_inputs
        )
        untouched_offset = {
            source: offset for offset, source in enumerate(sliced_untouched)
        }
        sliced_pair_count = len(selected_offsets)
        sliced_restore = []
        for producer in restore:
            if producer < pair_count:
                old_offset = producer
                if old_offset in selected_offset_map:
                    sliced_restore.append(selected_offset_map[old_offset])
                else:
                    source = left[old_offset]
                    sliced_restore.append(
                        2 * sliced_pair_count + untouched_offset[source]
                    )
            elif producer < 2 * pair_count:
                old_offset = producer - pair_count
                if old_offset in selected_offset_map:
                    sliced_restore.append(
                        sliced_pair_count + selected_offset_map[old_offset]
                    )
                else:
                    source = right[old_offset]
                    sliced_restore.append(
                        2 * sliced_pair_count + untouched_offset[source]
                    )
            else:
                source = untouched[producer - 2 * pair_count]
                sliced_restore.append(
                    2 * sliced_pair_count + untouched_offset[source]
                )
        if sorted(sliced_restore) != list(range(token_count)):
            raise AssertionError("sliced odd-even layer is not a wire permutation")
        sliced_layers.append(
            (
                sliced_left,
                sliced_right,
                tuple(sliced_restore),
                sliced_descending,
                sliced_untouched,
            )
        )
    return tuple(sliced_layers)


@lru_cache(maxsize=None)
def _odd_even_unpadded_selection_schedule(token_count: int, output_indices):
    """Return an exact output-sliced odd-even network over actual wires."""
    token_count = int(token_count)
    full_layers = _odd_even_unpadded_full_schedule(token_count)
    return _slice_routed_network_schedule(token_count, full_layers, output_indices)


def _apply_bitonic_value_schedule(values, schedule):
    """Apply a public bitonic schedule to a ``[batch, token]`` value tensor."""
    import jax.numpy as jnp

    x = values
    for left, right, restore, descending, untouched in schedule:
        left_value = x[:, left]
        right_value = x[:, right]
        take_left_as_high = left_value >= right_value
        high = jnp.where(take_left_as_high, left_value, right_value)
        low = jnp.where(take_left_as_high, right_value, left_value)
        descending_pair = jnp.asarray(descending, dtype=jnp.bool_)[None, :]
        sorted_left = jnp.where(descending_pair, high, low)
        sorted_right = jnp.where(descending_pair, low, high)
        gathered = [sorted_left, sorted_right]
        if untouched:
            gathered.append(x[:, untouched])
        x = jnp.concatenate(gathered, axis=1)[:, restore]
    return x


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
    token_count = int(values.shape[1])
    return _apply_bitonic_value_schedule(values, _bitonic_full_schedule(token_count))


def bitonic_select_desc(values, output_indices):
    """Return selected outputs of a descending bitonic sort exactly."""
    outputs = tuple(int(index) for index in output_indices)
    token_count = int(values.shape[1])
    selected = _apply_bitonic_value_schedule(
        values,
        _bitonic_selection_schedule(token_count, outputs),
    )
    return selected[:, outputs]


def bitonic_unpadded_select_desc(values, output_indices):
    """Select exact sorted outputs without padding to a power of two."""
    outputs = tuple(int(index) for index in output_indices)
    token_count = int(values.shape[1])
    selected = _apply_bitonic_value_schedule(
        values,
        _bitonic_unpadded_selection_schedule(token_count, outputs),
    )
    return selected[:, outputs]


def odd_even_unpadded_select_desc(values, output_indices):
    """Select exact sorted outputs with an unpadded odd-even network."""
    outputs = tuple(int(index) for index in output_indices)
    token_count = int(values.shape[1])
    selected = _apply_bitonic_value_schedule(
        values,
        _odd_even_unpadded_selection_schedule(token_count, outputs),
    )
    return selected[:, outputs]


def pruning_network_comparator_count(
    token_count: int,
    keep_count: int,
    *,
    pruning_network: str,
    threshold_only: bool = False,
) -> int:
    """Return the public comparator count for one Top-K network invocation."""
    token_count = int(token_count)
    keep_count = int(keep_count)
    pruning_network = str(pruning_network)
    if keep_count <= 0 or keep_count > token_count:
        raise ValueError(f"keep_count must be within [1, {token_count}], got {keep_count}")
    outputs = (keep_count - 1,) if threshold_only else tuple(range(keep_count))
    if pruning_network == "odd_even_selection":
        schedule = _odd_even_unpadded_selection_schedule(token_count, outputs)
    elif pruning_network == "unpadded_selection":
        schedule = _bitonic_unpadded_selection_schedule(token_count, outputs)
    else:
        padded_count = _next_power_of_two(token_count)
        if pruning_network == "selection":
            schedule = _bitonic_selection_schedule(padded_count, outputs)
        elif pruning_network == "full_sort":
            schedule = _bitonic_full_schedule(padded_count)
        else:
            raise ValueError(f"unsupported pruning_network: {pruning_network}")
    return sum(len(layer[0]) for layer in schedule)


def exact_topk_keep_mask(
    score,
    active_mask,
    keep_count: int,
    *,
    unique_keys: bool = False,
    pruning_network: str = "full_sort",
):
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
    pruning_network = str(pruning_network)
    if pruning_network not in {
        "full_sort",
        "selection",
        "unpadded_selection",
        "odd_even_selection",
    }:
        raise ValueError(f"unsupported pruning_network: {pruning_network}")

    active = jnp.asarray(active_mask).squeeze(-1) > 0
    masked_score = jnp.where(active, score, jnp.asarray(-1.0e6, dtype=score.dtype))
    padded_count = _next_power_of_two(token_count)
    if (
        pruning_network not in {"unpadded_selection", "odd_even_selection"}
        and padded_count > token_count
    ):
        padding = jnp.full(
            (int(score.shape[0]), padded_count - token_count),
            -1.0e6,
            dtype=score.dtype,
        )
        masked_score = jnp.concatenate([masked_score, padding], axis=1)

    if pruning_network == "odd_even_selection":
        threshold = odd_even_unpadded_select_desc(masked_score, (keep_count - 1,))
    elif pruning_network == "unpadded_selection":
        threshold = bitonic_unpadded_select_desc(masked_score, (keep_count - 1,))
    elif pruning_network == "selection":
        threshold = bitonic_select_desc(masked_score, (keep_count - 1,))
    else:
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
    pruning_network: str = "full_sort",
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
    pruning_network = str(pruning_network)
    if pruning_network not in {
        "full_sort",
        "selection",
        "unpadded_selection",
        "odd_even_selection",
    }:
        raise ValueError(f"unsupported pruning_network: {pruning_network}")
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
    if (
        pruning_network not in {"unpadded_selection", "odd_even_selection"}
        and padded_count > token_count
    ):
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
    if pruning_network == "odd_even_selection":
        schedule = _odd_even_unpadded_selection_schedule(
            token_count,
            tuple(range(keep_count)),
        )
    elif pruning_network == "unpadded_selection":
        schedule = _bitonic_unpadded_selection_schedule(
            token_count,
            tuple(range(keep_count)),
        )
    elif pruning_network == "selection":
        schedule = _bitonic_selection_schedule(padded_count, tuple(range(keep_count)))
    else:
        schedule = _bitonic_full_schedule(padded_count)
    for left, right, restore, descending, untouched in schedule:
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

        gathered_keys = [left_keys, right_keys]
        gathered_tokens = [left_tokens, right_tokens]
        gathered_indices = [left_indices, right_indices]
        if untouched:
            gathered_keys.append(keys[:, untouched])
            gathered_tokens.append(tokens[:, untouched, :])
            gathered_indices.append(indices[:, untouched])
        keys = jnp.concatenate(gathered_keys, axis=1)[:, restore]
        tokens = jnp.concatenate(gathered_tokens, axis=1)[:, restore, :]
        indices = jnp.concatenate(gathered_indices, axis=1)[:, restore]

    return tokens[:, :keep_count, :], indices[:, :keep_count]
