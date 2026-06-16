from __future__ import annotations

import numpy as np


def greedy_decode(similarity: np.ndarray, **_: object) -> np.ndarray:
    return np.argmax(similarity, axis=1).astype(np.int64)


def transition_costs(
    states: int,
    max_step: int = 3,
    stay_penalty: float = 0.15,
    jump_penalty: float = 0.4,
    backward_penalty: float = 1.0,
    reference_positions: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.arange(states, dtype=np.float32) if reference_positions is None else np.asarray(reference_positions, dtype=np.float32)
    if positions.shape != (states,):
        raise ValueError("reference_positions must contain one value per Street View image")
    positive_steps = np.diff(positions)
    positive_steps = positive_steps[positive_steps > 1e-6]
    unit = float(np.median(positive_steps)) if positive_steps.size else 1.0
    previous = positions[:, None]
    current = positions[None, :]
    delta = current - previous
    normalized_delta = delta / max(unit, 1e-6)
    costs = np.maximum(np.abs(normalized_delta) - max_step, 0) * jump_penalty
    costs = costs + (np.abs(delta) < 1e-6) * stay_penalty
    costs = costs + (delta < 0) * backward_penalty * np.abs(normalized_delta)
    return costs.astype(np.float32)


def viterbi_decode(
    similarity: np.ndarray,
    max_step: int = 3,
    stay_penalty: float = 0.15,
    jump_penalty: float = 0.4,
    backward_penalty: float = 1.0,
    emission_temperature: float = 0.1,
    reference_positions: np.ndarray | None = None,
    **_: object,
) -> np.ndarray:
    if similarity.ndim != 2 or not similarity.size:
        raise ValueError("Similarity must be a non-empty 2D matrix")
    frames, states = similarity.shape
    transition = transition_costs(
        states, max_step, stay_penalty, jump_penalty, backward_penalty, reference_positions
    )
    emissions = similarity / max(emission_temperature, 1e-6)
    dp = np.empty((frames, states), dtype=np.float32)
    back = np.zeros((frames, states), dtype=np.int32)
    dp[0] = emissions[0]
    for t in range(1, frames):
        candidates = dp[t - 1][:, None] - transition
        back[t] = np.argmax(candidates, axis=0)
        dp[t] = emissions[t] + candidates[back[t], np.arange(states)]
    path = np.empty(frames, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def route_viterbi_decode(
    similarity: np.ndarray,
    max_step: int = 3,
    stay_penalty: float = 0.15,
    emission_temperature: float = 0.1,
    reference_positions: np.ndarray | None = None,
    progress_weight: float = 0.28,
    progress_sigma_fraction: float = 0.20,
    speed_penalty: float = 0.18,
    allowed_states: np.ndarray | None = None,
    **_: object,
) -> np.ndarray:
    """Monotonic sequence alignment for a video traversing the ordered reference route."""
    if similarity.ndim != 2 or not similarity.size:
        raise ValueError("Similarity must be a non-empty 2D matrix")
    frames, states = similarity.shape
    if reference_positions is None:
        positions = np.arange(states, dtype=np.float32)
    else:
        positions = np.asarray(reference_positions, dtype=np.float32)
        if positions.shape != (states,):
            raise ValueError("reference_positions must contain one value per Street View image")
    allowed = np.ones(states, dtype=bool) if allowed_states is None else np.asarray(allowed_states, dtype=bool)
    if allowed.shape != (states,) or not np.any(allowed):
        raise ValueError("allowed_states must enable at least one Street View image")
    route_start = float(positions[allowed][0])
    route_end = float(positions[allowed][-1])
    route_length = max(route_end - route_start, 1e-6)
    expected_positions = np.linspace(route_start, route_end, frames, dtype=np.float32)
    sigma = max(route_length * progress_sigma_fraction, 1e-6)
    progress_prior = np.exp(-0.5 * ((positions[None, :] - expected_positions[:, None]) / sigma) ** 2)
    weight = min(max(progress_weight, 0.0), 1.0)
    emissions = ((1.0 - weight) * similarity + weight * progress_prior) / max(emission_temperature, 1e-6)
    emissions[:, ~allowed] = -1e9

    steps = np.diff(positions)
    positive_steps = steps[steps > 1e-6]
    unit = float(np.median(positive_steps)) if positive_steps.size else 1.0
    expected_step = route_length / max(frames - 1, 1) / unit
    previous = positions[:, None]
    current = positions[None, :]
    normalized_delta = (current - previous) / max(unit, 1e-6)
    valid = (normalized_delta >= -1e-6) & (normalized_delta <= max_step)
    valid &= allowed[:, None] & allowed[None, :]
    transition = -speed_penalty * np.abs(normalized_delta - expected_step)
    transition -= stay_penalty * (np.abs(normalized_delta) < 1e-6)
    transition /= max(emission_temperature, 1e-6)
    transition = np.where(valid, transition, -1e9).astype(np.float32)

    dp = np.empty((frames, states), dtype=np.float32)
    back = np.zeros((frames, states), dtype=np.int32)
    dp[0] = emissions[0]
    for frame in range(1, frames):
        candidates = dp[frame - 1][:, None] + transition
        back[frame] = np.argmax(candidates, axis=0)
        dp[frame] = emissions[frame] + candidates[back[frame], np.arange(states)]
    path = np.empty(frames, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for frame in range(frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def local_window_search_decode(
    similarity: np.ndarray,
    initial_search_size: int = 5,
    initial_evidence_frames: int = 3,
    initial_index_penalty: float = 0.04,
    window_backward: int = 0,
    window_forward: int = 5,
    max_window_distance_m: float = 30.0,
    temporal_smoothing: float = 0.65,
    local_stay_penalty: float = 0.01,
    reference_positions: np.ndarray | None = None,
    allowed_states: np.ndarray | None = None,
    **_: object,
) -> np.ndarray:
    """LWS: causal matching in a small window around the previous localization."""
    if similarity.ndim != 2 or not similarity.size:
        raise ValueError("Similarity must be a non-empty 2D matrix")
    frames, states = similarity.shape
    allowed = np.ones(states, dtype=bool) if allowed_states is None else np.asarray(allowed_states, dtype=bool)
    if allowed.shape != (states,) or not np.any(allowed):
        raise ValueError("allowed_states must enable at least one Street View image")
    positions = None if reference_positions is None else np.asarray(reference_positions, dtype=np.float32)
    if positions is not None and positions.shape != (states,):
        raise ValueError("reference_positions must contain one value per Street View image")
    alpha = min(max(float(temporal_smoothing), 0.0), 1.0)
    smoothed = np.empty_like(similarity, dtype=np.float32)
    smoothed[0] = similarity[0]
    for frame in range(1, frames):
        smoothed[frame] = alpha * similarity[frame] + (1.0 - alpha) * smoothed[frame - 1]

    path = np.empty(frames, dtype=np.int64)
    initial_limit = min(max(int(initial_search_size), 1), states)
    initial_candidates = np.flatnonzero(allowed[:initial_limit])
    if not len(initial_candidates):
        initial_candidates = np.flatnonzero(allowed)[:initial_limit]
    evidence_frames = min(max(int(initial_evidence_frames), 1), frames)
    initial_scores = np.mean(similarity[:evidence_frames, initial_candidates], axis=0)
    initial_scores -= max(float(initial_index_penalty), 0.0) * np.arange(len(initial_candidates))
    path[0] = initial_candidates[int(np.argmax(initial_scores))]

    for frame in range(1, frames):
        previous = int(path[frame - 1])
        start = max(0, previous - max(int(window_backward), 0))
        stop = min(states, previous + max(int(window_forward), 0) + 1)
        candidates = np.arange(start, stop, dtype=np.int64)
        candidates = candidates[allowed[candidates]]
        if positions is not None and max_window_distance_m > 0:
            candidates = candidates[
                np.abs(positions[candidates] - positions[previous]) <= max_window_distance_m
            ]
        if not len(candidates):
            candidates = np.asarray([previous], dtype=np.int64)
        local_scores = smoothed[frame, candidates].copy()
        local_scores[candidates == previous] -= max(float(local_stay_penalty), 0.0)
        path[frame] = candidates[int(np.argmax(local_scores))]
    return path


def local_search_windows(
    path: np.ndarray,
    states: int,
    initial_search_size: int = 5,
    window_backward: int = 0,
    window_forward: int = 5,
    reference_positions: np.ndarray | None = None,
    max_window_distance_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive search-window bounds used by LWS for audit and visualization."""
    starts = np.zeros(len(path), dtype=np.int64)
    stops = np.empty(len(path), dtype=np.int64)
    stops[0] = min(max(int(initial_search_size), 1), states) - 1
    for frame in range(1, len(path)):
        previous = int(path[frame - 1])
        starts[frame] = max(0, previous - max(int(window_backward), 0))
        stops[frame] = min(states - 1, previous + max(int(window_forward), 0))
        if reference_positions is not None and max_window_distance_m > 0:
            positions = np.asarray(reference_positions, dtype=np.float32)
            while stops[frame] > starts[frame] and positions[stops[frame]] - positions[previous] > max_window_distance_m:
                stops[frame] -= 1
            while starts[frame] < stops[frame] and positions[previous] - positions[starts[frame]] > max_window_distance_m:
                starts[frame] += 1
    return starts, stops


def decode_trajectory(similarity: np.ndarray, algorithm: str = "viterbi", **kwargs: object) -> np.ndarray:
    decoders = {
        "local_window_search": local_window_search_decode,
        "route_viterbi": route_viterbi_decode,
        "viterbi": viterbi_decode,
        "dynamic_programming": viterbi_decode,
        "greedy": greedy_decode,
    }
    try:
        return decoders[algorithm](similarity, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown trajectory algorithm: {algorithm}") from exc
