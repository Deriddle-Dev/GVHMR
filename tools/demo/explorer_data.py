"""Prepare GVHMR prediction tensors for parameter exploration."""

import numpy as np


def collect_arrays(prediction):
    """Return numeric leaves with their frame axis normalized and frame count."""
    try:
        frame_count = int(prediction["smpl_params_global"]["body_pose"].shape[0])
    except (KeyError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Prediction is missing global body pose frames.") from exc
    if frame_count < 1:
        raise ValueError("Prediction contains no frames.")

    arrays = {}

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif not isinstance(value, (np.ndarray, int, float, bool)):
            return
        value = np.asarray(value)
        if value.dtype.kind not in "biuf" or value.size == 0:
            return
        # Network outputs retain the batch dimension; top-level SMPL parameters do not.
        batched = path.startswith("net_outputs.") and value.ndim >= 2 and value.shape[:2] == (1, frame_count)
        if batched:
            value = value[0]
        top_level_time = path.startswith(("smpl_params_global.", "smpl_params_incam.")) or path == "K_fullimg"
        timed = batched or (top_level_time and value.ndim > 0 and value.shape[0] == frame_count)
        arrays[path] = (value, timed)

    visit(prediction, "")
    return arrays, frame_count


def component_series(array, timed, component):
    """Return one time series, or the flattened values for a static array."""
    if timed:
        values = array.reshape(array.shape[0], -1)
        if not 0 <= component < values.shape[1]:
            raise IndexError("Component out of range.")
        return values[:, component]
    return array.reshape(-1)
