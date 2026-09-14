"""T022 raw within-sample response in frozen non-clean state order."""
import numpy as np


def state_response(outputs):
    # outputs: [five frozen states, samples, feature dimensions]
    return np.concatenate([outputs[k]-outputs[0] for k in range(1,5)],axis=-1)
