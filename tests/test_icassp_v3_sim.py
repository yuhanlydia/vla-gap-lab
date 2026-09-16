import importlib

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module('vla_gap_lab.icassp_v3_sim')
    except ModuleNotFoundError:
        pytest.fail('physics-substep contact annotation is missing')


def test_contact_latches_transient_substep_and_restores_original_hook():
    from types import SimpleNamespace
    sim = module()
    state = {'force': 0., 'original_calls': 0}
    def original():
        state['original_calls'] += 1
    base = SimpleNamespace(
        agent=SimpleNamespace(robot=SimpleNamespace(get_links=lambda: ['link'])),
        ball='ball',
        scene=SimpleNamespace(get_pairwise_contact_forces=lambda a, b: np.array([[state['force'], 0., 0.]])),
        _after_simulation_step=original,
    )
    with sim.ContactRecorder(base, threshold=.001) as recorder:
        state['force'] = 3.
        base._after_simulation_step()
        state['force'] = 0.
        base._after_simulation_step()
        assert recorder.seen is True
        assert recorder.peak_force == 3.
        assert recorder.samples == 2
    assert base._after_simulation_step is original
    assert state['original_calls'] == 2


def test_missing_contact_api_is_an_error_not_all_zero_labels():
    from types import SimpleNamespace
    sim = module()
    base = SimpleNamespace(agent=SimpleNamespace(robot=SimpleNamespace(get_links=lambda: [])),
                           scene=SimpleNamespace(), _after_simulation_step=lambda: None)
    with pytest.raises((ValueError, RuntimeError), match='contact'):
        sim.ContactRecorder(base)
