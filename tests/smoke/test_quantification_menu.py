"""The Quantification options panel starts on the settings the validation
was measured with, and its baseline list and cascade controls stay in step
with the variables the engine and the saved projects read."""
import pytest

tk = pytest.importorskip('tkinter')

from controllers.analysis_defaults import QUANTIFICATION


@pytest.fixture
def stage():
    from tkinter import ttk
    from gui.stages.quantification_stage import QuantificationStage
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    root.withdraw()

    class _Peaks:
        peaks = [1] * 97

    class _App:
        stages = {}
    app = _App()
    app.root = root
    st = QuantificationStage(ttk.Frame(root), app, _Peaks(), _App())
    st.create_widgets()
    yield st
    root.destroy()


def test_it_starts_on_the_validated_defaults(stage):
    assert stage.shape_var.get() == QUANTIFICATION['quantify_shapes']
    assert stage.shared_var.get() == QUANTIFICATION['resolve_shared']
    assert stage.joint_fit_var.get() == QUANTIFICATION['joint_fit']
    assert stage.cascade_var.get() == QUANTIFICATION['cascade']
    assert stage.singlet_snr_var.get() == QUANTIFICATION['singlet_min_snr']
    assert stage.baseline_choice_var.get() == 'None'
    assert stage.proton_corr_var.get() is True


def test_group_a_switch_drives_its_three_variables(stage):
    assert stage.shape_windows_var.get() is True
    stage.shape_windows_var.set(False)
    assert not (stage.deconv_var.get() or stage.shape_var.get() or stage.shared_var.get())
    stage.shape_windows_var.set(True)
    assert stage.deconv_var.get() and stage.shape_var.get() and stage.shared_var.get()


def test_a_variable_set_on_its_own_sets_the_whole_group(stage):
    # a project saved before the group restores the three one by one
    stage.shared_var.set(False)
    assert stage.shape_windows_var.get() is False
    assert not (stage.deconv_var.get() or stage.shape_var.get())
    stage.deconv_var.set(True)
    assert stage.shape_windows_var.get() is True
    assert stage.shape_var.get() and stage.shared_var.get()


def test_the_baseline_list_drives_both_variables_and_back(stage):
    stage.baseline_choice_var.set('Local + model')
    assert stage.local_baseline_var.get() and stage.bl_model_quant_var.get()
    stage.local_baseline_var.set(False)      # as a project load does
    assert stage.baseline_choice_var.get() == 'Baseline model'
