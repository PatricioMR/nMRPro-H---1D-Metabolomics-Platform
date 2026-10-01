"""Every quantification setting the project saves is restored under the
same key, and to the same variable. A checkbox saved and never restored
comes back at its default, silently, on every reload."""
import pathlib
import re

SRC = (pathlib.Path(__file__).resolve().parents[2] / 'gui'
       / 'project_manager.py').read_text(encoding='utf-8')


def test_saved_quant_variables_are_restored_by_the_same_key():
    saved = dict(re.findall(
        r"'(\w+)':\s*_safe_var\(qnt, '(\w+_var)'\)", SRC))
    restored = {k: v for v, k in re.findall(
        r"_set_var\(qnt, '(\w+_var)',\s*qnt_state\.get\('(\w+)'\)\)", SRC)}
    assert saved, 'no quantification variables found'
    # 'integ' is restored through a migration of its old 'Lorentz' value
    restored.setdefault('integ', 'integ_var')
    missing = {k: v for k, v in saved.items() if restored.get(k) != v}
    assert not missing, f'saved but not restored to the same variable: {missing}'
    for key in ('shape_fit', 'shared_deconv', 'j_resolve', 'singlet_min_snr',
                'cascade', 'max_disagreement', 'joint_fit'):
        assert key in saved and key in restored


def test_alignment_saved_and_restored_by_the_same_key():
    # all_intensities is saved aligned once detection aligns it; the shifts
    # that made it so must come back with it, or the alignment is no longer
    # auditable after a reload.
    saved = dict(re.findall(r"'(\w+)':\s*getattr\(app, '(\w+)'", SRC))
    restored = dict(re.findall(r"app\.(\w+)\s*=\s*state\.get\('(\w+)'", SRC))
    restored = {k: a for a, k in restored.items()}
    for key in ('alignment_corrected', 'alignment_shifts_ppm'):
        assert saved.get(key) == key and restored.get(key) == key
