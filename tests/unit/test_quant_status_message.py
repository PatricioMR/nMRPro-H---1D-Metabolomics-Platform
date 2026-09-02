"""The count under the table must say what it counted.

_quant_results holds every analyte that was considered, quantified or not, so
its length is not the number of results. Reporting it as "100 metabolites" read
as 100 measurements when 65 were measured, 15 of those over a window the
database predicted rather than a peak the spectrum showed.
"""
import inspect

from gui.stages import _quant_computation as qc


def _msg_src():
    src = inspect.getsource(qc)
    i = src.index('_n_all = len(self._quant_results)')
    return src[i:src.index('self._status_ok(_msg_s)', i)]


def test_the_message_reports_what_was_quantified_not_the_row_count():
    src = _msg_src()
    assert '_n_ok_s = _n_all - _n_nq_s' in src
    assert 'identified and ' in src and 'quantified successfully' in src


def test_it_agrees_with_itself_on_one_metabolite():
    src = _msg_src()
    assert "_plural = '' if _n_ok_s == 1 else 's'" in src


def test_the_not_quantified_are_counted_separately():
    src = _msg_src()
    assert "_r_s.get('_not_quantified')" in src
    assert 'not quantified' in src


def test_analytes_with_no_detected_peak_are_called_out():
    """A number over a database-predicted window is a real number resting on
    weaker evidence, and the difference belongs where the count is read."""
    src = _msg_src()
    assert "_r_s.get('_synthetic_db')" in src
    assert 'without a detected peak' in src


def test_that_flag_reaches_the_row_it_describes():
    """It was computed per metabolite and left in meta_info, so nothing
    downstream — table, export or count — could tell the two apart."""
    src = inspect.getsource(qc)
    assert "'_synthetic_db':     bool(m.get('_synthetic_db', False))," in src


def test_a_row_without_the_flag_is_not_counted_as_one():
    src = _msg_src()
    i = src.index("_r_s.get('_synthetic_db')")
    assert "not _r_s.get('_not_quantified')" in src[i:i + 200]
