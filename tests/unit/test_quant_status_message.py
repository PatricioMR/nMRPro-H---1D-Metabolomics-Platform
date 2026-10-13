"""What the run reports it did.

The results list holds every analyte that was considered, quantified or not, so
its length is not the number of results. Reporting it as "100 metabolites" read
as 100 measurements when a third of the rows existed to explain why there was
no measurement — and two labels counting the same list two different ways is
how one of them kept saying 100 after the other had been taught to say 65.
"""
import inspect

from gui.stages import _quant_computation as qc


def _finish_ui_src():
    src = inspect.getsource(qc)
    i = src.index('def _finish_ui():')
    return src[i:src.index('self.app.root.after(0, _finish_ui)', i)]


def test_the_message_reports_what_was_quantified_not_the_row_count():
    src = _finish_ui_src()
    assert "_r.get('_not_quantified')" in src
    assert '_n_ok = len(_analytes) - _n_nq' in src


def test_the_standard_is_not_counted_as_a_quantified_analyte():
    """The report leaves the internal standard's row out; the stage said 70
    where the report said 69."""
    src = _finish_ui_src()
    assert "if not _r.get('_is_is')" in src


def test_both_labels_read_the_same_count():
    """Counted once, used twice. Two counts of the same thing drift."""
    src = _finish_ui_src()
    assert src.count('_n_nq =') == 1
    assert src.count('_n_ok = ') == 1
    i_run = src.index('self.run_status.config')
    i_msg = src.index('self._status_ok(')
    assert '_n_ok' in src[i_run:i_run + 200]
    assert '_n_ok' in src[:i_msg]


def test_the_run_label_separates_quantified_from_not():
    src = _finish_ui_src()
    assert 'quantified' in src
    assert '_n_nq} not' in src


def test_the_not_quantified_are_counted_separately_in_the_message():
    """They are part of what the run did, and hiding them makes the run look
    cleaner than it was."""
    src = _finish_ui_src()
    assert 'not quantified' in src


def test_it_agrees_with_itself_on_one_metabolite():
    """Plural handled, because "1 metabolites" reads as a bug in the count."""
    src = _finish_ui_src()
    assert "_plural = '' if _n_ok == 1 else 's'" in src


def test_the_message_is_in_english_like_the_rest_of_the_interface():
    src = _finish_ui_src()
    assert 'identified and' in src
    assert 'quantified successfully' in src


def test_the_message_counts_results_and_refusals_only():
    """Quantified or not: there is no third kind of row to count."""
    src = _finish_ui_src()
    assert '_n_cens' not in src
    assert 'limit of quantification' not in src


def test_the_reasons_are_in_english_like_the_interface():
    from processing.quantification import REJECTION_TEXT, WINDOW_INTEGRAL_TEXT
    for text in list(REJECTION_TEXT.values()) + [WINDOW_INTEGRAL_TEXT]:
        assert 'ã' not in text and 'ç' not in text, text
