"""The fuller tubes are the reference study with more in them, and nothing
else changed."""
import pytest

from validation import mixture_bench as mb
from validation.crowded_bench import STANDARD_CLEARANCE_PPM, panel


def test_the_reference_eighteen_are_in_every_tube():
    names = {n for n, _c, _f in mb.PANEL}
    for n in (18, 50, 100):
        got = panel(mb.DB_DEFAULT, n)
        assert len(got) == n
        assert names <= {nm for nm, _c, _f in got}
        # and at the amounts the reference study put them in at
        assert got[:18] == list(mb.PANEL)


def test_the_tubes_nest():
    """The 100-analyte tube is the 50-analyte tube with fifty more, each at
    the amount it had there -- so the difference between them is only what
    was added."""
    assert set(panel(mb.DB_DEFAULT, 50)) <= set(panel(mb.DB_DEFAULT, 100))


def test_no_analyte_is_added_twice():
    got = panel(mb.DB_DEFAULT, 100)
    assert len({n for n, _c, _f in got}) == len(got)


def test_nothing_added_sits_on_the_internal_standard():
    import sqlite3
    con = sqlite3.connect(mb.DB_DEFAULT)
    for name, _c, _f in panel(mb.DB_DEFAULT, 100)[18:]:
        bid = con.execute('select bmrb_id from metabolites where name=?',
                          (name,)).fetchone()[0]
        for (p,) in con.execute('select coalesce(ppm_meas, ppm) from shifts_1h '
                                'where bmrb_id=?', (bid,)):
            if p is not None:
                assert abs(float(p) - mb.STANDARD_PPM) >= STANDARD_CLEARANCE_PPM


def test_the_amounts_added_are_in_the_stated_range():
    from validation.crowded_bench import CONC_RANGE_MM
    for _n, c, f in panel(mb.DB_DEFAULT, 100)[18:]:
        assert CONC_RANGE_MM[0] <= c <= CONC_RANGE_MM[1] + 1e-9
        assert f == 1.0


def test_the_noise_does_not_move_with_the_panel():
    """Anchored to lactate's doublet: a fuller tube, not another
    spectrometer."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        _x, s18, _t, _f = mb.build_study(mb.DB_DEFAULT, seed=0)
        _x, s50, _t, _f = mb.build_study(mb.DB_DEFAULT, seed=0,
                                         panel=panel(mb.DB_DEFAULT, 50))
    assert s50 == pytest.approx(s18)
