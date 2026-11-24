"""Temporary: run the app printing every Python callback Tk invokes, so the
last line before a native crash names the callback that preceded it."""
import runpy
import sys
import tkinter

_orig = tkinter.CallWrapper.__call__


def _name(func):
    q = getattr(func, '__qualname__', repr(func))
    if q.endswith('callit'):        # after(): name the function it wraps
        for cell in func.__closure__ or ():
            c = cell.cell_contents
            if callable(c) and not isinstance(c, (str, bytes)):
                return 'after:' + _name(c)
    return f"{getattr(func, '__module__', '')}.{q}"


def _traced(self, *args):
    f = _name(self.func)
    m = ''
    print(f'[tk->py] {m}.{f}', flush=True)
    r = _orig(self, *args)
    print(f'[tk<-py] {m}.{f}', flush=True)
    return r


tkinter.CallWrapper.__call__ = _traced
sys.argv = ['main.py']
runpy.run_path('main.py', run_name='__main__')
