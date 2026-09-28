"""An installed application gets the bundled databases it lacks, and keeps
every database the user already has untouched."""
import app_paths


def _frozen(monkeypatch, tmp_path, bundled):
    seed = tmp_path / 'bundle' / 'databases'
    seed.mkdir(parents=True)
    for name, text in bundled.items():
        (seed / name).write_text(text)
    user = tmp_path / 'user'
    monkeypatch.setattr(app_paths, '_is_frozen', lambda: True)
    monkeypatch.setattr(app_paths, 'user_data_dir', lambda: str(user))
    monkeypatch.setattr(app_paths, 'asset_path',
                        lambda rel: str(tmp_path / 'bundle' / rel))
    return user / 'databases'


def test_a_new_version_reaches_an_existing_installation(monkeypatch, tmp_path):
    target = _frozen(monkeypatch, tmp_path, {'v03.db': 'new', 'v04.db': 'v4'})
    target.mkdir(parents=True)
    (target / 'v03.db').write_text('edited by the user')
    app_paths.databases_dir()
    assert (target / 'v04.db').read_text() == 'v4'
    assert (target / 'v03.db').read_text() == 'edited by the user'


def test_an_empty_folder_gets_everything(monkeypatch, tmp_path):
    target = _frozen(monkeypatch, tmp_path, {'a.db': '1', 'b.csv': '2',
                                             'notes.txt': 'x'})
    app_paths.databases_dir()
    assert sorted(p.name for p in target.iterdir()) == ['a.db', 'b.csv']
