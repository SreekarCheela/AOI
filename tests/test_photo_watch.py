"""FolderWatch: new photos only, once they stop growing, temp names ignored."""
import threading

from aoi.photo.watch import FolderWatch


def test_only_new_settled_photos_in_arrival_order(tmp_path):
    (tmp_path / "before.jpg").write_bytes(b"x" * 10)
    w = FolderWatch(tmp_path, poll_s=0.01)
    sizes: dict = {}
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    (tmp_path / ".hidden.jpg").write_bytes(b"x" * 10)
    (tmp_path / "b.jpg.part").write_bytes(b"x" * 10)
    (tmp_path / "notes.txt").write_bytes(b"x" * 10)
    (tmp_path / "empty.png").write_bytes(b"")
    assert w.ready(sizes) == []                     # first sight: size not yet known to hold
    (tmp_path / "c.png").write_bytes(b"x" * 5)
    assert [f.name for f in w.ready(sizes)] == ["a.jpg"]
    (tmp_path / "c.png").write_bytes(b"x" * 9)      # still being written
    assert w.ready(sizes) == [tmp_path / "a.jpg"]   # (not marked seen: the iterator does that)
    assert [f.name for f in w.ready(sizes)] == ["a.jpg", "c.png"]


def test_iterator_yields_then_stops(tmp_path):
    stop = threading.Event()
    got = []
    w = FolderWatch(tmp_path, stop, poll_s=0.01, on_new=got.append)
    (tmp_path / "a.jpg").write_bytes(b"x" * 10)
    it = iter(w)
    assert next(it).name == "a.jpg" and [f.name for f in got] == ["a.jpg"]
    stop.set()
    assert list(it) == []
