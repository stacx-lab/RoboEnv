"""Transient MJCF files never write into the shared read-only asset depot."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree


@contextmanager
def writable_object_xml(xml, source_directory):
    root = ElementTree.fromstring(xml)
    # Match MujocoXML.resolve_asset_dependency while preserving the original
    # asset base directory. The temporary XML may live on another filesystem.
    for element in root.findall("./asset/*[@file]"):
        source = Path(element.get("file"))
        if not source.is_absolute():
            element.set("file", str((Path(source_directory) / source).resolve()))
    with TemporaryDirectory(prefix="roboenv-object-") as temporary:
        path = Path(temporary) / "model.xml"
        path.write_text(ElementTree.tostring(root, encoding="unicode"))
        yield str(path)
