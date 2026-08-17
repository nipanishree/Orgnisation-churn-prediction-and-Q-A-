import sys


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_sanity():
    assert 1 + 1 == 2
