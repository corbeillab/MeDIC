import pytest
import os
from ...metabodashboard.service import Utils


def test_givenUtils_whenGetFilePath_thenReturnFilePath():
    assert Utils.DUMP_EXPE_PATH == os.path.join(
        os.sep.join(os.path.dirname(__file__).split(os.sep)[:-2]),
        "metabodashboard",
        "domain",
        "dumps",
        "save.mtxp",
    )


def test_givenBinaryClass_whenGettingBinary_thenBinaryIsReturned():
    assert Utils.get_binary(["b", "c", "b", "c"], ["b", "c"]) == [0, 1, 0, 1]


def test_givenMultiClass_whenGettingBinary_thenBinaryIsReturned():
    assert Utils.get_binary(["b", "c", "b", "c", "a", "c"], ["a", "b", "c"]) == [
        1,
        2,
        1,
        2,
        0,
        2,
    ]
