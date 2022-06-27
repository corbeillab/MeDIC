import pytest
import os
from ...metabodashboard.service import Utils


def test_givenUtils_whenGetFilePath_thenReturnFilePath():
    assert Utils.DUMP_EXPE_PATH == os.path.join(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-2]), "metabodashboard", "domain", "dumps", "save.mtxp")