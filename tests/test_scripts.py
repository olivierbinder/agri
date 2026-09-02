# %% IMPORTS

import json

import pytest
from _pytest import capture as pc

from agri import scripts

# %% SCRIPTS


def test_schema(capsys: pc.CaptureFixture[str]) -> None:
    # given
    args = ["prog", "--schema"]
    # when
    status = scripts.main(args)
    captured = capsys.readouterr()
    # then
    assert status == 0, "Schema mode should exit successfully!"
    assert captured.err == "", "Captured error should be empty!"
    assert json.loads(captured.out), "Captured output should be a JSON schema!"


def test_main__no_configs() -> None:
    # given
    argv: list[str] = []
    # when
    with pytest.raises(RuntimeError) as error:
        scripts.main(argv)
    # then
    assert error.match("No configs provided."), "RuntimeError should be raised!"
