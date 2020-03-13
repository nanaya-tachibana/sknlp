import os
import shutil
from sknlp.utils import make_tarball


def test_make_tarball(tmp_path):
    test_dir = tmp_path / 'tarball'
    test_dir.mkdir()
    for filename in ('a_tar_file', 'b_tar_file'):
        test_file = test_dir / filename
        test_file.touch()
    tarball_file = tmp_path / 'foo.tar'
    make_tarball(str(tarball_file), str(test_dir))
    shutil.unpack_archive(str(tarball_file), str(tmp_path), format='tar')
    tmp_files = set(os.listdir(str(tmp_path)))
    assert 'a_tar_file' in tmp_files
    assert 'b_tar_file' in tmp_files
