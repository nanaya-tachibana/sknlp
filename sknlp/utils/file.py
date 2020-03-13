import os
import tarfile


def make_tarball(filename, root_dir):
    """
    将`root_dir`下的文件打包为一个tar文件

    filename: 打包后的文件名, 如果结尾不为.tar, 打包后的文件名为`filename`.tar
    root_dir: 需要打包的文件所在的目录
    """
    if not filename.endswith('.tar'):
        filename = '.'.join([filename, 'tar'])
    with tarfile.open(filename, 'w') as f:
        for file in os.listdir(root_dir):
            f.add(os.path.join(root_dir, file), arcname=file)
