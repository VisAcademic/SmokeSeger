"""
@File  : paddle_utils.py
@Author: tao.jing
@Date  :
@Desc  :
"""
import argparse
import getpass
import logging
import os
import re
import subprocess


__all__ = [
    'on_ai_studio',
    'on_windows'
]


def get_python_path():
    # Get conda python dir
    python_path = None
    command_str = 'which python'
    ex_handler = subprocess.Popen(command_str, stdout=subprocess.PIPE, shell=True)
    out, err = ex_handler.communicate()
    status = ex_handler.wait()
    assert status == 0, \
        f'[disable_dataloader_warnings] Invalid command {command_str}'

    lines = out.decode().split()
    for line in lines:
        if 'paddle' in line and \
                'envs' in line and \
                'python' in line:
            if os.path.exists(line):
                if line.endswith('bin/python'):
                    python_path = line
                    break

    if python_path is not None:
        logging.info(f'[disable_dataloader_warnings] '
                     f'Found python in {python_path}')
    return python_path.rstrip('bin/python')


def get_python_version():
    # Get python version
    python_str = None
    command_str = 'python --version'
    ex_handler = subprocess.Popen(command_str, stdout=subprocess.PIPE, shell=True)
    out, err = ex_handler.communicate()
    status = ex_handler.wait()
    assert status == 0, \
        f'[disable_dataloader_warnings] Invalid command {command_str}'

    match_objs = re.match(r'Python (\d).(\d).(\d)', out.decode())
    if match_objs:
        major_ver = match_objs.group(1)
        minor_var = match_objs.group(2)
        python_str = f'{major_ver}.{minor_var}'
        logging.info(f'[disable_dataloader_warnings] Python version: {python_str}')
    else:
        python_str = '3.7'
        logging.info(f'[disable_dataloader_warnings] No valid python version')

    return python_str


def get_fluid_path():
    python_path = get_python_path()
    assert os.path.isdir(python_path), \
        f'[disable_dataloader_warnings] Invalid python path {python_path}'

    python_str = get_python_version()
    assert isinstance(python_str, str), \
        f'[disable_dataloader_warnings] Invalid python version {python_str}'

    fluid_path = os.path.join(python_path,
                              'lib',
                              'python' + python_str,
                              'site-packages',
                              'paddle',
                              'fluid')
    return fluid_path


def add_disable_warnings_in_func(target_file_path, target_func_name):
    assert os.path.exists(target_file_path), \
        f'[add_disable_warnings_in_func] Target python file not exists.'

    lines = list()
    need_insert = False
    with open(target_file_path, 'r') as f:
        lines = f.readlines()
        for cur_line_idx, line in enumerate(lines):
            if target_func_name in line:
                comment_idx = 0
                for idx in range(cur_line_idx, len(lines)):
                    if '"""' in lines[idx]:
                        comment_idx += 1
                    if comment_idx == 2:
                        # The ending of the comments
                        if 'warnings' not in lines[idx + 1]:
                            need_insert = True
                            lines.insert(idx + 1, '    import warnings\n')
                            lines.insert(idx + 2,
                                         "    warnings.filterwarnings('ignore', category=DeprecationWarning)\n")
                        break

    if need_insert:
        with open(target_file_path, 'w') as f:
            f.writelines(lines)


def disable_dataloader_warnings():
    assert_on_ai_studio('disable_dataloader_warnings')

    # target_file_path = os.path.join('files', 'dataloader_iter.tmp')
    target_file_path = os.path.join(get_fluid_path(), 'dataloader', 'dataloader_iter.py')

    assert os.path.exists(target_file_path), \
        f'[disable_dataloader_warnings] Target python file not exists.'

    add_disable_warnings_in_func(target_file_path, 'default_collate_fn')


def disable_utils_warnings():
    assert_on_ai_studio('disable_utils_warnings')

    # target_file_path = os.path.join('files', 'utils.tmp')
    target_file_path = os.path.join(get_fluid_path(), 'layers', 'utils.py')

    assert os.path.exists(target_file_path), \
        f'[disable_utils_warnings] Target python file not exists.'

    add_disable_warnings_in_func(target_file_path, 'convert_to_list')


def assert_on_ai_studio(util_str):
    cur_user = getpass.getuser()
    assert cur_user == 'aistudio', \
        f'This util {util_str} should be used on AI Studio'


def assert_on_local_host(util_str):
    cur_user = getpass.getuser()
    assert cur_user != 'aistudio', \
        f'This util {util_str} should be used on local host'


def on_ai_studio():
    cur_user = getpass.getuser()
    return cur_user == 'aistudio'


def on_windows():
    import platform
    return platform.system() == 'Windows'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run', type=str, default='none')

    return parser.parse_args()


if __name__ == '__main__':
    # Logging configure
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()

    if args.run == 'none':
        logging.info('Not set target util.')
    elif args.run == 'disable_dataloader_warnings':
        disable_dataloader_warnings()
    elif args.run == 'disable_utils_warnings':
        disable_utils_warnings()
