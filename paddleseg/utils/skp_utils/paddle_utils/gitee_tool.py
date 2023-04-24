"""
@File  : gitee_tool.py
@Author: tao.jing
@Date  :
@Desc  : Run git commands on ai studio
"""
import argparse
import getpass
import logging
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--work-dir', type=str, default='none')
    parser.add_argument('--command', type=str, default='none')
    parser.add_argument('--url', type=str, default='none')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    cur_user = getpass.getuser()
    assert cur_user == 'aistudio', \
        f'Git tool is used for AI Studio, not local host!'

    args = get_args()

    # Change work directory
    os.chdir('work')
    assert os.getcwd() == '/home/aistudio/work', \
        f'Wrong work directory {os.getcwd()}'

    if args.work_dir != 'none':
        os.chdir(args.work_dir)
        assert os.getcwd() == os.path.join('/home/aistudio/work', args.work_dir), \
            f'Wrong work directory {os.getcwd()}'

    command = 'None'
    if args.command == 'none':
        logging.error(f'No git command set.')
        assert False
    elif args.command == 'update':
        # Git pull new commits
        command = 'git pull'
    elif args.command == 'clone':
        # Clone new repository
        assert args.url != 'none', \
            f'Please set clone url!'
        command = f'git clone {args.url} --depth=1 --recursive'
    elif args.command == 'restore':
        # Abandon local changes
        command = 'git checkout -- *'

    logging.info(f'Run command: {command}')
    try:
        os.system(command)
    except Exception as e:
        logging.error(e)
    else:
        logging.info(f'Run successfully.')
