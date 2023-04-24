"""
@File  : notebook_utils.py
@Author: tao.jing
@Date  :
@Desc  :
"""
import argparse
import getpass
import logging
import os
import re
import sys

__all__ = ['is_in_notebook', 'clear_output']


def is_in_notebook():
    return 'ipykernel' in sys.modules

def clear_output():
    """
    Clear output for both jupyter notebook and the console
    :return: None
    """
    os.system("cls" if os.name == "nt" else "clear")
    if is_in_notebook():
        from IPython.display import clear_output as clear
        clear()


if __name__ == '__main__':
    pass