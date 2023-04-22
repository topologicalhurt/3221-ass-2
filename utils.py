import argparse
import json
import socket
from contextlib import closing
import threading
import os
from enum import Enum

from conf import Conf


class Utils:

    # TODO: move these to a separate constants file

    POSIX = os.name != 'nt'

    @staticmethod
    def __read_conf() -> Conf:
        with open('net_conf.json', 'r') as f:
            data = json.load(f)
        return Conf(**data)

    NET_CONF = __read_conf()

    ClientComms = Enum('Comms', {'RESUME_SERVER': b'1',
                                 'RESUME_CLIENT': b'2',
                                 'DISPATCH': b'3'})

    @staticmethod
    def thread_spawner(func):
        def wrapper(*args, **kwargs):
            if 'threads' not in kwargs or not hasattr(kwargs['threads'], '__iter__'):
                raise ValueError('Must provide thread accumulator')
            func(*args, **kwargs)
            for t in kwargs['threads']:
                t.start()
            for t in kwargs['threads']:
                t.join()

        return wrapper

    @staticmethod
    def verify_flag(arg: str) -> bool:
        arg = arg.lower()
        match arg:
            case 'true' | '1':
                return True
            case 'false' | '0':
                return False
            case _:
                raise argparse.ArgumentTypeError('A flag must be one of: [true, false, 1, 0]')

    @staticmethod
    def port_no_valid(arg: str) -> int:
        if not all(c.isdigit() for c in arg):
            raise argparse.ArgumentTypeError('Must be an integer')
        arg = int(arg)

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if not sock.connect_ex((Utils.NET_CONF.host, arg)) != 0:
                raise argparse.ArgumentTypeError('The port must not already be open!')

        if __debug__:
            if arg > 0xffff or arg < 6000:
                raise argparse.ArgumentTypeError('Must be in range [6000, 0xffff]')
        else:
            if arg != Utils.NET_CONF.port:
                raise argparse.ArgumentTypeError('Port number must be 6000 in production mode')

        return arg


class ThreadSafeCounter:

    LOCK = threading.Lock()

    def __init__(self, count: int):
        self.count = count

    def __add__(self, other: int):
        with ThreadSafeCounter.LOCK:
            return self.count + other

    def __sub__(self, other: int):
        with ThreadSafeCounter.LOCK:
            return self.count - other

    def __int__(self):
        return self.count


