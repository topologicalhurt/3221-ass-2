import argparse
import json
import socket
from contextlib import closing

from conf import Conf

class Utils:

    @staticmethod
    def __read_conf() -> Conf:
        with open('net_conf.json', 'r') as f:
            data = json.load(f)
        return Conf(**data)

    NET_CONF = __read_conf()

    @staticmethod
    def thread_spawner(func):
        def wrapper(*args, **kwargs):
            threads = []
            func(threads, *args, **kwargs)
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        return wrapper

    @staticmethod
    def verify_flag(arg: str) -> bool:
        arg = arg.lower()
        if arg not in {'true', 'false', '1', '0'}:
            raise argparse.ArgumentTypeError('A flag must be one of: [true, false, 1, 0]')

        match arg:
            case 'true':
                return True
            case 'false':
                return False
            case '1':
                return True
            case '0':
                return False

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


