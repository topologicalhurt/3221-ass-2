import argparse
import threading
import sys
import subprocess

import psutil
import os


from utils import Utils


def start_flclient(cid: int, opt_method: bool):
    opt_method = '1' if opt_method else '0'
    proc = subprocess.Popen(
        [sys.executable, 'Comp3221_FLClient.py', f'client{cid}', str(Utils.NET_CONF.port + cid), opt_method],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    proc.stdout.write(int.to_bytes(os.getpid(), byteorder='big', length=4))
    print(proc.stdout.readline())
    return proc


# @Utils.thread_spawner
# def spawn_clients(threads: list):
#     for i in range(Utils.NET_CONF.n_clients):
#         threads.append(threading.Thread(target=start_flclient, args=[i, True]))

def spawn_clients():
    clients = [p for p in [start_flclient(i, True) for i in range(Utils.NET_CONF.n_clients)]]

    # ppid = psutil.Process(os.getpid())
    # print(len(ppid.children()))
    # print(clients)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Comp3221_FLServer.py',
        description='Finds the federated average on MNIST hand-writing problem amongst 5 clients.')

    port_no = Utils.NET_CONF.port

    if __debug__:
        parser.add_argument('--port_no', type=Utils.port_no_valid,
                            help='Port_no in debug mode. Can be any number greater '
                                 'than 6000')
    else:
        parser.add_argument('port_no', type=Utils.port_no_valid,
                            help='Port_no fixed at 6000 according to specifications '
                                 '(https://canvas.sydney.edu.au/courses/48387/assignments/432740?module_item_id=1929246)')

    parser.add_argument('sub_client', type=Utils.verify_flag,
                        help='False -> aggregate all clients. True -> aggregate 2 clients')

    args = parser.parse_args()
    if args.port_no:
        port_no = args.port_no
    sub_client = args.sub_client

    spawn_clients()
