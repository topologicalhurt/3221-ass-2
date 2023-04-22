import argparse
import threading
import sys
import subprocess
import struct
import os
import socket

from utils import Utils, ThreadSafeCounter, Comms

if Utils.POSIX:
    import signal


class ClientManager:

    def __init__(self):
        self.tsc = ThreadSafeCounter(Utils.NET_CONF.n_clients)
        self.clients = None
        self.barrier = threading.Barrier(Utils.NET_CONF.n_clients)

        # Listen to incoming packets
        self.tcpm = Comms(Utils.NET_CONF.host, port_no)  # TCP manager
        self.tcpm.start_listener(False, threads=[])

        # if Utils.POSIX:
        #     self.event = threading.Event()
        #     signal.signal(signal.SIGUSR1, self.__dec_tsc_sigusr1())

    # Deprecated for now - just not really worth using signals over pipes
    # if Utils.POSIX:
    #     def __dec_tsc_sigusr1(self):
    #         self.tsc -= 1
    #         if int(self.tsc) <= 0:
    #             self.event.set()

    def start(self):
        self.clients = [ClientManager.__start_flclient(i, True) for i in range(Utils.NET_CONF.n_clients)]
        self.__spawn_client_watchers(True, threads=[])

    @staticmethod
    def __start_flclient(cid: int, opt_method: bool):
        opt_method = '1' if opt_method else '0'
        proc = subprocess.Popen(
            [sys.executable, 'Comp3221_FLClient.py', f'client{cid}', str(Utils.NET_CONF.port + cid), opt_method],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        _ = proc.communicate(input=struct.pack('!h', os.getpid()))[0]

        return proc

    @Utils.thread_spawner
    def __spawn_client_watchers(self, join, threads=None):
        for i, p in enumerate(self.clients):
            threads.append(threading.Thread(target=ClientManager.__watch_client, args=[i, p, self.tcpm, self.barrier]))

    @staticmethod
    def __watch_client(cid: int, p, tcpm: Comms, barrier: threading.Barrier):
        out = p.communicate()[0]
        # Used to ensure all subprocesses are not competing with threads before any packets arrive
        if out == Utils.ClientComms.RESUME_SERVER.value:
            sys.stdout.write('Client done\n')
            sys.stdout.write(f'{tcpm.buffer}\n')


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

    cm = ClientManager()
    cm.start()
