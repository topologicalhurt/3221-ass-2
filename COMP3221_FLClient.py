import argparse
import re
import struct
import threading
import subprocess
import sys
import os

from utils import Utils, Comms

if Utils.POSIX:
    import signal


class Client:

    def __init__(self, id: int, port_no: int, opt_method: bool):
        self.id = id
        self.port_no = port_no
        self.opt_method = opt_method

        # Listen to incoming packets
        self.tcpm = Comms(Utils.NET_CONF.host, self.port_no)  # TCP manager
        self.tcpm.start_listener(False, threads=[])

    @Comms.tcp_broadcaster
    def send_handshake(self, host, port):
        # Assume server centralised (we know where to find it without prior contact)
        header = Utils.ClientComms.HANDSHAKE.value
        client_size = b'0'  # TODO: how do I get this?
        packet = header + client_size + struct.pack('!h', self.id)
        return packet


if __name__ == '__main__':

    # Get parent pid
    line = sys.stdin.buffer.read(2)
    ppid = struct.unpack('!h', line)[0]

    parser = argparse.ArgumentParser(
        prog='Comp3221_FLClient.py',
        description='One of the five client instances that handle each communication round.')


    def client_id_valid(arg: str) -> str:
        if not re.match(r'(?i)client\d+', arg):
            raise argparse.ArgumentTypeError('Didn\'t match the pattern client[0-9]+')
        return arg


    parser.add_argument('client_id', type=client_id_valid, help='ID of a client in a federated learning network')
    parser.add_argument('port_client', type=Utils.port_no_valid, help='The port number of a client receiving the model '
                                                                      'packets from the server')
    parser.add_argument('opt_method', type=Utils.verify_flag, help='False -> GD opt. method. True -> Mini-batch GD')

    args = parser.parse_args()

    # Now check that the id's are indexed correctly
    cid = int(re.split(r'(?i)client', args.client_id)[-1])
    if cid != args.port_client - Utils.NET_CONF.port:
        raise argparse.ArgumentTypeError('ID indexed incorrectly (client_id digit must match difference '
                                         'between client_port and starting port')

    client = Client(cid, args.port_client, args.opt_method)

    # SETUP DONE AFTER THIS LINE (SERVER WILL NOT BLOCK FOR CHILD PAST THIS POINT)

    # if Utils.POSIX:
    #     os.kill(ppid, signal.SIGUSR1)

    # Prefer the pipe method - better portability - ensures TCP protocol doesn't run into any timing errors
    sys.stdout.buffer.write(Utils.ClientComms.RESUME_SERVER.value)

    client.send_handshake(Utils.NET_CONF.host, Utils.NET_CONF.port)


