import argparse
import sys
import subprocess
import psutil
from utils import Utils
from threading import Thread


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Launches Comp3221_FlServer.py in a predictable way.')

    parser.add_argument('sub_client', type=Utils.verify_flag,
                        help='False -> aggregate all clients. True -> aggregate 2 clients')

    args = parser.parse_args()

    proc = subprocess.Popen(
        [sys.executable, '-O', '-u', 'Comp3221_FLServer.py', '6000', str(args.sub_client)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    def watch_subprocess():

        # pid = psutil.Process(proc.pid)
        # children = pid.children(recursive=True)
        # print(children)

        try:
            if proc.stdout:
                while proc.returncode is None:
                    stdout = proc.stdout.readline()
                    if not stdout:
                        break
                    line = stdout.decode().rstrip()

                    print('\u001b[40m' + f'SERVER -> ' + line + '\033[0m')

            _, stderr = proc.communicate()
            if stderr:
                print('\033[0m' + stderr.decode().rstrip())
        finally:
            proc.terminate()

    T1 = Thread(target=watch_subprocess, args=())
    T1.start()
    T1.join()


