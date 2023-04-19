import argparse
from utils import Utils

parser = argparse.ArgumentParser(
    prog='Comp3221_FLServer.py',
    description='Finds the federated average on MNIST hand-writing problem amongst 5 clients.')

port_no = 6000

if __debug__:

    def port_no_valid(arg: str) -> int:
        if not all(c.isdigit() for c in arg):
            raise argparse.ArgumentTypeError('Must be an integer')
        arg = int(arg)
        if arg > 0xffff or arg < 6000:
            raise argparse.ArgumentTypeError('Must be in range [6000, 0xffff]')
        return arg

    parser.add_argument('--port_no', type=port_no_valid, help='Port_no in debug mode. Can be any number greater than '
                                                              '6000')

else:

    def port_no_valid(arg: str) -> int:
        if not all(c.isdigit() for c in arg):
            raise argparse.ArgumentTypeError('Must be an integer')
        arg = int(arg)
        if arg != 6000:
            raise argparse.ArgumentTypeError('Port number must be 6000 in production mode')
        return arg

    parser.add_argument('port_no', type=port_no_valid, help='''Port_no fixed at 6000 according to specifications
    (https://canvas.sydney.edu.au/courses/48387/assignments/432740?module_item_id=1929246)''')


parser.add_argument('sub_client', type=Utils.verify_flag,
                    help='False -> aggregate all clients. True -> aggregate 2 clients')


args = parser.parse_args()
if args.port_no:
    port_no = args.port_no
sub_client = args.sub_client
