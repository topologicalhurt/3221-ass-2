import argparse

class Utils:

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