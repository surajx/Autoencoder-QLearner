import sys


def create_uniq(src, dest):
    """ Finds unique states and creates a new file.

        (str, str) -> None
    """
    uniq_dict = {}
    with open(src, 'r') as orig:
        state_str = ""
        for l in orig:
            if l == '\n':
                if state_str not in uniq_dict:
                    uniq_dict[state_str] = 1
                state_str = ""
            state_str += l
    with open(dest, 'w') as new:
        for s in uniq_dict.keys():
            for r in s.split('\n'):
                if r == '':
                    continue
                new.write(r + '\n')
            new.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python flat.py src_file dest_file")
        sys.exit()
    create_uniq(sys.argv[1], sys.argv[2])
