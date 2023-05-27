import os

dirname = os.path.dirname(os.path.abspath(__file__))

code_path = os.path.join(dirname, 'optimal_mask/masks.cpp')
script_path = os.path.join(dirname, 'optimal_mask/cpp_mask')
buffer = os.path.join(dirname, '../data/processed/buffer')


def find_optimal(day, wait, adds):
    with open(buffer + '_in', 'w') as f:
        f.write(f'{day} {wait}\n{" ".join([str(add) for add in adds])}\n')
    with open(buffer + "_out", 'w') as f:
        f.write("")
    if not os.path.exists(script_path):
        os.system(f'g++ {code_path} -o {script_path}')
    os.system(f'{script_path} < {buffer + "_in"} > {buffer + "_out"}')
    status, delta = [float(val) for val in open(buffer + "_out", 'r').readline().strip().split(' ')]
    return int(status), delta


if __name__ == '__main__':
    adds = [160000, 90000, 105000, 99000, 107000, 110000, 60000, 75000, 89000, 95000, 116000, 85000, 110000, 84000, 51000, 51000, 150000, 0, 225000, 75000, 0, 175000, 55000, 125000, 110000, 115000, 100000, 75000, 99000, 100000]
    print(find_optimal(20, 13, adds))
