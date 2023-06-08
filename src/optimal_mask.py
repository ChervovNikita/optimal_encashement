import os
import random

# dirname = os.path.dirname(os.path.abspath(__file__))

code_path = 'src/optimal_mask/main.cpp'
script_path = 'src/cpp_mask'
buffer = 'data/processed/buffer3'


def find_optimal(day, wait, adds):
    # additional = random.randint(0, 100000000000)
    # buffer = buffer_origin + str(additional)
    # if random.random() < 0.1:
    #     print(day, wait, adds)

    with open(buffer + '_in', 'w') as f:
        f.write(f'{day} {wait}\n{" ".join([str(add) for add in adds])}\n')
    with open(buffer + "_out", 'w') as f:
        f.write("")
    if not os.path.exists(script_path):
        print('Build script')
        os.system(f'g++ {code_path} -o {script_path}')
    # else:
        # print('Script was not built')
    os.system(f'{script_path} < {buffer + "_in"} > {buffer + "_out"}')
    delta = float(open(buffer + "_out", 'r').readline().strip())
    return delta


if __name__ == '__main__':
    adds = [160000, 90000, 105000, 99000, 107000, 110000, 60000, 75000, 89000, 95000, 116000, 85000, 110000, 84000, 51000, 51000, 150000, 0, 225000, 75000, 0, 175000, 55000, 125000, 110000, 115000, 100000, 75000, 99000, 100000]
    print(find_optimal(20, 13, adds))
