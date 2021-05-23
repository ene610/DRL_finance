

import os
abs_path = os.path.abspath(os.getcwd())


dirname = abs_path + "/trained_agents"

for file in os.listdir(dirname):
    print(file)

input()






