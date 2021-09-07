
import re

filename = "format.txt"
with open(filename,"r", encoding="utf-8") as f:
    text = f.readlines()

for i in range(len(text)):
    text[i] = re.sub('\n', ' ', text[i])  # Delete pattern abc
    text[i] = re.sub("\.",".\n",text[i])

print(text)


with open(filename,"w", encoding="utf-8") as f:
    for i in range(len(text)):
        f.write(text[i])

# with open(filename,"a", encoding="utf-8") as f:
#     f.write("ciao")


#print(my_lines)

