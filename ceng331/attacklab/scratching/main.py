

filenames = ["ctarget.l1", "ctarget.l2", "ctarget.l3", "rtarget.l2"]
stack_size = 11

def func(filename):
    with open(filename, "r") as file:
        line = file.readlines()
        if(len(line) == 1):
            line = line[0]
        else:
            a = ""
            for i in line:
                a += i+" "
            line = a
        bytes = line.split()
        e_bytes = [bytes[i:i + 8] for i in range(0, len(bytes), 8)]
        print(f"name: {filename} - length: {len(e_bytes)}")
        for index, i in enumerate(e_bytes):
            print(i)
            if(index == stack_size - 1):
                print("-----------------")
        print("\n\n")
for name in filenames:
    print(func(name))