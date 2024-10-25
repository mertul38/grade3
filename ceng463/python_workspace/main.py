


import re



txt = """RegExr was created by gskinner.com.

Edit the Expression & Text to see matches. Roll over matches or the expression for details. PCRE & JavaScript flavors of RegEx are supported. Validate your expression with Tests mode.

The side bar includes a Cheatsheet, full Reference, and Help. You can also Save & Share with the Community and view patterns you create or favorite in My Patterns.

Explore results with the Tools below. Replace & List output custom results. Details lists capture groups. Explain describes your expression in plain English."""

strings = txt.split(" ")
print(f"strings: {strings}")

result_lst = []

for index, s in enumerate(strings):
    result = re.findall("^[A-Ba-d]", s)
    if result:
        result_lst.append((result, index))

print(result_lst)

ex_txt_0 = "the faster they ran, the faster we ran"

r = re.findall(r"the (.*) they (.*), the \1 we \2", ex_txt_0)
print(r)