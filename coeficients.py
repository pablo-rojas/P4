#!usr/bin/python3
import sys
import statistics
import struct

def cms(matrix):
    for x in matrix:
        print(x)
        m = statistics.mean(x)
        for y in x:
            y -= m

def dynamic_inf1(c, l):
    d = [0] * len(c)
    d[0] = 1
    d[len(c)-1] = 1
    for m in range(1, l):
        for k in range(-m, m):
            d[m] += k * c[m+k]/(2*m)
            d[len(c)-m-1] += k * c[len(c)-m-1+k]/(2*m)

    for m in range(l, len(c)-l-1):
        for k in range(-l, l):
            d[m] += k * c[m+k]/(2*l)

    return d
            
def dynamic_inf2(c, l):
    d = [0] * len(c)
    d[0] = 1
    d[len(c)-1] = 1
    for m in range(1, l):
        for k in range(-m, m):
            d[m] += (k**2) * c[m+k] *3/(m * (m+1) * (2*m + m))
            d[len(c) - m-1] += (k**2) * c[len(c)-m-1+k] *3/(m * (m+1) * (2*m + m))

    for m in range(l, len(c)-l-1):
        for k in range(-l, l):
            d[m] += (k**2) * c[m+k]*3/(l * (l+1) * (2*l + 1))

    return d



if len(sys.argv) != 3:
    print('Error: input must be python3 coeficients.py <input_file> <output_file>')
else:
    with open(str(sys.argv[1]), 'rb') as f:
        format0 = '2I'
        data = f.read(struct.calcsize(format0))
        size = struct.unpack(format0, data)
        format1 = str(size[1]) + 'f'
        c = []
        header = []
        for line in range(0, size[0]-1):
            data = f.read(struct.calcsize(format1))
            c.append(struct.unpack(format1, data))
    
    c_t = list(zip(*c))   
    cms(c_t)
    cms = list(zip(*c_t))
    print(cms)

    with open(str(sys.argv[2]), 'wb') as fw:
        data = struct.pack(format0, size[0], size[1])
        fw.write(data)
        for line in c:
            for n in line:
                data = struct.pack('f', n)
                fw.write(data)


