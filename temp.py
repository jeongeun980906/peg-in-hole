#!/usr/bin/python3

import math

temp=[-0.08917862921953201, 0.6996442079544067, 0.08796319365501404, 0.7034257054328918]
x=temp[0]
y=temp[1]
print(math.pi/2)
z=temp[2]
w=temp[3]

t0 = +2.0 * (w * x + y * z)
t1 = +1.0 - 2.0 * (x * x + y * y)
X = math.atan2(t0, t1)
print(t0,t1,w)
t2 = +2.0 * (w * y - z * x)
t2 = +1.0 if t2 > +1.0 else t2
t2 = -1.0 if t2 < -1.0 else t2
Y = math.asin(t2)

t3 = +2.0 * (w * z + x * y)
t4 = +1.0 - 2.0 * (y * y + z * z)
Z = math.atan2(t3, t4)

print(X,Y,Z)