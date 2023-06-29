from fringes import Fringes

f = Fringes()
f.logger.setLevel("DEBUG")
f.Y = 1000
f.X = 1300
f.v = [[1, 5], [1, 3]]
I = f.encode()
dec = f.decode(I)
