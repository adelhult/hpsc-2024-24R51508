import numpy as np
import matplotlib.pyplot as plt

nx = 10
ny = 10
nt = 500
nit = 50
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = .01
rho = 1
nu = .02

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
X, Y = np.meshgrid(x, y)

# For easier debugging
np.set_printoptions(threshold=np.inf)

def parse_file(file_content):
    """Parse the contents of a file into a numpy 2d array"""
    dim, *lines = file_content.splitlines()
    rows, cols = map(int, dim.split(" "))
    
    result = np.zeros((rows, cols))

    for i, line in enumerate(lines):
        for j, cell in enumerate(line.split(" ")):
            result[i, j] = float(cell)
    
    return result

def read_file(path):
    """Read a file and parse it as numpy 2d array"""
    with open(path, "r") as f:
        return parse_file(f.read())

for n in range(nt):
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            b[j, i] = rho * (1 / dt *\
                    ((u[j, i+1] - u[j, i-1]) / (2 * dx) + (v[j+1, i] - v[j-1, i]) / (2 * dy)) -\
                    ((u[j, i+1] - u[j, i-1]) / (2 * dx))**2 - 2 * ((u[j+1, i] - u[j-1, i]) / (2 * dy) *\
                     (v[j, i+1] - v[j, i-1]) / (2 * dx)) - ((v[j+1, i] - v[j-1, i]) / (2 * dy))**2)
    for it in range(nit):
        pn = p.copy()
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                p[j, i] = (dy**2 * (pn[j, i+1] + pn[j, i-1]) +\
                           dx**2 * (pn[j+1, i] + pn[j-1, i]) -\
                           b[j, i] * dx**2 * dy**2)\
                          / (2 * (dx**2 + dy**2))
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0
    un = u.copy()
    vn = v.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            u[j, i] = un[j, i] - un[j, i] * dt / dx * (un[j, i] - un[j, i - 1])\
                               - un[j, i] * dt / dy * (un[j, i] - un[j - 1, i])\
                               - dt / (2 * rho * dx) * (p[j, i+1] - p[j, i-1])\
                               + nu * dt / dx**2 * (un[j, i+1] - 2 * un[j, i] + un[j, i-1])\
                               + nu * dt / dy**2 * (un[j+1, i] - 2 * un[j, i] + un[j-1, i])
            v[j, i] = vn[j, i] - vn[j, i] * dt / dx * (vn[j, i] - vn[j, i - 1])\
                               - vn[j, i] * dt / dy * (vn[j, i] - vn[j - 1, i])\
                               - dt / (2 * rho * dy) * (p[j+1, i] - p[j-1, i])\
                               + nu * dt / dx**2 * (vn[j, i+1] - 2 * vn[j, i] + vn[j, i-1])\
                               + nu * dt / dy**2 * (vn[j+1, i] - 2 * vn[j, i] + vn[j-1, i])
    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = 1
    v[0, :]  = 0
    v[-1, :] = 0
    v[:, 0]  = 0
    v[:, -1] = 0

    # Debugging
    if (n == 5):
        error_margin = 1e-3 # rounding differences? (At least I hopes so :sweat: !)
        print("u:")
        print((read_file("./final_report/u.txt") - u) < error_margin)
        #assert ((read_file("./final_report/u.txt") - u) < error_margin).all()
        
        print("v:")
        print((read_file("./final_report/v.txt") - v) < error_margin)
        #assert ((read_file("./final_report/v.txt") -v) < error_margin).all()
        
        print("p:")
        print((read_file("./final_report/p.txt") - p) < error_margin)
        #assert ((read_file("./final_report/p.txt") -p) < error_margin).all()

        print("b:")
        print((read_file("./final_report/b.txt") - b) < error_margin)
        #assert ((read_file("./final_report/b.txt") - b) < error_margin).all()
        break

    # plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
    # plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    # plt.pause(.01)
    # plt.clf()
#plt.show()
