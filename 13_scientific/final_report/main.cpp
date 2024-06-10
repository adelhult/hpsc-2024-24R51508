#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>


template<class T>
class Matrix {
    std::vector<T> data;
    unsigned int columns;

public:
    Matrix(unsigned int row, unsigned int col) : data(row * col), columns(col) {}

    Matrix(unsigned int row, unsigned int col, T fill) : data(row * col, fill), columns(col) {}

    // Got inspiration from here to use the function application operator for indexing:
    // https://stackoverflow.com/questions/6465133/stdvector-and-contiguous-memory-of-multidimensional-arrays
    T &operator()(unsigned int row, unsigned int col) {
        return data[row * columns + col];
    }

    unsigned int row_count() const {
        return data.size() / columns;
    }

    unsigned int columns_count() const {
        return columns;
    }

    void print() const {
        for (auto row = 0; row < row_count(); row++) {
            for (auto col = 0; col < columns_count(); col++) {
                std::cout << data[row * columns + col];
                if (col < columns - 1) {
                    std::cout << " ";
                }
            }
            std::cout << "\n";
        }
    }

    void save_on_disk(const char *path) const {
        std::ofstream file(path);

        file << row_count() << " " <<  columns_count() << std::endl;

        for (auto row = 0; row < row_count(); row++) {
            for (auto col = 0; col < columns_count(); col++) {
                file << data[row * columns + col];
                if (col < columns - 1) {
                    file << " ";
                }
            }
            file << "\n";
        }
    }
};


int main() {
    const auto nx = 41;
    const auto ny = 41;

    const auto nt = 500;
    const auto nit = 50;

    const float dx = 2.0 / (nx - 1);
    const float dy = 2.0 / (ny - 1);

    const float dt = 0.01f;
    const float rho = 1.0f;
    const float nu = 0.02f;

    auto u = Matrix<float>(ny, nx, 0.0);
    auto v = Matrix<float>(ny, nx, 0.0);
    auto p = Matrix<float>(ny, nx, 0.0);
    auto b = Matrix<float>(ny, nx, 0.0);

    for (auto n = 0; n < nt; n++) {
        for (auto j = 1; j < ny - 1; j++) {
            for (auto i = 1; i < nx - 1; i++) {
                // dependent on
                // u(j, i+1), u(j, i-1), u(j + 1, i), u(j - 1, i)
                // v(j + 1, i), v(j - 1, i), v(j, i + 1), v(j, i - 1)
                b(j, i) = rho * (1 / dt *
                                  ((u(j, i + 1) - u(j, i - 1)) / (2 * dx) + (v(j + 1, i) - v(j - 1, i)) / (2 * dy)) -
                                  powf((u(j, i + 1) - u(j, i - 1)) / (2 * dx), 2) -
                                  2 * ((u(j + 1, i) - u(j - 1, i)) / (2 * dy) *
                                       (v(j, i + 1) - v(j, i - 1)) / (2 * dx)) -
                                  powf((v(j + 1, i) - v(j - 1, i)) / (2 * dy), 2));
            }
        }

        for (auto it = 0; it < nit; it++) {
            auto pn = Matrix<float>(p);
            for (auto j = 1; j < ny - 1; j++) {
                for (auto i = 1; i < nx - 1; i++) {
                    // dependent on
                    // previous values of p: pn(j, i + 1), pn(j, i - 1), pn(j + 1, i), pn(j - 1, i)
                    // b(j, i)
                    p(j, i) = (powf(dy, 2) * (pn(j, i + 1) + pn(j, i - 1)) +
                               powf(dx, 2) * (pn(j + 1, i) + pn(j - 1, i)) -
                               b(j, i) * powf(dx, 2) * powf(dy, 2))
                              / (2 * (powf(dx, 2) + powf(dy, 2)));
                }
            }

            const auto cols = p.columns_count();
            const auto rows = p.row_count();

            for (auto i = 0; i < rows; i++) {
                p(i, cols - 1) = p(i,cols - 2); // p[:, -1] = p[:, -2]
                p(i, 0) = p(i, 1);              // p[:, 0] = p[:, 1]
            }


            for (auto j = 0; j < cols; j++) {
                p(0, j) = p(1, j);              // p[0, :] = p[1, :]
                p(rows - 1, j) = 0;                     // p[-1, :] = 0
            }
        }

        auto un = Matrix<float>(u);
        auto vn = Matrix<float>(v);

        for (auto j = 1; j < ny-1; j++) {
            for (auto i = 1; i < nx-1; i++) {
                // dependent on
                // previous values: un(j, i), un(j-1, i), un(j+1, i),  un(j, i - 1), un(j, i+1)
                // p(j, i+1), p(j, i-1),
                u(j, i) = un(j, i) - un(j, i) * dt / dx * (un(j, i) - un(j, i - 1))
                               - un(j, i) * dt / dy * (un(j, i) - un(j - 1, i))
                               - dt / (2 * rho * dx) * (p(j, i+1) - p(j, i-1))
                               + nu * dt / powf(dx,2) * (un(j, i+1) - 2 * un(j, i) + un(j, i-1))
                               + nu * dt / powf(dy,2) * (un(j+1, i) - 2 * un(j, i) + un(j-1, i));
                // same thing but for vn and p
                v(j, i) = vn(j, i) - vn(j, i) * dt / dx * (vn(j, i) - vn(j, i - 1))
                               - vn(j, i) * dt / dy * (vn(j, i) - vn(j - 1, i))
                               - dt / (2 * rho * dy) * (p(j+1, i) - p(j-1, i))
                               + nu * dt / powf(dx,2) * (vn(j, i+1) - 2 * vn(j, i) + vn(j, i-1))
                               + nu * dt / powf(dy,2) * (vn(j+1, i) - 2 * vn(j, i) + vn(j-1, i));
            }
        }

        const auto cols = u.columns_count();
        const auto rows = u.row_count();

        for (auto j = 0; j < cols; j++) {
            u(0, j) = 0;      // u[0, :]  = 0
            u(rows-1, j) = 1; // u[-1, :] = 1
            v(0, j) = 0;      // v[0, :]  = 0
            v(rows-1, j) = 0; // v[-1, :] = 0
        }
        
        for (auto i = 0; i < rows; i++) {
            u(i, 0) = 0;      // u[:, 0]  = 0
            u(i, cols-1) = 0; // u[:, -1] = 0
            v(i, 0) = 0;      // v[:, 0]  = 0
            v(i, cols-1) = 0; // v[:, -1] = 0
        }

        // Debugging
#ifdef DEBUGGING
        if (n == 5) {
            std::cout << "Saved debug files" << std::endl;
            u.save_on_disk("/home/eli/tokyo-tech/hpsc-2024-24R51508/13_scientific/output/u.txt");
            v.save_on_disk("/home/eli/tokyo-tech/hpsc-2024-24R51508/13_scientific/output/v.txt");
            p.save_on_disk("/home/eli/tokyo-tech/hpsc-2024-24R51508/13_scientific/output/p.txt");
            b.save_on_disk("/home/eli/tokyo-tech/hpsc-2024-24R51508/13_scientific/output/b.txt");
            break;
        }
#endif
    }
    std::cout << "Simulation completed" << std::endl;
}