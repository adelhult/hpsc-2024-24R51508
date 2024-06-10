#include <math.h>
#include <optional>
#include <mpi.h>
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

    T *get() {
        return data.data();
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

        file << row_count() << " " << columns_count() << std::endl;

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


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto nx = 41;
    const auto ny = 41;

    const auto nt = 500;
    const auto nit = 50;

    const float dx = 2.0 / (nx - 1);
    const float dy = 2.0 / (ny - 1);

    const float dt = 0.01f;
    const float rho = 1.0f;
    const float nu = 0.02f;

    // Determine number of rows needed locally
    int local_ny = ny / size;
    int remainder = ny % size;
    if (rank < remainder) {
        local_ny++;
    }

    // We add an extra "ghost row" at the top and bottom (not really needed for the first and last row,
    // but let's keep it simple).
    auto u = Matrix<float>(local_ny + 2, nx, 0.0);
    auto v = Matrix<float>(local_ny + 2, nx, 0.0);
    auto p = Matrix<float>(local_ny + 2, nx, 0.0);
    auto b = Matrix<float>(local_ny + 2, nx, 0.0);

    int first_row = 1;
    // The first process has no use for the ghost row at the top
    if (rank == 0) {
        first_row++;
    }

    int last_row = nx + 1;
    // The last process has no use for the ghost row at the bottom
    if (rank == size - 1) {
        last_row--;
    }

    // Also, to make it easier to debug and save the final matrix we
    // store the matrix in its entirety at the first process.
    std::optional<Matrix<float>> u_full, v_full, p_full, b_full;
    if (rank == 0) {
        u_full.emplace(ny, nx, 0.0);
        v_full.emplace(ny, nx, 0.0);
        p_full.emplace(ny, nx, 0.0);
        b_full.emplace(ny, nx, 0.0);
    }

    for (auto n = 0; n < nt; n++) {
        for (auto j = first_row; j < last_row; j++) {
            for (auto i = 1; i < nx - 1; i++) {
                b(j, i) = rank;
              /*  b(j, i) = rho * (1 / dt *
                                 ((u(j, i + 1) - u(j, i - 1)) / (2 * dx) + (v(j + 1, i) - v(j - 1, i)) / (2 * dy)) -
                                 powf((u(j, i + 1) - u(j, i - 1)) / (2 * dx), 2) -
                                 2 * ((u(j + 1, i) - u(j - 1, i)) / (2 * dy) *
                                      (v(j, i + 1) - v(j, i - 1)) / (2 * dx)) -
                                 powf((v(j + 1, i) - v(j - 1, i)) / (2 * dy), 2));*/
            }
        }

        for (auto it = 0; it < nit; it++) {
            auto pn = Matrix<float>(p);
            for (auto j = first_row; j < last_row; j++) {
                for (auto i = 1; i < nx - 1; i++) {
                    p(j, i) = rank;
               /*     p(j, i) = (powf(dy, 2) * (pn(j, i + 1) + pn(j, i - 1)) +
                               powf(dx, 2) * (pn(j + 1, i) + pn(j - 1, i)) -
                               b(j, i) * powf(dx, 2) * powf(dy, 2))
                              / (2 * (powf(dx, 2) + powf(dy, 2)));*/
                }
            }

            // Gather the entire matrix at process 0 and then scatter the data again to update the ghost rows.
            // IMPORTANT NOTE: This adds a *lot* of needless overhead since we really only need to sync the ghost
            // rows in each iteration!
            MPI_Gather(b.get() + nx, nx * local_ny, MPI_FLOAT, b_full->get() + nx * local_ny * rank, nx * local_ny,
                       MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(p.get() + nx, nx * local_ny, MPI_FLOAT, p_full->get() + nx * local_ny * rank, nx * local_ny,
                       MPI_FLOAT, 0, MPI_COMM_WORLD);

            const char *send_buffer = nullptr;
            if (b_full) {
                send_buffer = b_full->get();
            }
            MPI_Scatter(send_buffer, nx * (local_ny + 2), MPI_FLOAT,
                        b.get(), nx * (local_ny + 2), MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (b_full) {
                send_buffer = p_full->get();
            }
            MPI_Scatter(send_buffer, nx * (local_ny + 2), MPI_FLOAT,
                        p.get(), nx * (local_ny + 2), MPI_FLOAT, 0, MPI_COMM_WORLD);

//            const auto cols = p.columns_count();
//            const auto rows = p.row_count(); // TODO: should not use rows! instead use local_ny!
//
//
//            // Boundary conditions for p
//            for (auto i = 0; i < rows; i++) {
//                p(i, cols - 1) = p(i, cols - 2); // p[:, -1] = p[:, -2]
//                p(i, 0) = p(i, 1);               // p[:, 0] = p[:, 1]
//            }
//
//            if (rank == 0) {
//                for (auto j = 0; j < cols; j++) {
//                    p(0, j) = p(1,
//                                j);  // p[0, :] = p[1, :] // TODO: this is probably wrong since we have that extra row! Could just do all this at the first processes maybe?
//                }
//            }
//
//            if (rank == size - 1) {
//                for (auto j = 0; j < cols; j++) {
//                    p(rows - 1, j) = 0; // p[-1, :] = 0
//                }
//            }
        }
//
//        auto un = Matrix<float>(u);
//        auto vn = Matrix<float>(v);
//
//        for (auto j = 1; j < local_ny + 1; j++) {
//            for (auto i = 1; i < nx - 1; i++) {
//                u(j, i) = un(j, i) - un(j, i) * dt / dx * (un(j, i) - un(j, i - 1))
//                          - un(j, i) * dt / dy * (un(j, i) - un(j - 1, i))
//                          - dt / (2 * rho * dx) * (p(j, i + 1) - p(j, i - 1))
//                          + nu * dt / powf(dx, 2) * (un(j, i + 1) - 2 * un(j, i) + un(j, i - 1))
//                          + nu * dt / powf(dy, 2) * (un(j + 1, i) - 2 * un(j, i) + un(j - 1, i));
//
//                v(j, i) = vn(j, i) - vn(j, i) * dt / dx * (vn(j, i) - vn(j, i - 1))
//                          - vn(j, i) * dt / dy * (vn(j, i) - vn(j - 1, i))
//                          - dt / (2 * rho * dy) * (p(j + 1, i) - p(j - 1, i))
//                          + nu * dt / powf(dx, 2) * (vn(j, i + 1) - 2 * vn(j, i) + vn(j, i - 1))
//                          + nu * dt / powf(dy, 2) * (vn(j + 1, i) - 2 * vn(j, i) + vn(j - 1, i));
//            }
//        }
//
//        const auto cols = u.columns_count();
//        const auto rows = u.row_count();
//
//        // The first row
//        if (rank == 0) {
//            for (auto j = 0; j < cols; j++) {
//                u(0, j) = 0;
//                v(0, j) = 0;
//            }
//        }
//
//        // The last row
//        if (rank == size - 1) {
//            for (auto j = 0; j < cols; j++) {
//                u(rows - 1, j) = 1;
//                v(rows - 1, j) = 0;
//            }
//        }
//
//        for (auto i = 0; i < rows; i++) {
//            u(i, 0) = 0;
//            u(i, cols - 1) = 0;
//            v(i, 0) = 0;
//            v(i, cols - 1) = 0;
//        }

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
    MPI_Finalize();
    std::cout << "Simulation completed" << std::endl;
    return 0;
}
