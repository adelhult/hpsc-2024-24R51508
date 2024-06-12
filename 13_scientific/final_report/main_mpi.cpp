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

    //const auto nx = 41;
    //const auto ny = 41;
    const auto nx = 10;
    const auto ny = 10;

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

    std::cout << "local_ny: " << local_ny << std::endl;
    // We add an extra "ghost row" at the top and bottom (not really needed for the first and last row,
    // but let's keep it simple).
    auto u = Matrix<float>(local_ny + 2, nx, 0.0);
    auto v = Matrix<float>(local_ny + 2, nx, 0.0);
    auto p = Matrix<float>(local_ny + 2, nx, 0.0);
    auto b = Matrix<float>(local_ny + 2, nx, 0.0);

    // Also, to make it easier to debug and save the final matrix we
    // store the matrix in its entirety at the first process.
    std::optional<Matrix<float>> u_full, v_full, p_full, b_full;
    if (rank == 0) {
        u_full.emplace(ny, nx, 0.0);
        v_full.emplace(ny, nx, 0.0);
        p_full.emplace(ny, nx, 0.0);
        b_full.emplace(ny, nx, 0.0);
    }

    // Used to send my rows to neighbors
    // Note: here I let the grid loop around and send messages between the last and first process
    // but they just ignore those ghost rows when calculating the actual values later!
    auto prev = (rank - 1 + size) % size;
    auto next = (rank + 1) % size;

    int first_row = 1;
    // The first process has no use for the ghost row at the top
    if (rank == 0) {
        first_row++;
    }

    int last_row = local_ny + 1;
    // The last process has no use for the ghost row at the bottom
    if (rank == size - 1) {
        last_row--;
    }

    // Let's actually run the algorithm!
    for (auto n = 0; n < nt; n++) {
        for (auto j = first_row; j < last_row; j++) {
            for (auto i = 1; i < nx - 1; i++) {
                b(j, i) = rho * (1 / dt *
                                 ((u(j, i + 1) - u(j, i - 1)) / (2 * dx) + (v(j + 1, i) - v(j - 1, i)) / (2 * dy)) -
                                 powf((u(j, i + 1) - u(j, i - 1)) / (2 * dx), 2) -
                                 2 * ((u(j + 1, i) - u(j - 1, i)) / (2 * dy) *
                                      (v(j, i + 1) - v(j, i - 1)) / (2 * dx)) -
                                 powf((v(j + 1, i) - v(j - 1, i)) / (2 * dy), 2));
            }
        }

        MPI_Request requests_b[2];
        // exchange first row with the one before you
        MPI_Isendrecv(b.get() + nx, nx, MPI_FLOAT,
                      prev, 0,
                      b.get(), nx, MPI_FLOAT,
                      prev, 0, MPI_COMM_WORLD, &requests_b[0]);

        // exchange last row with the one after you
        MPI_Isendrecv(b.get() + local_ny * nx, nx, MPI_FLOAT,
                      next, 0,
                      b.get() + (local_ny + 1) * nx, nx, MPI_FLOAT,
                      next, 0, MPI_COMM_WORLD, &requests_b[1]);

        MPI_Waitall(2, requests_b, MPI_STATUS_IGNORE);

        for (auto it = 0; it < nit; it++) {
            auto pn = Matrix<float>(p);
            for (auto j = first_row; j < last_row; j++) {
                for (auto i = 1; i < nx - 1; i++) {
                    p(j, i) = (powf(dy, 2) * (pn(j, i + 1) + pn(j, i - 1)) +
                               powf(dx, 2) * (pn(j + 1, i) + pn(j - 1, i)) -
                               b(j, i) * powf(dx, 2) * powf(dy, 2))
                              / (2 * (powf(dx, 2) + powf(dy, 2)));
                }
            }

            MPI_Request requests_p[2];

            // Updating the ghost rows in p between each iteration!
            MPI_Isendrecv(p.get() + nx, nx, MPI_FLOAT,
                          prev, 0,
                          p.get(), nx, MPI_FLOAT,
                          prev, 0, MPI_COMM_WORLD, &requests_p[0]);

            MPI_Isendrecv(p.get() + local_ny * nx, nx, MPI_FLOAT,
                          next, 0,
                          p.get() + (local_ny + 1) * nx, nx, MPI_FLOAT,
                          next, 0, MPI_COMM_WORLD, &requests_p[1]);

            MPI_Waitall(2, requests_p, MPI_STATUS_IGNORE);

            // Boundary conditions for p
            const auto rows = p.row_count();
            const auto cols = p.columns_count();

            for (auto i = 0; i < rows; i++) {
                p(i, cols - 1) = p(i, cols - 2); // p[:, -1] = p[:, -2]
                p(i, 0) = p(i, 1);               // p[:, 0] = p[:, 1]
            }

            if (rank == 0) {
                for (auto j = 0; j < cols; j++) {
                    p(1, j) = p(2, j);  // p[0, :] = p[1, :]
                    // Note: it's 1 and 2 and not 0 since the first row is just a ghost row
                }
            }

            if (rank == size - 1) {
                for (auto j = 0; j < cols; j++) {
                    p(rows - 2, j) = 0; // p[-1, :] = 0
                    // Same here: it's -2 since the last row is just a ghost row
                }
            }
        }

        auto un = Matrix<float>(u);
        auto vn = Matrix<float>(v);

        for (auto j = first_row; j < last_row; j++) {
            for (auto i = 1; i < nx - 1; i++) {
                // dependent on
                // previous values: un(j, i), un(j-1, i), un(j+1, i),  un(j, i - 1), un(j, i+1)
                // p(j, i+1), p(j, i-1),
                u(j, i) = un(j, i) - un(j, i) * dt / dx * (un(j, i) - un(j, i - 1))
                          - un(j, i) * dt / dy * (un(j, i) - un(j - 1, i))
                          - dt / (2 * rho * dx) * (p(j, i + 1) - p(j, i - 1))
                          + nu * dt / powf(dx, 2) * (un(j, i + 1) - 2 * un(j, i) + un(j, i - 1))
                          + nu * dt / powf(dy, 2) * (un(j + 1, i) - 2 * un(j, i) + un(j - 1, i));

                v(j, i) = vn(j, i) - vn(j, i) * dt / dx * (vn(j, i) - vn(j, i - 1))
                          - vn(j, i) * dt / dy * (vn(j, i) - vn(j - 1, i))
                          - dt / (2 * rho * dy) * (p(j + 1, i) - p(j - 1, i))
                          + nu * dt / powf(dx, 2) * (vn(j, i + 1) - 2 * vn(j, i) + vn(j, i - 1))
                          + nu * dt / powf(dy, 2) * (vn(j + 1, i) - 2 * vn(j, i) + vn(j - 1, i));
            }
        }

        const auto cols = u.columns_count();
        const auto rows = u.row_count();

        // The first row
        if (rank == 0) {
            for (auto j = 0; j < cols; j++) {
                u(1, j) = 0; // u[0, :]  = 0
                v(1, j) = 0; // v[0, :]  = 0
                // Note: it's 1 and not 0 due to the ghost row
            }
        }

        for (auto i = 0; i < rows; i++) {
            u(i, 0) = 0;
            u(i, cols - 1) = 0;
            v(i, 0) = 0;
            v(i, cols - 1) = 0;
        }

        // The last row
        if (rank == size - 1) {
            for (auto j = 0; j < cols; j++) {
                u(rows - 2, j) = 1; // u[-1, :] = 1
                v(rows - 2, j) = 0; // v[-1, :] = 0
                // Note: It's -2 and not -1 due to the ghost row
            }
        }

        // Send ghost rows to neigbors
        MPI_Request requests_u_v[4];

        // Updating the ghost rows in p between each iteration!
        MPI_Isendrecv(u.get() + nx, nx, MPI_FLOAT,
                      prev, 0,
                      u.get(), nx, MPI_FLOAT,
                      prev, 0, MPI_COMM_WORLD, &requests_u_v[0]);

        MPI_Isendrecv(u.get() + local_ny * nx, nx, MPI_FLOAT,
                      next, 0,
                      u.get() + (local_ny + 1) * nx, nx, MPI_FLOAT,
                      next, 0, MPI_COMM_WORLD, &requests_u_v[1]);

        // Updating the ghost rows in p between each iteration!
        MPI_Isendrecv(v.get() + nx, nx, MPI_FLOAT,
                      prev, 0,
                      v.get(), nx, MPI_FLOAT,
                      prev, 0, MPI_COMM_WORLD, &requests_u_v[2]);

        MPI_Isendrecv(v.get() + local_ny * nx, nx, MPI_FLOAT,
                      next, 0,
                      v.get() + (local_ny + 1) * nx, nx, MPI_FLOAT,
                      next, 0, MPI_COMM_WORLD, &requests_u_v[3]);

        MPI_Waitall(4, requests_u_v, MPI_STATUS_IGNORE);


        // Debugging
#ifdef DEBUGGING
        if (n == 5) {
            // TODO: this breaks if local_ny is different between processes
            // i.e. when ny is not divisible by size. 
            MPI_Gather(u.get() + nx, nx * local_ny, MPI_FLOAT, u_full->get() + nx * local_ny * rank, nx * local_ny,
                       MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(v.get() + nx, nx * local_ny, MPI_FLOAT, v_full->get() + nx * local_ny * rank, nx * local_ny,
                       MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(b.get() + nx, nx * local_ny, MPI_FLOAT, b_full->get() + nx * local_ny * rank, nx * local_ny,
                       MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gather(p.get() + nx, nx * local_ny, MPI_FLOAT, p_full->get() + nx * local_ny * rank, nx * local_ny,
                       MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                std::cout << "Saved debug files" << std::endl;
                std::cout << rank << ":" << std::endl;
                u.print();
                u_full->save_on_disk("./u.txt");
                v_full->save_on_disk("./v.txt");
                p_full->save_on_disk("./p.txt");
                b_full->save_on_disk("./b.txt");
            }
            break;
        }
#endif
    }
    MPI_Finalize();
    std::cout << "Simulation completed" << std::endl;
    return 0;
}
