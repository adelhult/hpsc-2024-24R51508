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
    const int remainder = ny % size;
    int all_local_ny[size];
    for (auto i = 0; i < size; i++) {
        all_local_ny[i] = ny / size;
        if (i < remainder) {
            all_local_ny[i]++;
        }
    }
    const auto local_ny = all_local_ny[rank]; 


    std::cout << "rank " << rank << "will process " << local_ny << " rows" << std::endl;
    // We add an extra "ghost row" at the top and bottom (not really needed for the first and last row,
    // but let's keep it simple).
    auto u = Matrix<float>(local_ny + 2, nx, 0.0);
    auto v = Matrix<float>(local_ny + 2, nx, 0.0);
    auto p = Matrix<float>(local_ny + 2, nx, 0.0);
    auto b = Matrix<float>(local_ny + 2, nx, 0.0);

    // Also, to make it easier to debug and save the final matrix we
    // store the matrix in its entirety at the first process.
    // only used when debugging right now (see the end of the file).
    std::optional<Matrix<float>> u_full, v_full, p_full, b_full;
    if (rank == 0) {
        u_full.emplace(ny, nx, 0.0);
        v_full.emplace(ny, nx, 0.0);
        p_full.emplace(ny, nx, 0.0);
        b_full.emplace(ny, nx, 0.0);
    }

    // Used to send my rows to neighbors
    // Note: here I let the grid loop around and send messages between the last and first process
    // (but they just ignore those ghost rows when calculating the actual values later!)
    const auto prev = (rank - 1 + size) % size;
    const auto next = (rank + 1) % size;
    const int FIRST_ROW_TAG = 1;
    const int LAST_ROW_TAG = 2;

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

        // Exchange first row with the one before you
        MPI_Isendrecv(b.get() + nx, nx, MPI_FLOAT,
                      prev, FIRST_ROW_TAG,
                      b.get(), nx, MPI_FLOAT,
                      prev, LAST_ROW_TAG, MPI_COMM_WORLD, &requests_b[0]);

        // Exchange last row with the one after you
        MPI_Isendrecv(b.get() + local_ny * nx, nx, MPI_FLOAT,
                      next, LAST_ROW_TAG,
                      b.get() + (local_ny + 1) * nx, nx, MPI_FLOAT,
                      next, FIRST_ROW_TAG, MPI_COMM_WORLD, &requests_b[1]);

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
                          prev, FIRST_ROW_TAG,
                          p.get(), nx, MPI_FLOAT,
                          prev, LAST_ROW_TAG, MPI_COMM_WORLD, &requests_p[0]);

            MPI_Isendrecv(p.get() + local_ny * nx, nx, MPI_FLOAT,
                          next, LAST_ROW_TAG,
                          p.get() + (local_ny + 1) * nx, nx, MPI_FLOAT,
                          next, FIRST_ROW_TAG, MPI_COMM_WORLD, &requests_p[1]);

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

        // Send ghost rows to neighbors
        MPI_Request requests_u_v[4];

        MPI_Isendrecv(u.get() + nx, nx, MPI_FLOAT,
                      prev, FIRST_ROW_TAG,
                      u.get(), nx, MPI_FLOAT,
                      prev, LAST_ROW_TAG, MPI_COMM_WORLD, &requests_u_v[0]);

        MPI_Isendrecv(u.get() + local_ny * nx, nx, MPI_FLOAT,
                      next, LAST_ROW_TAG,
                      u.get() + (local_ny + 1) * nx, nx, MPI_FLOAT,
                      next, FIRST_ROW_TAG, MPI_COMM_WORLD, &requests_u_v[1]);

        MPI_Isendrecv(v.get() + nx, nx, MPI_FLOAT,
                      prev, FIRST_ROW_TAG,
                      v.get(), nx, MPI_FLOAT,
                      prev, LAST_ROW_TAG, MPI_COMM_WORLD, &requests_u_v[2]);

        MPI_Isendrecv(v.get() + local_ny * nx, nx, MPI_FLOAT,
                      next, LAST_ROW_TAG,
                      v.get() + (local_ny + 1) * nx, nx, MPI_FLOAT,
                      next, FIRST_ROW_TAG, MPI_COMM_WORLD, &requests_u_v[3]);

        MPI_Waitall(4, requests_u_v, MPI_STATUS_IGNORE);

        // Debugging (saving the matrix as a .txt file, see the modified 10_cavity.py file for testing)
#ifdef DEBUGGING
        if (n == 5) {
            int receive_counts[size];
            for (auto i = 0; i < size; i++) {
                receive_counts[i] = nx * all_local_ny[i];
            }
            int displacements[size];
            if (rank == 0) {
                displacements[0] = 0;
                for (auto i = 1; i < size; i++) {
                    displacements[i] = displacements[i - 1] + receive_counts[i - 1];
                }
            }

            MPI_Gatherv(u.get() + nx, local_ny * nx, MPI_FLOAT, u_full->get(), receive_counts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gatherv(v.get() + nx, local_ny * nx, MPI_FLOAT, v_full->get(), receive_counts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gatherv(b.get() + nx, local_ny * nx, MPI_FLOAT, b_full->get(), receive_counts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Gatherv(p.get() + nx, local_ny * nx, MPI_FLOAT, p_full->get(), receive_counts, displacements, MPI_FLOAT, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::cout << "Saved debug files" << std::endl;
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
