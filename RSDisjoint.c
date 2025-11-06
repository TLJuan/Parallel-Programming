/*
Reduce-scatter disjoint algorithm, is a variation of all-reduce where the
data is partitioned into disjoint blocks, and each process performs a reduction
on the blocks it receives. 
The algorithm involves:
  Rotating the blocks.
  Performing the scatter-reduce communication phase.
  Each process ends up with a reduced block of data that combines contributions from all processes.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <mpi.h>

void reduce_scatter_disjoint(const int* V_in, int vector_size, int* W_out, MPI_Comm comm) {
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    if (vector_size % p != 0) {
        if (rank == 0) {
            std::cerr << "Error: Vector size must be divisible by the number of processes (p)." << std::endl;
        }
        return;
    }
    int block_size = vector_size / p;
    int num_blocks = p;
    // --- Initialization: R buffer and Rotated Copy 
    // R will hold p blocks of size block_size. R[i] is the block 
    // that contributes to the final result of processor (rank + i) mod p.
    std::vector<int> R(vector_size);
    std::vector<int> T_recv(vector_size); // Temporary receive buffer

    // Perform the Rotated Copy and initial local reduction
    // R[i * block_size] stores the block that goes to rank (rank + i) mod p.
    for (int i = 0; i < num_blocks; ++i) {
        int src_rank = (rank + i) % p;
        
        // Destination in R buffer: R[i * block_size]
        std::copy(V_in + src_rank * block_size, 
                  V_in + src_rank * block_size + block_size, 
                  R.begin() + i * block_size);
    }

    // W_out will be filled from the final result in R[0 * block_size].
    // --- Reduction (Phase 2)
    int s_prime = p; // s' from the paper
    
    while (s_prime > 1) {
        // Calculate s (new skip distance)
        int s = (s_prime + 1) / 2;
        
        int send_count = (s_prime - s) * block_size; // Number of elements to send
        int recv_count = send_count;                 // Number of elements to receive
        
        // Calculate communication partners (t and f)
        // t: to-processor (rank + s) mod p
        // f: from-processor (rank - s + p) mod p
        int dest = (rank + s) % p;
        int source = (rank - s + p) % p;

        // Calculate buffers for Send and Recv
        // Send buffer starts at R[s * block_size] and spans (s_prime - s) blocks
        int* send_buf = &R[s * block_size];

        // Recv buffer is the temporary T_recv buffer
        int* recv_buf = T_recv.data();

        // Perform Send and Receive
        MPI_Sendrecv(send_buf, send_count, MPI_INT, dest, 0,
                     recv_buf, recv_count, MPI_INT, source, 0,
                     comm, MPI_STATUS_IGNORE);

        // --- Local Reduction
        // The received T_recv blocks need to be reduced into R[0...s'-s-1]
        for (int i = 0; i < (s_prime - s); ++i) {
            int* R_block = &R[i * block_size];
            int* T_block = &T_recv[i * block_size];
            
            // Perform the reduction: R[i] ← R[i] ⊕ T[i] (Vector addition, for now)
            for (int k = 0; k < block_size; ++k) {
                R_block[k] += T_block[k]; // HERE! Replace with any commutative reduction operation
            }
        }
        // Update s_prime for the next round
        s_prime = s;
    }

    // --- Final Result
    std::copy(R.begin(), R.begin() + block_size, W_out);
	// The final reduced block (W_out) is stored in R[0 * block_size]
}

int main(int argc, char* argv[]) {
	//Main with comparison with Native MPI. 
    MPI_Init(&argc, &argv);

    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int vector_size = (argc > 1) ? std::atoi(argv[1]) : 8 * p; //Argument, should be divible by P(# of processors)
    if (vector_size % p != 0) {
        if (rank == 0) {
            std::cerr << "Error: Vector size must be divisible by number of processes (p)." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int block_size = vector_size / p;
    int runs = 5;

    std::vector<int> recv_counts(p, block_size);

    for (int run = 0; run < runs; ++run) {
        std::vector<int> V_in(vector_size);
        std::generate(V_in.begin(), V_in.end(), [rank]() { return rank + 1; });

        std::vector<int> W_out(block_size);
        double t1 = MPI_Wtime();
        reduce_scatter_disjoint(V_in.data(), vector_size, W_out.data(), MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        double custom_time = t2 - t1;

        std::vector<int> native_out(block_size);
        double t3 = MPI_Wtime();
        MPI_Reduce_scatter(V_in.data(), native_out.data(), recv_counts.data(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double t4 = MPI_Wtime();
        double native_time = t4 - t3;

        if (rank == 0) {
            std::cout << "Run " << run + 1 << ": Custom = " << custom_time << " s, Native = " << native_time << " s\n";
        }
    }

    MPI_Finalize();
    return 0;
}