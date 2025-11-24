/*
Reduce-scatter disjoint algorithm, is a variation of all-reduce where the
data is partitioned into disjoint blocks, and each process performs a reduction
on the blocks it receives.
The algorithm involves:
  Rotating the blocks.
  Performing the scatter-reduce communication phase.
  Each process ends up with a reduced block of data that combines contributions from all processes.

Optimization: Near Zero-Copy using MPI Datatypes to eliminate the send-side packing copy.
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
    
    // --- Initial Parameter Check and Setup ---
    if (vector_size % p != 0) {
        if (rank == 0) {
            std::cerr << "Error: Vector size must be divisible by the number of processes (p)." << std::endl;
        }
        return;
    }
    int block_size = vector_size / p;
    int num_blocks = p;

    // R: Main work buffer (vector_size elements, p blocks).
    // R[i] holds the block for the final result of processor (rank + i) mod p.
    std::vector<int> R(vector_size);
    // T_recv: Temporary buffer is required for the reduction step.
    std::vector<int> T_recv(vector_size); 

    // Rotated Copy
    for (int i = 0; i < num_blocks; ++i) {
        // Source block index in V_in: (rank + i) mod p
        int src_rank = (rank + i) % p;
        
        // Copy V_in block from src_rank into R[i]
        std::copy(V_in + src_rank * block_size, 
                  V_in + src_rank * block_size + block_size, 
                  R.begin() + i * block_size);
    }

    // Reduction 
    int s_prime = p; 
    
    while (s_prime > 1) {
        int s = (s_prime + 1) / 2; // New skip distance (s)
        
        int num_blocks_to_xfer = s_prime - s; 
        int send_count_elems = num_blocks_to_xfer * block_size;
        
        // From the paper: t=(r+s) mod p, f=(r-s+p) mod p
        int dest = (rank + s) % p;
        int source = (rank - s + p) % p;

        // ZERO-COPY OPTIMIZATION ATTEMPT
	// Scattered datatype to send blocks R[s]...R[s'-1]
        // from the R buffer, theorically eliminating the T_send buffer copy.
        
        MPI_Datatype send_type;
        MPI_Type_vector(num_blocks_to_xfer, block_size, block_size, MPI_INT, &send_type);
        
        // Resize the type so it starts at R[s]
        MPI_Aint base_addr;
        MPI_Get_address(R.data(), &base_addr);
        
        // Displacement (in bytes) to the start of the first block being sent (R[s])
        MPI_Aint send_disp_bytes = (MPI_Aint)s * block_size * sizeof(int);
        
        MPI_Aint new_extent = (MPI_Aint)num_blocks_to_xfer * block_size * sizeof(int);
        
        MPI_Datatype final_send_type;
        MPI_Type_create_resized(send_type, send_disp_bytes, new_extent, &final_send_type);
        MPI_Type_commit(&final_send_type);
                
        MPI_Sendrecv(R.data(), 1, final_send_type, dest, 0,
                     T_recv.data(), send_count_elems, MPI_INT, source, 0,
                     comm, MPI_STATUS_IGNORE);

        MPI_Type_free(&send_type);
        MPI_Type_free(&final_send_type);
        

        // --- Local Reduction (Requires T_recv) ---
        // The received T_recv blocks need to be reduced into R[0...s'-s-1]
        for (int i = 0; i < num_blocks_to_xfer; ++i) {
            int* R_block = &R[i * block_size];
            int* T_block = &T_recv[i * block_size];
            
            // Reduction: R[i] ← R[i] ⊕ T[i] (MPI_SUM)
            for (int k = 0; k < block_size; ++k) {
                R_block[k] += T_block[k]; 
            }
        }

        s_prime = s;    }

    std::copy(R.begin(), R.begin() + block_size, W_out);
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int vector_size = (argc > 1) ? std::atoi(argv[1]) : 8 * p; // Default: 8 blocks total
    if (vector_size % p != 0) {
        if (rank == 0) {
            std::cerr << "Error: Vector size must be divisible by number of processes (p)." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int block_size = vector_size / p;
    int runs = 5;

    // recv_counts for native MPI_Reduce_scatter
    std::vector<int> recv_counts(p, 0);
    recv_counts[rank] = block_size; // Only rank 'r' receives its block W_r

    if (rank == 0) {
        std::cout << "--- Disjoint Reduce-Scatter Benchmark ---\n";
        std::cout << "P=" << p << ", Vector Size=" << vector_size << ", Block Size=" << block_size << ", Runs=" << runs << "\n";
    }

    for (int run = 0; run < runs; ++run) {
        std::vector<int> V_in(vector_size);
        // Fill V_in: Block i for rank r contains all 'r+1's.
        // V_in[j] = rank + 1
        std::generate(V_in.begin(), V_in.end(), [rank]() { return rank + 1; });
        
        std::vector<int> W_out(block_size);
        double t1 = MPI_Wtime();
        reduce_scatter_disjoint(V_in.data(), vector_size, W_out.data(), MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        
        double local_time_custom = t2 - t1;
        double max_time_custom;
        MPI_Reduce(&local_time_custom, &max_time_custom, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Native MPI 
        std::vector<int> native_out(block_size);
        double t3 = MPI_Wtime();
        MPI_Reduce_scatter(V_in.data(), native_out.data(), recv_counts.data(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double t4 = MPI_Wtime();
        
        double local_time_native = t4 - t3;
        double max_time_native;
        MPI_Reduce(&local_time_native, &max_time_native, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Expected result for the sum of rank+1 across P processes: P * (rank + 1)
        long long expected_val = (long long)p * (rank + 1);
        int custom_ok = 1;
        int native_ok = 1;

        for (int val : W_out) {
            if (val != expected_val) custom_ok = 0;
        }
        for (int val : native_out) {
            if (val != expected_val) native_ok = 0;
        }

        if (rank == 0) {
            std::cout << "Run " << run + 1 
                      << ": Custom = " << max_time_custom * 1e6 << " us (" << (custom_ok ? "OK" : "FAIL") << ")"
                      << ", Native = " << max_time_native * 1e6 << " us (" << (native_ok ? "OK" : "FAIL") << ")\n";
        }
    }

    MPI_Finalize();
    return 0;
}