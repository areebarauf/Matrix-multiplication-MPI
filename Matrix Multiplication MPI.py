from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
workers = comm.Get_size() - 1
root = 0

N = 10
start_time = MPI.Wtime()


def matrix_multiplication(row, row1):
    result = np.dot(row, row1)
    return result


def create_transpose(matrix):
    desired_output = matrix.transpose()
    return desired_output


def matrix_to_array(matrix, size):
    rows = []
    _number_array = len(matrix) // size
    # print('_number_array:',_number_array)
    remainder_to_add = len(matrix) % size
    # print('remainder_to_add:',remainder_to_add)
    row_adjustment, elements = 0, _number_array + min(1, remainder_to_add)
    # print('row_adjustment:{} and elements:{}',row_adjustment,elements)
    for i in range(size):
        rows.append(matrix[row_adjustment:elements])
        row_to_create = max(0, row_adjustment - 1)
        row_adjustment, elements = elements, elements + _number_array + min(1, remainder_to_add)
        # print('Process:{} and data:{} and Data Transfered:{}', row_adjustment, elements, rows,'\n')
    return rows


if rank == 0:
    result = []
    B = np.random.randint(5, size=(N, N))
    # print('Matrix B:', B, '\n')
    A = np.random.randint(5, size=(N, N))
    # Creating chunks of Matrix A
    chunks_A = matrix_to_array(A, size)

else:
    A = None
    B = None
    chunks_A = None
    transposed_B = None
    chunk_transposed_B = None
    result = None

result1 = []
# chunks will be sent to the each process
recv_chunk_A = comm.scatter(chunks_A, root=0)
# broadcasting Matrix B to all the processws
recv_chunk_T_B = comm.bcast(B, root=0)

multiplied_row = matrix_multiplication(recv_chunk_A, recv_chunk_T_B)
# gathering Data
data = comm.gather(multiplied_row, root=root)
if rank == 0:
    gathered_data = np.vstack(data)
    print('\n data:', gathered_data)
    seq_res = np.dot(A, B)
    print('\n seq_res:', seq_res)

end_time = MPI.Wtime()

Net_time = end_time - start_time
print('Total Time for process: rank=%.0f is Net_time=%.5f' % (rank, Net_time))
