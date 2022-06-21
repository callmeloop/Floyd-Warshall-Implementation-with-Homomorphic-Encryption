from eva import EvaProgram, Expr, Input, Output, evaluate, save, load
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import timeit
import random
import numpy as np
from floyd_warshall import floydWarshall

# from plot_results import plot_results

INF = 99999
MAX_WEIGHT = 10
VECTOR_SIZE = 0
NUMBEROFNODES = 0

# To display the generated graph
def printGraph(graph, n, vector_index):
    matrix = np.zeros((n,n))
    for i in range(n):
            for j in range(n):
                key = f"node_{i}_{j}"
                matrix[i,j] = graph[key][vector_index]

    np.set_printoptions(precision=2, suppress=True)
    print(matrix)

def floydWarshallWithoutFhe(graph, n, vector_index):
    matrix = np.zeros((n,n))
    for i in range(n):
            for j in range(n):
                key = f"node_{i}_{j}"
                matrix[i,j] = graph[key][vector_index]

    np.set_printoptions(precision=2, suppress=True)
    floydWarshall(matrix, n)


# Eva requires special input, this function prepares the eva input
# Eva will then encrypt them


def prepareInputs(n, m):
    inputs = {}

    for i in range(m):
        # Create a matrix of size (n,n) with random 0s and 1s.
        matrix = np.random.randint(2, size=(n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0
                elif matrix[i,j] == 0:
                    matrix[i, j] = INF
                else:
                    matrix[i, j] *= random.randint(0, MAX_WEIGHT)
                key = f"node_{i}_{j}"
                if key not in inputs.keys():
                    inputs[key] = [matrix[i, j]]
                else:
                    inputs[key].append(matrix[i, j])

    return inputs


# This is the dummy analytic service
# You will implement this service based on your selected algorithm
# you can other parameters using global variables !!! do not change the signature of this function
# Note that you cannot compute everything using EVA/CKKS
# For instance, comparison is not possible
# You can add, subtract, multiply, negate, shift right/left
# You will have to implement an interface with the trusted entity for comparison (send back the encrypted values, push the trusted entity to compare and get the comparison output)

def graphanalticprogram(graph, midVertex):
    # Create a numpy zeros matrix with Expr data type for input data
    matrix = np.zeros(shape=(2,NUMBEROFNODES, NUMBEROFNODES), dtype=Expr)
    for i in range(NUMBEROFNODES):
            for j in range(NUMBEROFNODES):
                key1 = f"node_{i}_{j}"
                matrix[0][i][j] = graph[key1]
                key2 = f"node_{i}_{midVertex}"
                key3 = f"node_{midVertex}_{j}"
                matrix[1][i][j] = graph[key2] + graph[key3]
            
    return matrix
   
# Do not change this
# the parameter n can be passed in the call from simulate function


class EvaProgramDriver(EvaProgram):
    def __init__(self, name, vec_size=4096, n=4):
        self.n = n
        super().__init__(name, vec_size)

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)


def floydWarshallFhe(inputs, config, midVertex):
    #################################################
    # Compile time
    graphanaltic = EvaProgramDriver(
        "graphanaltic", vec_size=VECTOR_SIZE, n=NUMBEROFNODES)
    with graphanaltic:
        input = {}
        for i in range(NUMBEROFNODES):
            for j in range(NUMBEROFNODES):
                key = f"node_{i}_{j}"
                input[key] = Input(key)

        reval = graphanalticprogram(input, midVertex)
        for i in range(NUMBEROFNODES):
            for j in range(NUMBEROFNODES):
                for k in range(2):
                    key = f"node_{k}_{i}_{j}"
                    Output(key, reval[k, i, j])

    prog = graphanaltic
    prog.set_output_ranges(30)
    prog.set_input_scales(30)
    start = timeit.default_timer()
    compiler = CKKSCompiler(config=config)
    compiled_multfunc, params, signature = compiler.compile(prog)
    compiletime = (timeit.default_timer() - start) * 1000.0  # ms

    #################################################
    # Key generation time
    start = timeit.default_timer()
    public_ctx, secret_ctx = generate_keys(params)
    keygenerationtime = (timeit.default_timer() - start) * 1000.0  # ms

    #################################################
    # Runtime on client
    start = timeit.default_timer()
    encInputs = public_ctx.encrypt(inputs, signature)
    encryptiontime = (timeit.default_timer() - start) * 1000.0  # ms

    #################################################
    # Runtime on server
    start = timeit.default_timer()
    encOutputs = public_ctx.execute(compiled_multfunc, encInputs)
    executiontime = (timeit.default_timer() - start) * 1000.0  # ms

    #################################################
    # Back on client
    start = timeit.default_timer()
    outputs = secret_ctx.decrypt(encOutputs, signature)
    decryptiontime = (timeit.default_timer() - start) * 1000.0  # ms
    start = timeit.default_timer()
    reference = evaluate(compiled_multfunc, inputs)
    referenceexecutiontime = (timeit.default_timer() -
                              start) * 1000.0  # ms

    # since CKKS does approximate computations, this is an important measure that depicts the amount of error
    mse = valuation_mse(outputs, reference)

    results = {'compiletime': compiletime,
               'keygenerationtime': keygenerationtime,
               'encryptiontime': encryptiontime,
               'executiontime': executiontime,
               'decryptiontime': decryptiontime,
               'referenceexecutiontime': referenceexecutiontime,
               'mse': mse
               }

    return outputs, results

# returns the minimum number from encrypted outputs
# needed in each floyd warshall iteration
def updateInputsFromMinDistance(distances):
    input = {}

    for v in range(VECTOR_SIZE):           
        for i in range(NUMBEROFNODES):
            for j in range(NUMBEROFNODES):
                key = f"node_{i}_{j}"
                distancekey1 = f"node_0_{i}_{j}"
                distancekey2 = f"node_1_{i}_{j}"
                if key not in input.keys():
                    input[key] = [min(distances[distancekey1][v], distances[distancekey2][v])]
                else :
                    input[key].append(min(distances[distancekey1][v], distances[distancekey2][v]))
    return input

# Repeat the experiments and show averages with confidence intervals
# You can modify the input parameters
# n is the number of nodes in your graph
# If you require additional parameters, add them


def simulate(n, vector_size):
    print("Will start simulation for ", n, " nodes")
    config = {}
    config['warn_vec_size'] = 'false'
    config['lazy_relinearize'] = 'true'
    config['rescaler'] = 'always'
    config['balance_reductions'] = 'true'
    
    global VECTOR_SIZE
    global NUMBEROFNODES
    NUMBEROFNODES = n
    VECTOR_SIZE = vector_size
    inputs = prepareInputs(NUMBEROFNODES, vector_size)
    print("INPUT GRAPH")
    printGraph(inputs, NUMBEROFNODES, 0)
    # Result without FHE:
    floydWarshallWithoutFhe(inputs, NUMBEROFNODES, 0)

    compiletime = 0
    keygenerationtime = 0
    encryptiontime = 0
    executiontime = 0
    decryptiontime = 0
    referenceexecutiontime = 0
    mse = 0
    for i in range(n):
        print("ITERATION ", i)
        outputs, results = floydWarshallFhe(inputs, config, i)
        compiletime += results['compiletime']
        keygenerationtime += results['keygenerationtime']
        encryptiontime += results['encryptiontime']
        executiontime += results['executiontime']
        decryptiontime += results['decryptiontime']
        referenceexecutiontime += results['referenceexecutiontime']
        mse = results['mse']
        inputs = updateInputsFromMinDistance(outputs)

    print("OUTPUT MATRIX")
    output = inputs
    printGraph(output, NUMBEROFNODES, 0)
    return compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse


if __name__ == "__main__":
    simcnt = 100  # The number of simulation runs, set it to 3 during development otherwise you will wait for a long time
    # For benchmarking you must set it to a large number, e.g., 100
    # Note that file is opened in append mode, previous results will be kept in the file
    # Measurement results are collated in this file for you to plot later on
    resultfile = open("results.csv", "a")
    resultfile.write(
        "NodeCount,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
    resultfile.close()

    print("Simulation campaign started:")
    for nc in range(2, 14, 4):  # Node counts for experimenting various graph sizes
        n = nc
        vectorsize = 1024
        resultfile = open("results.csv", "a")
        for i in range(simcnt):
            # Call the simulator
            compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = simulate(
                n, vectorsize)
            res = str(n) + "," + str(i) + "," + str(compiletime) + "," + str(keygenerationtime) + "," + str(encryptiontime) + \
                "," + str(executiontime) + "," + str(decryptiontime) + \
                "," + str(referenceexecutiontime) + "," + str(mse) + "\n"
            print(res)
            resultfile.write(res)
        resultfile.close()

    # plot_results()
