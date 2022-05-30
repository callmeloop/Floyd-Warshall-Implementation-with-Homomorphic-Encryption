from eva import EvaProgram, Input, Output, evaluate, save, load
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import timeit
import networkx as nx
import random
from floyd_warshall import floydWarshall
# from plot_results import plot_results

INF = 99999
MAX_WEIGHT = 10
VECTOR_SIZE = 4096 * 4

numberofnodes = 0


# Using networkx, generate a random graph
# You can change the way you generate the graph
def generateGraph(n, k, p):
    # ws = nx.cycle_graph(n)
    ws = nx.watts_strogatz_graph(n, k, p)
    return ws

# If there is an edge between two vertices its weight is 1 otherwise it is zero
# You can change the weight assignment as required
# Two dimensional adjacency matrix is represented as a vector
# Assume there are n vertices
# (i,j)th element of the adjacency matrix corresponds to (i*n + j)th element in the vector representations


def serializeGraphZeroOne(GG, vec_size):
    n = GG.size()
    graphdict = {}
    g = []
    for row in range(n):
        for column in range(n):
            if GG.has_edge(row, column):
                weight = GG[row][column]["weight"]
            elif row == column:  # I assumed the vertices are connected to themselves
                weight = 0
            else:
                weight = INF
            g.append(weight)
            key = str(row)+'-'+str(column)
            graphdict[key] = [weight]  # EVA requires str:listoffloat
    # EVA vector size has to be large, if the vector representation of the graph is smaller, fill the eva vector with zeros
    for i in range(vec_size - n*n):
        g.append(0.0)
    return g, graphdict

# To display the generated graph


def printGraph(graph, n):
    for row in range(n):
        for column in range(n):
            print("{:.2f}".format(graph[row*n+column]), end='\t')
        print()

# Eva requires special input, this function prepares the eva input
# Eva will then encrypt them


def prepareInput(n, m):
    input = {}
    GG = generateGraph(n, 3, 0.5)
    for (u, v) in GG.edges():
        GG.edges[u, v]['weight'] = random.randint(0, MAX_WEIGHT)
    graph, graphdict = serializeGraphZeroOne(GG, m)
    printGraph(graph, n)
    input['Graph'] = graph
    return input


def findMinDistanceSealVal(encOutputs, index1, index2, index3):
    secret_ctx = load('floyd.sealsecret')
    signature = load('floyd.evasignature')
    outputs = secret_ctx.decrypt(encOutputs, signature)
    encVal1 = encOutputs[index1]
    encVal2 = encOutputs[index2] + encOutputs[index3]
    val1 = outputs[index1]
    val2 = outputs[index2] + outputs[index3]
    return encVal1 if val1 <= val2 else encVal2

# This is the dummy analytic service
# You will implement this service based on your selected algorithm
# you can other parameters using global variables !!! do not change the signature of this function
# Note that you cannot compute everything using EVA/CKKS
# For instance, comparison is not possible
# You can add, subtract, multiply, negate, shift right/left
# You will have to implement an interface with the trusted entity for comparison (send back the encrypted values, push the trusted entity to compare and get the comparison output)


def graphanalticprogram(graph):
    dist = graph
    for k in range(numberofnodes):

        # pick all vertices as source one by one
        for i in range(numberofnodes):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(numberofnodes):

                findMinResult = findMinDistanceSealVal(dist, i * numberofnodes + j,
                                 i * numberofnodes + k, k * numberofnodes + j)

                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                dist[i * numberofnodes + j] = findMinResult

    return dist

# Do not change this
# the parameter n can be passed in the call from simulate function


class EvaProgramDriver(EvaProgram):
    def __init__(self, name, vec_size=4096, n=4):
        self.n = n
        super().__init__(name, vec_size)

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

# Repeat the experiments and show averages with confidence intervals
# You can modify the input parameters
# n is the number of nodes in your graph
# If you require additional parameters, add them


def simulate(n):
    m = VECTOR_SIZE
    print("Will start simulation for ", n, " nodes")
    config = {}
    config['warn_vec_size'] = 'false'
    config['lazy_relinearize'] = 'true'
    config['rescaler'] = 'always'
    config['balance_reductions'] = 'true'
    inputs = prepareInput(n, m)

    numberofnodes = n
    graph = inputs['Graph']
    floydWarshall(graph, n)  # to check correctness of the algorithm

    #################################################
    print('Compile time')
    graphanaltic = EvaProgramDriver("graphanaltic", vec_size=m, n=n)
    with graphanaltic:
         graph = Input('Graph')
         reval = graphanalticprogram(graph)
         Output('ReturnedValue', reval)
    prog = graphanaltic
    prog.set_output_ranges(30)
    prog.set_input_scales(30)
    start = timeit.default_timer()
    compiler = CKKSCompiler(config=config)
    compiled_multfunc, params, signature = compiler.compile(prog)
    compiletime = (timeit.default_timer() - start) * 1000.0  # ms
    save(prog, 'floyd.eva')
    save(params, 'floyd.evaparams')
    save(signature, 'floyd.evasignature')

    #################################################
    print('Key generation time')
    start = timeit.default_timer()
    public_ctx, secret_ctx = generate_keys(params)
    keygenerationtime = (timeit.default_timer() - start) * 1000.0  # ms
    save(public_ctx, 'floyd.sealpublic')
    save(secret_ctx, 'floyd.sealsecret')

    #################################################
    print('Runtime on client')
    start = timeit.default_timer()
    encInputs = public_ctx.encrypt(inputs, signature)
    encryptiontime = (timeit.default_timer() - start) * 1000.0  # ms
    save(encInputs, 'floyd_inputs.sealvals')

    #################################################
    print('Runtime on server')
    start = timeit.default_timer()
    encOutputs = public_ctx.execute(compiled_multfunc, encInputs)
    executiontime = (timeit.default_timer() - start) * 1000.0  # ms
    save(encOutputs, 'floyd_outputs.sealvals')

    #################################################
    print('Back on client')
    start = timeit.default_timer()
    outputs = secret_ctx.decrypt(encOutputs, signature)
    decryptiontime = (timeit.default_timer() - start) * 1000.0  # ms
    start = timeit.default_timer()
    reference = evaluate(compiled_multfunc, inputs)
    referenceexecutiontime = (timeit.default_timer() -
                               start) * 1000.0  # ms
    # Change this if you want to output something or comment out the two lines below
    # for key in outputs:
    #     print(key, float(outputs[key][0]), float(reference[key][0]))

    # since CKKS does approximate computations, this is an important measure that depicts the amount of error
    mse = valuation_mse(outputs, reference)

    return compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse


if __name__ == "__main__":
    simcnt = 3  # The number of simulation runs, set it to 3 during development otherwise you will wait for a long time
    # For benchmarking you must set it to a large number, e.g., 100
    # Note that file is opened in append mode, previous results will be kept in the file
    # Measurement results are collated in this file for you to plot later on
    resultfile = open("results.csv", "a")
    resultfile.write(
        "NodeCount,PathLength,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
    resultfile.close()

    print("Simulation campaing started:")
    for nc in range(36, 64, 4):  # Node counts for experimenting various graph sizes
        n = nc
        resultfile = open("results.csv", "a")
        for i in range(simcnt):
            # Call the simulator
            compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = simulate(
                n)
            res = str(n) + "," + str(i) + "," + str(compiletime) + "," + str(keygenerationtime) + "," + str(encryptiontime) + "," + str(executiontime) + "," + str(decryptiontime) + "," +  str(referenceexecutiontime) + "," +  str(mse) + "\n"
            print(res)
            resultfile.write(res)
        resultfile.close()

    # plot_results()
