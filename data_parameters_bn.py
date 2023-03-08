# This script is based on pgmpy
# pgmpy is a python library for working with Probabilistic Graphical Models
# For more information: https://github.com/pgmpy/pgmpy


from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.readwrite import BIFReader
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import ApproxInference

from pgmpy.metrics import log_likelihood_score
from pgmpy.sampling import BayesianModelInference
from itertools import chain
from pgmpy.models import JunctionTree
import math
import networkx as nx
import numpy as np
import pandas as pd
import re

class BayesianModelProbability(BayesianModelInference):
    """
    Class to calculate probability (pmf) values specific to Bayesian Models
    """

    def __init__(self, model):
        """
        Class to calculate probability (pmf) values specific to Bayesian Models

        Parameters
        ----------
        model: Bayesian Model
            model on which inference queries will be computed
        """

        # super().__init__(model)
        super().__init__(model)
        from pgmpy.models import BayesianNetwork
        if not isinstance(model, BayesianNetwork):
            raise TypeError(
                f"Model expected type: BayesianNetwork, got type: {type(model)}"
            )

        self.model = model
        if isinstance(self.model, JunctionTree):
            self.variables = set(chain(*self.model.nodes()))
        else:
            self.variables = self.model.nodes()

        self._initialize_structures()
        self.topological_order = list(nx.topological_sort(model))

        # super(BayesianModelProbability, self).__init__(model)

    def _log_probability_node(self, data, ordering, node):
        """
        Evaluate the log probability of each datapoint for a specific node.

        Internal function used by log_probability().

        Parameters
        ----------
        data: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        ordering: list
            ordering of columns in data, used by the Bayesian model.
            default is topological ordering used by model.

        node: Bayesian Model Node
            node from the Bayesian network.

        Returns
        -------
        Log probability of node: np.array (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """

        def vec_translate(a, my_dict):
            return np.vectorize(my_dict.__getitem__)(a)

        cpd = self.model.get_cpds(node)

        # variable to probe: data[n], where n is the node number
        current = cpd.variables[0]
        current_idx = ordering.index(current)
        current_val = data[:, current_idx]
        current_no = vec_translate(current_val, cpd.name_to_no[current])

        # conditional dependencies E of the probed variable
        evidence = cpd.variables[:0:-1]
        evidence_idx = [ordering.index(ev) for ev in evidence]
        evidence_val = data[:, evidence_idx]
        evidence_no = np.empty_like(evidence_val, dtype=int)
        for i, ev in enumerate(evidence):
            evidence_no[:, i] = vec_translate(evidence_val[:, i], cpd.name_to_no[ev])

        if evidence:
            # there are conditional dependencies E for data[n] for this node
            # Here we retrieve the array: p(x[n]|E). We do this for each x in data.
            # We pick the specific node value from the arrays below.

            state_to_index, index_to_weight = self.pre_compute_reduce_maps(
                variable=node
            )
            unique, inverse = np.unique(evidence_no, axis=0, return_inverse=True)
            weights = np.array(
                [index_to_weight[state_to_index[tuple(u)]] for u in unique]
            )[inverse]
        else:
            # there are NO conditional dependencies for this node
            # retrieve array: p(x[n]).  We do this for each x in data.
            # We pick the specific node value from the arrays below.
            weights = np.array([cpd.values] * len(data))

        # pick the specific node value x[n] from the array p(x[n]|E) or p(x[n])
        # We do this for each x in data.
        probability_node = np.array([weights[i][cn] for i, cn in enumerate(current_no)])

        return np.log(probability_node)

    def log_probability(self, data, ordering=None):
        """
        Evaluate the logarithmic probability of each point in a data set.

        Parameters
        ----------
        data: pandas dataframe OR array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        ordering: list
            ordering of columns in data, used by the Bayesian model.
            default is topological ordering used by model.

        Returns
        -------
        Log probability of each datapoint: np.array (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        if isinstance(data, pd.DataFrame):
            # use numpy array from now on.
            ordering = data.columns.to_list()
            data = data.values
        if ordering is None:
            ordering = self.topological_order
            data = data.loc[:, ordering].values

        logp = np.array(
            [self._log_probability_node(data, ordering, node) for node in ordering]
        )
        return np.sum(logp, axis=0)

    def score(self, data, ordering=None):
        """
        Compute the total log probability density under the model.

        Parameters
        ----------
        data: pandas dataframe OR array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        ordering: list
            ordering of columns in data, used by the Bayesian model.
            default is topological ordering used by model.

        Returns
        -------
        Log-likelihood of data: float
            This is normalized to be a probability density, so the value
            will be low for high-dimensional data.
        """

        log_prob = self.log_probability(data, ordering)
        return log_prob
        # return np.sum(self.log_probability(data, ordering))


def kl_divergence(p, q):

    """
    Compute the KL divergence between discrete probability distributions p and q.
    It is the amount of information lost when q is used to approximate p.
    p typically represents the "true" distribution of data,while q typically represents approximation of p

    :param p: discrete probability distribution
    :param q: discrete probability distribution
    :return: KL divergence "score"
    """

    return np.sum(np.where(p != 0, p * np.log2(p / q), 0))


def generate_data(ans, filename, output_file, num_tuples):

    """
    This functions generate data based on BNs,using forward_sampling method

    :param ans: internal parameter
    :param filename: the input file or name of BN
    :param output_file:  the name of file where the data will be written
    :param num_tuples: the size of dataset
    :return: 0 on success , -1 on failure
    """

    if ans == 1:
        # Read the file
        reader = BIFReader(filename)

        # Get the model from file
        model = reader.get_model()
    else:
        # Load the model
        model = get_example_model(filename)

    # Check model
    if model.check_model():
        print("CPDs of model are valid")
    else:
        return -1

    # Print the nodes and edges from BN
    print("Nodes in the model:", model.nodes())
    print("Edges in the model:", model.edges())

    # Get the CPDs of model
    cpds = model.get_cpds()

    # Print the CPDs of model
    for cpd in cpds:
        print(cpd)

    # Get the cardinalities of nodes
    cards_nodes = model.get_cardinality()

    # Print the cardinalities of nodes
    print("Cardinalities of nodes")
    for key, value in cards_nodes.items():
        print(key, ":", value)

    # Generate data based on BN using forward sampling
    # or model.simulate(n_samples=num_tuples)
    samples = BayesianModelSampling(model).forward_sample(size=num_tuples)

    # Write samples to output_file
    # mode = 'a' for cardinality
    samples.to_csv(output_file, header=False, index=False)

    return 0


def generate_data_np(ans, filename, output_file, num_tuples):

    """
    This functions generate data based on BNs,using forward_sampling method

    :param ans: internal parameter
    :param filename: the input file or name of BN
    :param output_file:  the name of file where the data will be written
    :param num_tuples: the size of dataset
    :return: 0 on success , -1 on failure
    """

    if ans == 1:
        # Read the file
        reader = BIFReader(filename)

        # Get the model from file
        model = reader.get_model()
    else:
        # Load the model
        model = get_example_model(filename)

    # Check model
    if model.check_model():
        print("CPDs of model are valid")
    else:
        return -1

    # Generate data based on BN using forward sampling
    # or model.simulate(n_samples=num_tuples)
    samples = BayesianModelSampling(model).forward_sample(size=num_tuples)

    # Write samples to output_file
    # mode = 'a' for cardinality
    samples.to_csv(output_file, header=True, index=False)
    print("\n")

    return 0


def learning_parameter(choice, filename, num_tuples):

    """
    Learning the model parameters using Maximum Likelihood Estimate(MLE)

    :param choice: internal parameter
    :param filename: the input file or name of BN
    :param num_tuples: the size of dataset
    :return: 0 on success , -1 on failure
    """
    np.set_printoptions(linewidth=np.inf)

    if choice == "I":
        # Read the file
        reader = BIFReader(filename)

        # Get the model from file
        model = reader.get_model()
    else:
        # Load the model
        model = get_example_model(filename)

    # Check model
    if model.check_model():
        print("CPDs of model are valid")
    else:
        return -1

    # Print the nodes and edges from BN
    print("Nodes in the model:", model.nodes())
    print("Edges in the model:", model.edges())

    # Generate data using forward sampling
    data = BayesianModelSampling(model).forward_sample(size=num_tuples)

    # Parameter estimator
    par_est = ParameterEstimator(model, data)

    # Print the state counts for each variable of model
    print("==================State counts==================")
    for node in model.nodes():
        print("\n", par_est.state_counts(node))
    print("================================================\n\n")

    # Fitting the model using Maximum Likelihood Estimator
    # or using the method fit => model.fit(data,estimator=MaximumLikelihoodEstimator)
    mle = MaximumLikelihoodEstimator(model, data)

    # Estimate the CPD of each node
    print("Estimated CPDs")
    for node in model.nodes():
        print(mle.estimate_cpd(node))

    # Get all the parameters
    cpds = mle.get_parameters()

    # Print the CPDs(same as estimated CPDs)
    print("\n\nParameters of model")
    for cpd in cpds:
        print(cpd)

    # Correctness check of parameters
    # Print the differences between true parameters and learned parameters
    print("\n")
    for cpd in cpds:

        # Print the name of node
        print("Node", cpd.variable)

        # Print flattened array(row-major)
        print("True parameters : ", np.flipud(model.get_cpds(cpd.variable).values.flatten()))  # flip the array vertically so the values of arrays are matched
        print("Learned parameters : ", cpd.values.flatten())

        # Print the KL-divergence between true CPD and learned CPD
        print("KL(T||L) = %1.3f" % kl_divergence(np.flipud(model.get_cpds(cpd.variable).values.flatten()), cpd.values.flatten()))

        # Print the differences(true-learned) of arrays
        print("Differences: ", np.subtract(np.flipud(model.get_cpds(cpd.variable).values.flatten()), cpd.values.flatten()), "\n")

    return 0


def estimate_gt_mle_prob(choice, filename, num_tuples, query_size):

    """
    Estimate the joint probability using the ground truth parameters and parameters which learnt from MLE

    :param choice: internal parameter
    :param filename: the input file or name of BN
    :param num_tuples: the size of dataset
    :param query_size: the size of queries
    :return: 0 on success , -1 on failure
    """
    np.set_printoptions(linewidth=np.inf)

    if choice == "I":
        # Read the file
        reader = BIFReader(filename)

        # Get the model from file
        model = reader.get_model()
    else:
        # Load the model
        model = get_example_model(filename)

    # Check model
    if model.check_model():
        print("CPDs of model are valid")
    else:
        return -1

    # Print the nodes and edges from BN
    print("Nodes in the model:", model.nodes())
    print("Edges in the model:", model.edges())

    # Generate data using forward sampling
    data = BayesianModelSampling(model).forward_sample(size=num_tuples)

    # Parameter estimator
    # par_est = ParameterEstimator(model, data)

    # Print the state counts for each variable of model
    # print("==================State counts==================")
    # for node in model.nodes():
    #    print("\n", par_est.state_counts(node))
    # print("================================================\n\n")

    # Fitting the model using Maximum Likelihood Estimator
    # or using the method fit => model.fit(data,estimator=MaximumLikelihoodEstimator)
    mle = MaximumLikelihoodEstimator(model, data)

    # Get all the parameters
    cpds = mle.get_parameters()

    # Correctness check of parameters
    # Print the differences between true parameters and learned parameters
    print("\n")
    for cpd in cpds:

        # Print the name of node
        print("Node", cpd.variable)

        # Print flattened array(row-major)
        print("True parameters : ", np.flipud(model.get_cpds(cpd.variable).values.flatten()))  # flip the array vertically so the values of arrays are matched
        print("Learned parameters : ", cpd.values.flatten())

        # Print the KL-divergence between true CPD and learned CPD
        # print("KL(T||L) = %1.3f" % kl_divergence(np.flipud(model.get_cpds(cpd.variable).values.flatten()), cpd.values.flatten()))

        # Print the differences(true-learned) of arrays
        print("Differences: ", np.subtract(np.flipud(model.get_cpds(cpd.variable).values.flatten()), cpd.values.flatten()), "\n")

    # Log likelihood of a given dataset/queries
    queries = BayesianModelSampling(model).forward_sample(size=query_size)
    queries.to_csv("querySource_"+filename, header=False, index=False)
    # queries = queries.drop(['Burglary', 'Alarm', 'JohnCalls', 'MaryCalls'], axis=1)

    # Ground truth log likelihood
    gt_log_likelihood = BayesianModelProbability(model).score(queries)
    print("Ground truth Log likelihood :", gt_log_likelihood)
    print("Average Ground truth Log likelihood :", np.sum(gt_log_likelihood)/gt_log_likelihood.shape[0])
    print("Count of log(prob) > log(0.01) : ", np.count_nonzero(gt_log_likelihood > math.log(0.01)))
    # print("Ground truth Log likelihood score :", log_likelihood_score(model, queries))

    # Estimated log likelihood
    # Adjust all cpds from model with the estimated cpds
    # Remove all cpds from the model
    model_cpds = model.get_cpds().copy()
    for cpd in model_cpds:
        model.remove_cpds(cpd)

    # Add all cpds from MLE
    for cpd in cpds:
        model.add_cpds(cpd)

    est_log_likelihood = BayesianModelProbability(model).score(queries)
    print("Estimated Log likelihood :", est_log_likelihood)
    # print("Estimated Log likelihood score :", log_likelihood_score(model, queries))

    # Print the differences between them and get the average
    diff = np.abs(est_log_likelihood-gt_log_likelihood)
    print("Absolute error :", diff)
    diff[~np.isfinite(diff)] = -1
    diff = diff[diff > 0]
    print("Queries size included on average :", diff.shape[0])
    print("Average Absolute error :", np.sum(diff)/diff.shape[0])

    # Write statistics about querySource to file
    f = open("querySource_"+filename+"_stats", "w")
    f.write("Training dataset : " + str(num_tuples)+"\n")
    f.write("Query size : " + str(query_size)+"\n")
    f.write("\n\nGround truth Log likelihood : " + str(gt_log_likelihood))
    f.write("\nAverage Ground truth Log likelihood : " + str(np.sum(gt_log_likelihood)/gt_log_likelihood.shape[0]))
    f.write("\nCount of log(prob) > log(0.01) : " + str(np.count_nonzero(gt_log_likelihood > math.log(0.01))))
    f.write("\n\nEstimated Log likelihood : " + str(est_log_likelihood))
    f.write("\n\nAbsolute error : " + str(diff))
    f.write("\n\nQueries size included on average : " + str(diff.shape[0]))
    f.write("\nAverage Absolute error : " + str(np.sum(diff) / diff.shape[0]))
    f.close()

    # Estimate query using learning parameters
    # query = "False,False,False,True,True"
    # q = infer.query(variables=["Burglary","Earthquake","Alarm"],n_samples=1000)
    # print(q)
    infer = ApproxInference(model)
    # Probability distribution of Earthquake
    # dist = infer.get_distribution(data, variables=['Earthquake'])
    # print(dist)

    # Probability distribution of Bulglary
    # dist = infer.get_distribution(data, variables=['Burglary'])
    # print(dist)

    # Probability distribution of Bulglary,Earthquake,Alarm
    # dist = infer.get_distribution(data, variables=['Burglary','Earthquake','Alarm'],joint=True)
    # print(dist)

    # Probability distribution of all nodes of BN
    dist = infer.get_distribution(data, variables=mle.variables, joint=True)
    for x in dist:
        print(dist.get(x))

    return 0


def convert_bn_to_string(choice, filename):

    """
    Convert a Bayesian Network to String with the follow format:\n
    (Node_Name:Cardinality_of_Node[Values] | Parents) for every node in BN \n

    Node_Name : the name of node \n
    Cardinality_of_Node : the cardinality of node \n
    Values : Correspond to comma-separated list of node values \n
    Parents : Correspond to comma-separated list of node parents where every node depicted as Node_Name:Cardinality_of_Node[Values] \n

    :param choice: internal parameter
    :param filename: the input file or name of BN
    :return: The transformed string
    """

    if choice == "I":
        # Read the file
        reader = BIFReader(filename)

        # Get the model from file
        model = reader.get_model()
    else:
        # Load the model
        model = get_example_model(filename)

    # Create the string
    strBN = ""
    for cpd in model.get_cpds():

        strBN += "( " + cpd.variable + ":" + str(model.get_cardinality(cpd.variable))

        # Get states of variable
        states = "[ "
        for state in cpd.state_names.get(cpd.variable):
            states += state + ","

        # Appended to str
        states += " ]"
        states = states.replace(", ]", " ]")
        strBN += states

        # Check for parents
        if len(cpd.get_evidence()) > 0:
            strBN += " | "
            for par in cpd.get_evidence():
                strBN += par + ":" + str(model.get_cardinality(par))

                # Get states of variable
                states = "[ "
                for state in cpd.state_names.get(par):
                    states += state + ","

                # Appended to str
                states += " ]"
                states = states.replace(", ]", " ]")
                strBN += states + ","

        strBN += " )\n"

    # Return string
    strBN = strBN.replace(", )", " )")
    print(strBN)
    return strBN


def convert_bn_to_json(choice, filename):

    """
    Convert a Bayesian Network to JSON string with the follow format:\n
    {"name":node_name,"trueParameters":trueParameters,"cardinality":cardinality_of_Node,"values":[values],parents":[{node}]} for every node in BN \n

    node_Name : the name of node \n
    trueParameters : the true parameters of node \n
    cardinality_of_Node : the cardinality of node \n
    values : Correspond to comma-separated list of node values \n
    parents : Correspond to comma-separated list of node parents where every node depicted as {"name":node_name,"cardinality":cardinality_of_Node,"values":[values],"parents":[]} \n

    :param choice: internal parameter
    :param filename: the input file or name of BN
    :return: The transformed string
    """

    if choice == "I":
        # Read the file
        reader = BIFReader(filename)

        # Get the model from file
        model = reader.get_model()
    else:
        # Load the model
        model = get_example_model(filename)

    # Create the json string
    strBN = "\"["
    for cpd in model.get_cpds():

        # Get the parameters
        trueParameters = ",\\\"trueParameters\\\":["
        if len(cpd.get_evidence()) > 0:
            for value in cpd.values:
                tmp_var = str(value)
                tmp_var = "\""+tmp_var.replace("\n", "").replace("[", "").replace("]", "").strip().replace(" ", ",")
                tmp_var = re.sub(",+", ",", tmp_var).strip()+"\","
                trueParameters += tmp_var
                # trueParameters += "\\\""+str(np.hstack(value).tolist()).replace(" ", "").replace("[", "").replace("]", "")+"\\\","
        else:
            # No parents exists
            for i in range(len(cpd.values)):
                trueParameters += "\\\""+str(cpd.values[i])+"\\\","

        trueParameters += "]"
        trueParameters = trueParameters.replace(",]", "]")

        strBN += "{\\\"name\\\":" + str("\\\""+cpd.variable+"\\\"") + trueParameters + "," + "\\\"cardinality\\\":" + str(model.get_cardinality(cpd.variable))

        # Get states of variable
        states = ",\\\"values\\\":["
        for state in cpd.state_names.get(cpd.variable):
            states += str("\\\""+state+"\\\"") + ","

        # Appended to str
        states += "]"
        states = states.replace(",]", "]")
        strBN += states

        # Check for parents
        strBN += ",\\\"parents\\\":["
        if len(cpd.get_evidence()) > 0:
            for par in cpd.get_evidence()[::-1]:
                strBN += "{\\\"name\\\":"+str("\\\""+par+"\\\"") + "," + "\\\"cardinality\\\":" + str(model.get_cardinality(par))

                # Get states of variable
                states = ",\\\"values\\\":["
                for state in cpd.state_names.get(par):
                    states += str("\\\""+state+"\\\"") + ","

                # Appended to str
                states += "]"
                states = states.replace(",]", "]")
                states += ",\\\"parents\\\":[]"
                strBN += states + "},"

        strBN += " "
        strBN = strBN.replace("}, ", "}")
        strBN += "] },"

    # Return string
    strBN += " "
    strBN = strBN.replace("}, ", "}")
    strBN += "]\""

    print(strBN)
    return strBN


def convert_bn_to_json_obj(choice, filename):

    """
    Convert a Bayesian Network to JSON string with the follow format:\n
    {"name":node_name,"trueParameters":trueParameters,"cardinality":cardinality_of_Node,"values":[values],parents":[{node}]} for every node in BN \n

    node_Name : the name of node \n
    trueParameters : the true parameters of node \n
    cardinality_of_Node : the cardinality of node \n
    values : Correspond to comma-separated list of node values \n
    parents : Correspond to comma-separated list of node parents where every node depicted as {"name":node_name,"cardinality":cardinality_of_Node,"values":[values],"parents":[]} \n

    :param choice: internal parameter
    :param filename: the input file or name of BN
    :return: The transformed string
    """

    if choice == "I":
        # Read the file
        reader = BIFReader(filename)

        # Get the model from file
        model = reader.get_model()
    else:
        # Load the model
        model = get_example_model(filename)

    # Create the json string
    strBN = "["
    
    for cpd in model.get_cpds():

        # Get the parameters
        trueParameters = ",\"trueParameters\":["
        if len(cpd.get_evidence()) > 0:
            for value in cpd.values:
                tmp_var = str(value)
                tmp_var = "\""+tmp_var.replace("\n", "").replace("[", "").replace("]", "").strip().replace(" ", ",")
                tmp_var = re.sub(",+", ",", tmp_var).strip()+"\","
                trueParameters += tmp_var
                # trueParameters += "\""+str(np.hstack(value).tolist()).replace(" ", "").replace("[", "").replace("]", "")+"\","
        else:
            # No parents exists
            for i in range(len(cpd.values)):
                trueParameters += "\""+str(cpd.values[i])+"\","

        trueParameters += "]"
        trueParameters = trueParameters.replace(",]", "]")

        # Append name,trueParameters and cardinality to string
        strBN += "{\"name\":" + str("\""+cpd.variable+"\"") + trueParameters + "," + "\"cardinality\":" + str(model.get_cardinality(cpd.variable))

        # Get states of variable
        states = ",\"values\":["
        for state in cpd.state_names.get(cpd.variable):
            states += str("\""+state+"\"") + ","

        # Appended to str
        states += "]"
        states = states.replace(",]", "]")
        strBN += states

        # Check for parents
        strBN += ",\"parents\":["
        if len(cpd.get_evidence()) > 0:
            for par in cpd.get_evidence()[::-1]:
                strBN += "{\"name\":"+str("\""+par+"\"") + "," + "\"cardinality\":" + str(model.get_cardinality(par))

                # Get states of variable
                states = ",\"values\":["
                for state in cpd.state_names.get(par):
                    states += str("\""+state+"\"") + ","

                # Appended to str
                states += "]"
                states = states.replace(",]", "]")
                states += ",\"parents\":[]"
                strBN += states + "},"

        strBN += " "
        strBN = strBN.replace("}, ", "}")
        strBN += "] },\n"

    # Return string
    strBN += " "
    strBN = strBN.replace("},\n ", "}")
    strBN += "]"

    print(strBN)
    return strBN


def print_menu():
    while 1:
        try:
            print("==========================================\n"
                  + "1:Generate data for BNs\n"
                  + "2:Generate data for built-in BNs\n"
                  + "3:Learning the model parameters using MLE\n"
                  + "4:Convert BN to string\n"
                  + "5:Convert BN to JSON object as string\n"
                  + "6:Convert BN to JSON object\n"
                  + "7:Estimate the log of probability of joint distribution(GT,MLE)\n"
                  + "8:Exit the program\n==========================================\n")
            ans = int(input("Give your preference: "))
            if 1 <= ans <= 8:
                if ans == 1:
                    print("The input file must have .BIF extension")
                    file = str(input("Give your input file path: "))
                    output_file = str(input("Give your output file path: "))
                    num_tuples = int(input("Give the size of dataset: "))
                    if generate_data(ans, file, output_file, num_tuples) == -1:
                        print("Invalid Bayesian Network")
                    print("\n\n")

                elif ans == 2:
                    print("Contain any model from bnlearn repository (http://www.bnlearn.com/bnrepository)")
                    name_model = str(input("Give the name of model:")).strip()
                    output_file = str(input("Give your output file path: "))
                    num_tuples = int(input("Give the size of dataset: "))
                    for i in range(1):
                        if generate_data_np(ans, name_model.lower(), "data_"+output_file+str(i)+"_"+str(num_tuples), num_tuples) == -1:
                            print("Invalid Bayesian Network")
                    print("\n\n")

                elif ans == 3:
                    print("Import BN(I) or Example BN(E)")
                    choice = str(input("Give your preference(I or E): "))

                    if choice == "I":
                        print("The input file must have .BIF extension")
                        filename = str(input("Give your input file path: "))
                    elif choice == "E":
                        print("Contain any model from bnlearn repository (http://www.bnlearn.com/bnrepository)")
                        filename = str(input("Give the name of model:")).strip().lower()
                    else:
                        print("Invalid option\n\n")
                        return

                    num_tuples = int(input("Give the size of dataset for training: "))
                    if learning_parameter(choice, filename, num_tuples) == -1:
                        print("Invalid Bayesian Network")
                    print("\n\n")

                elif ans == 4:
                    print("Import BN(I) or Example BN(E)")
                    choice = str(input("Give your preference(I or E): "))

                    if choice == "I":
                        print("The input file must have .BIF extension")
                        filename = str(input("Give your input file path: "))
                    elif choice == "E":
                        print("Contain any model from bnlearn repository (http://www.bnlearn.com/bnrepository)")
                        filename = str(input("Give the name of model:")).strip().lower()
                    else:
                        print("Invalid option\n\n")
                        return

                    strBN = convert_bn_to_string(choice, filename)
                    # Write to file
                    f = open("bn" + ("_"+filename if choice == "E" else "") + ".txt", "w")
                    f.write(strBN)
                    f.close()

                    print("\n\n")

                elif ans == 5:
                    print("Import BN(I) or Example BN(E)")
                    choice = str(input("Give your preference(I or E): "))

                    if choice == "I":
                        print("The input file must have .BIF extension")
                        filename = str(input("Give your input file path: "))
                    elif choice == "E":
                        print("Contain any model from bnlearn repository (http://www.bnlearn.com/bnrepository)")
                        filename = str(input("Give the name of model:")).strip().lower()
                    else:
                        print("Invalid option\n\n")
                        return

                    strBN = convert_bn_to_json(choice, filename)
                    # Write to file
                    f = open("bn_JSON" + ("_"+filename if choice == "E" else "") + ".txt", "w")
                    f.write(strBN)
                    f.close()

                elif ans == 6:
                    print("Import BN(I) or Example BN(E)")
                    choice = str(input("Give your preference(I or E): "))

                    if choice == "I":
                        print("The input file must have .BIF extension")
                        filename = str(input("Give your input file path: "))
                    elif choice == "E":
                        print("Contain any model from bnlearn repository (http://www.bnlearn.com/bnrepository)")
                        filename = str(input("Give the name of model:")).strip().lower()
                    else:
                        print("Invalid option\n\n")
                        return

                    strBN = convert_bn_to_json_obj(choice, filename)
                    # Write to file
                    f = open("bn_JSON" + ("_"+filename if choice == "E" else "") + ".json", "w")
                    f.write(strBN)
                    f.close()

                elif ans == 7:
                    print("Import BN(I) or Example BN(E)")
                    choice = str(input("Give your preference(I or E): "))

                    if choice == "I":
                        print("The input file must have .BIF extension")
                        filename = str(input("Give your input file path: "))
                    elif choice == "E":
                        print("Contain any model from bnlearn repository (http://www.bnlearn.com/bnrepository)")
                        filename = str(input("Give the name of model:")).strip().lower()
                    else:
                        print("Invalid option\n\n")
                        return

                    num_tuples = int(input("Give the size of dataset for training: "))
                    query_size = int(input("Give the size of queries for estimation: "))
                    if estimate_gt_mle_prob(choice, filename, num_tuples,query_size) == -1:
                        print("Invalid Bayesian Network")
                    print("\n\n")
                else:
                    return
        except (ValueError, IOError):
            pass


print_menu()
