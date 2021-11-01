# MIT 6.034 Lab 7: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x < threshold:
        return 0
    else:
        return 1

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+e**(-steepness*(x-midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    if x > 0:
        return x
    else:
        return 0

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5*(desired_output - actual_output)**2


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    lst = net.topological_sort() 
    
    outputs = {}
    for node in lst:
        inputs = net.get_incoming_neighbors(node)
        # outputs = net.get_outgoing_neighbors(node)
        weighted_sum = 0
        for input_ in inputs:
            wires = net.get_wires(startNode=input_, endNode=node)
            
            value = node_value(input_, input_values, outputs)
            weight = wires[0].get_weight()
            weighted_sum += weight*value
        out_value = threshold_fn(weighted_sum)
        outputs[node] = out_value

    output_neuron = net.get_output_neuron()
    output_neuron_value = outputs[output_neuron]

    return (output_neuron_value, outputs)


#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    best_inputs = [] #inputs.copy()
    max_output = -INF#func(best_inputs[0], best_inputs[1], best_inputs[2])

    possible_inputs = []
    for i in range(len(inputs)):
        cur_input = inputs.copy()[i]
        possible_input = [cur_input, cur_input+step_size, cur_input-step_size]
        possible_inputs.append(possible_input)
   
    for j in range(3):
        for k in range(3):
            for m in range(3):
                cur_inputs = [possible_inputs[0][j], possible_inputs[1][k], possible_inputs[2][m]]
                cur_val = func(possible_inputs[0][j], possible_inputs[1][k], possible_inputs[2][m])

                if max_output < cur_val:
                    max_output = cur_val
                    best_inputs = cur_inputs

    return (max_output, best_inputs)


def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    
    start_node = wire.startNode
    end_node = wire.endNode
    
    queue = [end_node]
    recorded = set()
    recorded.add(wire)
    recorded.add(start_node)
    
    # print(recorded)
    while queue:
        cur_node = queue.pop(0)
        if net.is_output_neuron(cur_node):
            recorded.add(cur_node)
            # return recorded 
        else:
            for i in net.get_outgoing_neighbors(cur_node):
                wire = net.get_wires(cur_node,i)[0]
                recorded.add(cur_node)
                recorded.add(wire)
                # recorded.add(i)
                
                queue.append(i) 
    
    return recorded

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    
    # get the reversed neurons so that I can compute from backward
    neurons = net.topological_sort()
    neurons.reverse()

    updated_values = {}
    for node in neurons:
        if net.is_output_neuron(node):
            value = neuron_outputs[node] * (1- neuron_outputs[node]) * (desired_output-neuron_outputs[node])
            updated_values[node] = value
            
        else:
            sum_w = 0
            for next_node in net.get_outgoing_neighbors(node):
                wire = net.get_wires(node,next_node)[0]
                weight = wire.get_weight()
                delta_next_node = updated_values[next_node]
                sum_w += weight*delta_next_node
            value = neuron_outputs[node] * (1- neuron_outputs[node]) * sum_w
            updated_values[node] = value
    
    return updated_values


def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    new_net = net.copy()
    wires = new_net.get_wires()
    deltas = calculate_deltas(new_net, desired_output, neuron_outputs)
    
    for wire in wires:
        start_node = wire.startNode
        end_node = wire.endNode
        weight = wire.get_weight()
        delta = deltas[end_node]
        out_value = node_value(start_node, input_values, neuron_outputs)
        new_weight = weight + r * out_value * delta
        wire.set_weight(new_weight)
    
    return new_net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    i = 0
    actual_output = forward_prop(net, input_values, threshold_fn=sigmoid)
    acc = accuracy(desired_output, actual_output[0])
    while acc < minimum_accuracy:
        i += 1
        net = update_weights(net, input_values, desired_output, actual_output[1], r=r)
        actual_output = forward_prop(net, input_values, threshold_fn=sigmoid)
        acc = accuracy(desired_output, actual_output[0])
       
    return (net, i)


#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 34
ANSWER_2 = 17
ANSWER_3 = 9
ANSWER_4 = 200
ANSWER_5 = 26

ANSWER_6 = 1
ANSWER_7 = "checkerboard"
ANSWER_8 = ["small","medium","large"]
ANSWER_9 = "B"

ANSWER_10 = "D"
ANSWER_11 = ["A", "C"]
ANSWER_12 = ["A", "E"]


#### SURVEY ####################################################################

NAME = "Xiaohan Yang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = "around 8 hours"
WHAT_I_FOUND_INTERESTING = "I found part 4 intersting as they helped me to understand the backward propagation step by step. "
WHAT_I_FOUND_BORING = "I think the last part is kind of boring. I don't think we need to play with the training.py so many times."
SUGGESTIONS = ""
