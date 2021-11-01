# MIT 6.034 Lab 5: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    agenda = list(net.get_parents(var))
    ancestors = []
    while agenda:
        cur_ancestors = agenda.pop(0)
        for ancestor in cur_ancestors:
            ancestors += ancestor
            if net.get_parents(ancestor):
                agenda = list(net.get_parents(ancestor)) + agenda

    return set(ancestors)

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    agenda = list(net.get_children(var))
    children = []
    while agenda:
        cur_children = agenda.pop(0)
        for child in cur_children:
            children += child
            if net.get_parents(child):
                agenda = list(net.get_children(child)) + agenda

    return set(children)

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    all_var = net.get_variables()
    children = get_descendants(net, var)
    nondescendants = []
    for var_ in all_var:
        if (var_ not in children) and (var_ != var):
            nondescendants.append(var_)
    return set(nondescendants)

#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    givens_key = set(givens.keys())
    parents = net.get_parents(var)
    
    descendants = net.get_children(var)
    if (parents.issubset(givens_key)) and (descendants.isdisjoint(givens_key)):
        new_givens = {k:v for k,v in givens.items() if k in parents}
        return new_givens
    else:
        return givens

    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    if not hypothesis:
        return None
    # Supposse we only deal with hypothesis with one variable
    vars = list(hypothesis.keys())
    if len(vars) > 1:
        raise LookupError("Hypothesis has multiple variables.")
    else:
        var = vars[0]
    try:
        if givens:
            givens = simplify_givens(net, var, givens)
            probability = net.get_probability(hypothesis, parents_vals=givens, infer_missing=True)
        else:
            probability = net.get_probability(hypothesis, parents_vals=None, infer_missing=True)
        return probability 

    except ValueError as e:
        raise LookupError(e)
    # get_probability(hypothesis, parents_vals=None, infer_missing=True)
    


def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    vars = list(hypothesis.keys())
    sorted_vars = net.topological_sort(vars)

    joint_prob = 1
    for i in range(len(sorted_vars)):
        var = sorted_vars[-1-i]
        givens = sorted_vars[:-1-i]

        givens_dict = {k:v for k,v in hypothesis.items() if k in givens}
        hypo_dict = {var: hypothesis[var]}

        prob = probability_lookup(net, hypo_dict, givens=givens_dict)
        joint_prob = joint_prob * prob
    return joint_prob

    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    all_vars = set(net.get_variables())
    vars = set(hypothesis.keys())
    other_vars = list(all_vars - vars) 
    other_var_comb = net.combinations(other_vars) 
    marginal_prob = 0
    for comb in other_var_comb:
        new_hypo = hypothesis.copy()
        new_hypo.update(comb)
        prob = probability_joint(net, new_hypo)
        marginal_prob += prob
    return marginal_prob


def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if not givens:
        return probability_marginal(net, hypothesis)
    
    hypo_keys = set(hypothesis.keys())
    given_keys = set(givens.keys())
    if not hypo_keys.isdisjoint(given_keys):
        intersects = list(hypo_keys.intersection(given_keys))
        for intersect in intersects:
            if hypothesis[intersect] != givens[intersect]:
                return 0
            else:
                hypothesis.pop(intersect)
                givens.pop(intersect)

    hypothesis.update(givens)
    prob1 = probability_marginal(net, hypothesis)
    prob2 = probability_marginal(net, givens)
    return prob1/prob2    


    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    all_vars = set(net.get_variables())
    vars = set(hypothesis.keys())
    if givens:
        return probability_conditional(net, hypothesis, givens)
    elif vars == all_vars:
        return probability_joint(net, hypothesis)
    else:
        return probability_marginal(net, hypothesis)
    


#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    num_params_list = []
    all_vars = net.get_variables()
    for var in all_vars: 
        domain_size = len(net.get_domain(var))-1
        num_params = domain_size 
        for parent in net.get_parents(var):
            parent_domain_size  = len(net.get_domain(parent))
            num_params *= parent_domain_size
        num_params_list.append(num_params)
        # print(var, num_params)
    return sum(num_params_list)



#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    for domain1 in net.get_domain(var1):
        for domain2 in net.get_domain(var2):
            if not givens:
                prob1 = probability(net, {var1: domain1},  {var2: domain2})
                prob2 = probability(net,{var1:domain1},None)
            else:
                prob2 = probability(net,{var1:domain1},givens.copy())
                givens.update({var2: domain2})
                prob1 = probability(net, {var1: domain1}, givens)
                
            if not approx_equal(prob1, prob2):
                return False
    return True
    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    # Simplify the net
    mentioned_vars = {var1, var2}
    ancestors1 = get_ancestors(net, var1)
    ancestors2 = get_ancestors(net,var2)
    mentioned_vars.update(ancestors1)
    mentioned_vars.update(ancestors2)

    if givens:
        givens_vars = set(givens.keys())
        mentioned_vars.update(givens_vars)

        for given in givens:
            if net.find_path(var1,given) != None:
                mentioned_vars.update(net.find_path(var1,given))
            if net.find_path(var2,given) != None:
                mentioned_vars.update(net.find_path(var2,given))

    # mentioned_vars = set(net.topological_sort(list(mentioned_vars)))
    subnet_ = net.subnet(mentioned_vars)
    subnet_sorted_vars = subnet_.topological_sort()
    
    # Build links between parents
    linked_parents = []
    subnet_copy = subnet_.copy()
    for i in range(len(subnet_sorted_vars)):
        child = subnet_sorted_vars[-1-i]
        parents = list(subnet_copy.get_parents(child))
        if len(parents) > 0:
            for m in range(len(parents)-1):
                for n in range(m, len(parents)):
                    parent_pair = [parents[m], parents[n]]
                    if parent_pair not in linked_parents: 
                        subnet_.link(parents[m], parents[n])
                        linked_parents.append(parent_pair)

    # Unorient the net
    subnet_.make_bidirectional()

    # Remove givens
    if givens:
        for given in givens:
             subnet_.remove_variable(given)

    if subnet_.find_path(var1, var2):
        return False
    else:
        return True

#### SURVEY ####################################################################

NAME =  "Xiaohan Yang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = "around 8 hours"
WHAT_I_FOUND_INTERESTING = "I found part 3 and part 4 very helpful. They helped me to understand the relating concepts "
WHAT_I_FOUND_BORING = "I think this lab is well defined. Nothing is confusing or boring."
SUGGESTIONS = ""
