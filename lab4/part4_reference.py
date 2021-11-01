#### Part 4: Backward Chaining #########################################

# Import additional methods for backchaining
from production import PASS, FAIL, match, populate, simplify, variables


def backchain_to_goal_tree(rules, hypothesis):
    """
    Takes a hypothesis (string) and a list of rules (list
    of IF objects), returning an AND/OR tree representing the
    backchain of possible statements we may need to test
    to determine if this hypothesis is reachable or not.

    This method should return an AND/OR tree, that is, an
    AND or OR object, whose constituents are the subgoals that
    need to be tested. The leaves of this tree should be strings
    (possibly with unbound variables), *not* AND or OR objects.
    Make sure to use simplify(...) to flatten trees where appropriate.
    """
    def get_populated_antecedent(rule, statement):

        binding = match(rule.consequent(), statement)

        if binding:
            return populate(rule.antecedent(), binding)
        else:
            return None

    def get_antecedents_to_check(statement, rules):

        if isinstance(statement, str):

            antecedents_to_check = []
            for rule in rules:
                antecedent = get_populated_antecedent(rule, statement)
                if antecedent != None:
                    antecedents_to_check.append(antecedent)

            if len(antecedents_to_check) == 0:
                return statement

            return OR([statement, get_antecedents_to_check(OR(antecedents_to_check), rules=rules)])

        else:
            if isinstance(statement, OR):
                return OR([get_antecedents_to_check(ele, rules=rules) for ele in statement])
            elif isinstance(statement, AND):
                return AND([get_antecedents_to_check(ele, rules=rules) for ele in statement])

    return simplify(get_antecedents_to_check(hypothesis, rules))


# Uncomment this to test out your backward chainer:
pretty_goal_tree(backchain_to_goal_tree(zookeeper_rules, 'opus is a penguin'))
