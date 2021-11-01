# MIT 6.034 Lab 4: Rule-Based Systems
# Written by 6.034 staff

from production import IF, AND, OR, NOT, THEN, DELETE, forward_chain, pretty_goal_tree
from data import *
import pprint

pp = pprint.PrettyPrinter(indent=1)
pprint = pp.pprint

#### Part 1: Multiple Choice #########################################

ANSWER_1 = '2'

ANSWER_2 = '4'

ANSWER_3 = '2'

ANSWER_4 = '0'

ANSWER_5 = '3'

ANSWER_6 = '1'

ANSWER_7 = '0'

#### Part 2: Transitive Rule #########################################

# Fill this in with your rule 
transitive_rule = IF( AND( '(?x) beats (?y)',
                 '(?y) beats (?z)'), THEN('(?x) beats (?z)') )

# You can test your rule by uncommenting these pretty print statements
#  and observing the results printed to your screen after executing lab1.py
# pprint(forward_chain([transitive_rule], abc_data))
# pprint(forward_chain([transitive_rule], poker_data))
# pprint(forward_chain([transitive_rule], minecraft_data))


#### Part 3: Family Relations #########################################

# Define your rules here. We've given you an example rule whose lead you can follow:
same_rule = IF(OR('person (?x)'), THEN('same (?x) (?x)'))
sibling_rule = IF( AND("person (?x)", "person (?y)", "person (?z)" ,
                    "parent (?z) (?x)","parent (?z) (?y)", NOT('same (?x) (?y)')), 
                  THEN ("sibling (?x) (?y)", "sibling (?y) (?x)") )


friend_rule = IF(AND("person (?x)", "person (?y)",NOT("same (?x) (?y)")), THEN ("friend (?x) (?y)", "friend (?y) (?x)") )


child_rule = IF( AND("person (?x)", "person (?y)","parent (?x) (?y)"), THEN ("child (?y) (?x)") )

cousin_rule = IF(AND("child (?m) (?x)", "child (?n) (?y)",
                "sibling (?x) (?y)", NOT("same (?m) (?n)")), THEN ("cousin (?m) (?n)", "cousin (?n) (?m)") )

grandparent_child_rule = IF(AND("child (?x) (?y)", "child (?y) (?z)"), 
                        THEN ("grandparent (?z) (?x)","grandchild (?x) (?z)") )


# Add your rules to this list:
family_rules = [same_rule,sibling_rule,child_rule,cousin_rule, grandparent_child_rule]
# family_rules = [same_rule,sibling_rule,child_rule,cousin_rule, grandparent_child_rule]

# family_rules = [child_rule]

# Uncomment this to test your data on the Simpsons family:
# pprint(forward_chain(family_rules, simpsons_data, verbose=True))

# These smaller datasets might be helpful for debugging:
# print("sibling test: ", sibling_test_data)
# pprint(forward_chain(family_rules, sibling_test_data, verbose=True))

# print("grandparent test: ", grandparent_test_data)
# pprint(forward_chain(family_rules, grandparent_test_data, verbose=True))

# # The following should generate 14 cousin relationships, representing 7 pairs
# # of people who are cousins:
# print("Harry Potter: ", harry_potter_family_data)
harry_potter_family_cousins = [
    relation for relation in
    forward_chain(family_rules, harry_potter_family_data, verbose=False)
    if "cousin" in relation ]

# To see if you found them all, uncomment this line:
# pprint(harry_potter_family_cousins)


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
    # print("rules: ", rules)
    # print("original input: ", hypothesis)
    def get_populated_antecedent(rule, statement):
        binding = match(rule.consequent(), statement)
        # print("b: ",binding)
        if binding != None:
            # print("p: ", populate(rule.antecedent(), binding))
            return populate(rule.antecedent(), binding)
        else:
            return None

    def get_antecedents_to_check(statement, rules):

        if isinstance(statement, str):
            antecedents_to_check = []
            for rule in rules:
                antecedent = get_populated_antecedent(rule, statement)
                # print("ante: ", antecedent)
                if antecedent != None:
                    antecedents_to_check.append(antecedent)

            if len(antecedents_to_check) == 0:
                return statement
            return OR([statement, get_antecedents_to_check(OR(antecedents_to_check), rules)])

        else:
            if isinstance(statement, OR):
                return OR([get_antecedents_to_check(ele, rules) for ele in statement])
            elif isinstance(statement, AND):
                return AND([get_antecedents_to_check(ele, rules) for ele in statement])

    return  simplify(get_antecedents_to_check(hypothesis, rules))



# Uncomment this to test out your backward chainer:
# ppretty_goal_tree(backchain_to_goal_tree(zookeeper_rules, 'opus is a penguin'))


#### Survey #########################################

NAME = "Xiaohan Yang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = "12-13 hours"
WHAT_I_FOUND_INTERESTING = """I found Part 3 interesting but the most challenging. 
I didn't know how to use just the person rule to exclude the same person at the beginning. 
Gladly, our classmates' discussion on Piazza inspired me to come up with an additional rule. That helped a lot!!"""
WHAT_I_FOUND_BORING = """I don't think there is any boring part. This assignment deepened my understanding about rules,
especially in backward chaining."""
SUGGESTIONS = ""


###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

# The following lines are used in the tester. DO NOT CHANGE!
print("(Doing forward chaining. This may take a minute.)")
transitive_rule_poker = forward_chain([transitive_rule], poker_data)
transitive_rule_abc = forward_chain([transitive_rule], abc_data)
transitive_rule_minecraft = forward_chain([transitive_rule], minecraft_data)
family_rules_simpsons = forward_chain(family_rules, simpsons_data)
family_rules_harry_potter_family = forward_chain(family_rules, harry_potter_family_data)
family_rules_sibling = forward_chain(family_rules, sibling_test_data)
family_rules_grandparent = forward_chain(family_rules, grandparent_test_data)
family_rules_anonymous_family = forward_chain(family_rules, anonymous_family_test_data)
