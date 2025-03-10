class PDDLState:
    def __init__(self, predicates):
        """
        Initialize the state with a set of predicates.
        :param predicates: A set of predicate strings representing the initial state.
        """
        self.predicates = set(predicates)
    
    def apply_action(self, action):
        """
        Apply an action to modify the state.
        :param action: A dictionary with 'preconditions', 'add_effects', and 'del_effects'.
        """
        print("============================")
        print(self.predicates)
        if not action["preconditions"].issubset(self.predicates):
            raise ValueError(f"Preconditions {action['preconditions']} not met for action {action['name']}")
        
        self.predicates.difference_update(action["del_effects"])  # Remove negative effects
        self.predicates.update(action["add_effects"])  # Add positive effects
    
    def __repr__(self):
        return f"State({self.predicates})"


def simulate_pddl(initial_state, actions):
    """
    Simulate a sequence of actions from an initial state.
    :param initial_state: Set of predicates representing the initial state.
    :param actions: List of action dictionaries with 'name', 'preconditions', 'add_effects', 'del_effects'.
    :return: Final state after applying all actions.
    """
    state = PDDLState(initial_state)
    
    for action in actions:
        try:
            state.apply_action(action)
            print(f"Applied action: {action['name']}")
            print(state)
        except ValueError as e:
            print(f"Action {action['name']} failed: {e}")
            break
    
    return state.predicates

# Example Usage
initial_state = {"at(A, Room1)", "at(B, Room2)", "connected(Room1, Room2)", "connected(Room2, Room3)"}
actions = [
    {
        "name": "move(A, Room1, Room2)",
        "preconditions": {"at(A, Room1)", "connected(Room1, Room2)"},
        "add_effects": {"at(A, Room2)"},
        "del_effects": {"at(A, Room1)"}
    },
    {
        "name": "move(B, Room2, Room3)",
        "preconditions": {"at(B, Room2)", "connected(Room2, Room3)"},
        "add_effects": {"at(B, Room3)"},
        "del_effects": {"at(B, Room2)"}
    }
]

final_state = simulate_pddl(initial_state, actions)
print("Final State:", final_state)
