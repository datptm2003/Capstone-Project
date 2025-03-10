class BikeShopping:
    def __init__(self):
        # Initial state
        self.state = {
            "at(supermarket)": False,
            "riding(bike)": True,
            "has(apples, 0)": 0,
            "has(oranges, 0)": 0,
            "at(outside_supermarket)": False
        }
        
    def stop_at_supermarket(self):
        if self.state["riding(bike)"]:
            self.state["riding(bike)"] = False
            self.state["at(supermarket)"] = True
            print("You stopped at the supermarket.")
        else:
            print("You are not riding a bike.")

    def buy(self, item, quantity):
        if self.state["at(supermarket)"]:
            key = f"has({item}, {quantity})"
            self.state[key] = quantity
            print(f"You bought {quantity} {item}.")
        else:
            print("You are not at the supermarket.")

    def check_out(self):
        if any(self.state[key] > 0 for key in self.state if "has(" in key):
            print("You checked out.")
        else:
            print("You have nothing to check out.")

    def go_outside(self):
        if self.state["at(supermarket)"]:
            self.state["at(supermarket)"] = False
            self.state["at(outside_supermarket)"] = True
            print("You went outside the supermarket.")
        else:
            print("You are not inside the supermarket.")

    def reached_goal(self):
        return self.state.get("has(oranges, 1)", 0) > 0 and self.state["at(outside_supermarket)"]

    def execute_actions(self, actions):
        for action in actions:
            if action == "Stop at a supermarket":
                self.stop_at_supermarket()
            elif action.startswith("Buy"):
                parts = action.split()
                self.buy(parts[1], int(parts[2]))
            elif action == "Check out":
                self.check_out()
            elif action == "Go outside the supermarket":
                self.go_outside()
            else:
                print(f"Unknown action: {action}")

            if self.reached_goal():
                print("Goal state reached: You bought some oranges and are standing outside the supermarket.")
                break

# Define actions sequence
actions = [
    "Stop at a supermarket",
    "Buy oranges 1",
    "Check out",
    "Go outside the supermarket"
]

# Run the simulation
simulation = BikeShopping()
simulation.execute_actions(actions)
