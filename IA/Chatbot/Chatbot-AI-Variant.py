
import random

class Environment:
    def __init__(self):
        self.state = ""
        
def get_reward(self, action):
    if action == "correct_answer":
        return 1
    else:
        return 0
    
class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.Q = {}
        
    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            if state not in self.Q:
                self.Q[state] = {action: 0 for action in self.actions}
            return max(self.Q[state], key=self.Q[state].get)
        
        def update_Q(self, state, action, reward, next_state, alpha, gamma):
            if state not in self.Q:
                self.Q[state] = {action: 0 for action in self.actions}
            if next_state not in self.Q:
                self.Q[next_state] = {action: 0 for action in self.actions}
            max_next_Q = max(self.Q[next_state].values())
            self.Q[state][action] += alpha * gamma * max_next_Q - self.Q[state][action]

def train(env, agent, episodes, alpha, gamma, epsilon):
    for episode in range(episodes):
        state = env.state
        while True:
            action = agent.choose_action(state, epsilon)
            reward = env.get_reward(action)
            next_state = env.state #The next state it's the same in this example
            agent.update_Q(state, action, reward, next_state, alpha, gamma)
            state = next_state
            if reward == 1:
                break

def main():
    #Creating a an instance of the context and agent
    env = Environment()
    actions = ["correct_answer", "incorrect_answer"]
    agent = QLearningAgent(actions)
    
    #Agent training
    train(env, agent, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    #User interaction
    while True:
        user_input = input("You: ")
        env.state = user_input
        action = agent.choose_action(user_input, epsilon=0) #action exploringless selection
        if action == "correct_answer":
            print("Chatbot: Respuesta correcta.")
        else:
            print("Chatbot: No sé la respuesta. Vuelva a intentarlo")
            