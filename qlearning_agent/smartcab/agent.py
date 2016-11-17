import pdb
import random
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator
import operator
import collections
import time
from pprint import pprint

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):    
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        #Q dictionary named  to  Qdictionary 
        self.Qdictionary = {} 
#         self.default_Q = 0   
        # Optimal default Q value is  1 instead  of default  0 which makes poor success count.
        self.default_Q = 1
        # agent's number of reaching the target for  given trials.
        self.success = 0
        #each trial count for a given trials.
        self.total = 0
        #maximum trial numbers.
        self.max_trials = 100
        # penalties to get for an agent
        self.penalties = 0
        #number of movement of  agent to get to the destination.
        self.num_moves = 0
        #total reward
        self.num_total_reward = 0
    
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        # following line returns {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        inputs = self.env.sense(self) 
        deadline = self.env.get_deadline(self) #deadline state is not for modeling driving agent, but for displaying statics analysis.   
        self.state = (self.next_waypoint, inputs['light'])
        # TODO: Update state
        # Q learning chosen action
        Qdictionary, action = self.get_Qmax(self.state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.get_final_statistics(deadline, reward, self.max_trials)         
        # TODO: Learn policy based on state, action, reward
        # Note that we are updating for the previous state's Q value since Utility function is always +1 future state.
        if self.previous_state != None:
            if (self.previous_state, self.previous_action) not in self.Qdictionary:
                self.Qdictionary[(self.previous_state, self.previous_action)] = self.default_Q
            # Update to Q matrix according  to  the Udacity lecture 
            self.update_Q(self.previous_state, self.previous_action, self.previous_reward, self.state)  
        self.previous_state = self.state
        self.previous_action = action
        self.previous_reward = reward
#         self.env.status_text += ' ' + self._more_stats()
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]       
##Write statics  data to  file  #####        
        result = "{},{},{},{},{},{},{}\n".format(deadline,inputs['light'], \
                 inputs['oncoming'],inputs['right'], inputs['left'], action, reward)
        with open("LearningAgent_update.csv", "a") as myfile:
            myfile.write(result)
#####################################
    # return success count to reach the destination, move, total reward, total, penalties.
    def get_final_statistics(self, deadline, reward, max_trials):        
        self.num_moves += 1
        self.num_total_reward += reward
        if reward < 0:
            self.penalties+= 1 
            
        self.is_Final=False    
        if deadline == 0:
            self.is_Final = True         
        #if reward  is bigger  than 10,  which means getting  to the destination. So increase  success  by 1.  
        #When agent reach the destination, reward is [12,10.5,9,11],i.e. 10(bonus reward) + one of [2 , 0.5 , -1, 1] 
        #If the reward for an action is greater 8, agent reach the destination
        if reward > 8:    
            self.success += 1
            self.is_Final = True
        #When agent reach the destination or deadline becomes 0,  Increase total(number  of try) by 1.  
        if self.is_Final:
            self.total += 1
            print "success/total = {}/{} of {} trials (net reward: {})\npenalties/num_moves (penalty rate): {}/{} ({})".format(
                  self.success, self.total, self.max_trials, self.num_total_reward, self.penalties, self.num_moves, round(float(self.penalties)/float(self.num_moves), 2)) 

##Write statics data to file  ###########            
            result = "{},{},{},{},{},{},{},{}\n".format(self.success, self.total, self.max_trials,reward, self.num_total_reward, \
                  self.penalties, self.num_moves,round(float(self.penalties)/float(self.num_moves), 2))
            with open("success_total.csv", "a") as myfile:
                myfile.write(result) 
#########################################   

    # return current action   list from  forward, left,   right, None 
    def get_action_list(self, state):
        return ['forward', 'left', 'right',None]
    
    def get_final_success_reward(self):
        return {'self.success':self.success,
                'self.num_total_reward':self.num_total_reward,
                }
        
    # If random number is less  than  epsilon, random action is  selected, else best action selected.     
    def randomTFValue(self, p):            
        return True if random.random() < p else False
    
    #Select maximum Q value and action of the corresponding  state
    #Find  the best  action state pair.  Input is state  , Outputs are best action with max Q value 
    def get_Qmax(self, state):  #state is tuple, consists  of next_waypoint and inputs['light']
        # If random number is less than  epsilon,  the Qmax is corresponding  state, best_action pair
        # the more epsilon value it  has, it explores more, i.e.  more random actions are selected.
        action_list = self.get_action_list(state)
        #when True, a random  action is  chosen.
        if self.randomTFValue(self.epsilon):  
            best_action = random.choice(Environment.valid_actions)
            Qmax = self.getQ(state, best_action)
        else:
            Qmax = -999999
            best_action = None
#             for action in Environment.valid_actions:
            for action in action_list:    
                Qdictionary = self.getQ(state, action)
                if Qdictionary > Qmax:
                    Qmax = Qdictionary
                    best_action = action    
                if Qdictionary == Qmax:   #when values are equal,, assign epsilon  =  0.5
                    if self.randomTFValue(0.5):
                        Qmax = self.getQ(state, action) 
                        best_action = action
        return (Qmax, best_action)
    # update Q value according to previous state, previous action,  reward and  state.
    def update_Q(self,previous_state, previous_action,previous_reward,state ): 
        self.Qdictionary[(previous_state,previous_action)] = (1 - self.learning_rate) * self.Qdictionary[(previous_state,previous_action)] + \
            self.learning_rate * (self.previous_reward + self.discount_factor * self.get_Qmax(self.state)[0])
            
    # return Q value according to  state , action pair    
    def getQ(self, each_state, each_action):   
        # return 0(default_Q) if  Q value is  not  in Qdictinary,  else return the state, action pair
        return self.Qdictionary.get((each_state, each_action), self.default_Q)   
   
#run agent  according to optimal learning rate, discount factor, epsilon which was found in run_find_optimal() module.
def run():
    """Run the agent for a finite number of trials."""
#     learning_rate = 0.9         # learning rate,  alpha
#     discount_factor = 0.33     # discount_factor, discount
#     epsilon = 0.1               # epsilon to pick between random action and explore new paths
    learning_rate = 0.7         # learning rate,  alpha
    discount_factor = 0.1     # discount_factor, discount
    epsilon = 0.1               # epsilon to pick between random action and explore new paths
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    agent = LearningAgent   
    ## learning rate: 0 means Q- values never updated. learning rate: 0.9 means  quick learning
    agent.learning_rate = learning_rate  
    #discount factor between 0 and 1,  the less discount factor it has, the less worth of future rewards than immediate reward.
    agent.discount_factor = discount_factor  #  between 0 and 1. 
    agent.epsilon = epsilon
    a = e.create_agent(agent)  # create agent    
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    #Change to  speed up
    sim = Simulator(e, update_delay=0.00000001)  # reduce update_delay to speed up simulation
#     sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=a.max_trials)  # press Esc or close pygame window to quit


# this module  is to find the optimal  epsilon, learning rate, discount factor. 
def run_find_optimal(): 
    optimal_epsilon = 0
    optimal_learning_rate = 0
    optimal_discount_factor = 0
    optimal_number_of_success = 0
    #create list to store  learning rate, discount factor, epsilon pairs
    list_of_optimal_status = []
    
    for learning_rate in [0.7,0.8,0.9] :
#     for learning_rate in [0.9] :    
        for discount_factor in [0.1, 0.2,0.3,0.33,0.4, 0.44]: 
#         for discount_factor in [0.33]:
             for epsilon in [x * 0.1 for x in range(0,5)]:
                # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)
                agent = LearningAgent
                agent.learning_rate = learning_rate  # learning rate between 0 and 1. Setting it to 0 means nothing is learned i.e Q- values are never updated. while setting it to 0.9 means that learning can occur quickly
                agent.discount_factor = discount_factor  # discount factor between 0 and 1. Lesser means that furture rewards are  worth less than immediate reward.
                agent.epsilon = epsilon
                a = e.create_agent(agent)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

                # Now simulate it
                sim = Simulator(e, update_delay=0.00000001)  # reduce update_delay to speed up simulation
                sim.run(n_trials=a.max_trials)  # press Esc or close pygame window to quit

                list_of_optimal_status.append( [learning_rate,  discount_factor, epsilon, a.get_final_success_reward()['self.success'],\
                                                 a.get_final_success_reward()['self.num_total_reward']] )

                if a.get_final_success_reward()['self.success'] > optimal_number_of_success:
                   optimal_conf = a.get_final_success_reward()
                   optimal_number_of_success =  a.get_final_success_reward()['self.success']
                   optimal_learning_rate = learning_rate
                   optimal_discount_factor = discount_factor
                   optimal_epsilon = epsilon
           
    print('Optimal results  : \n')
    pprint(optimal_conf)   
    print('optimal_number_of_success :',optimal_number_of_success,'. optimal_learning_rate :',optimal_learning_rate,\
          '. optimal_discount_factor : ',optimal_discount_factor,'. optimal_epsilon ::',optimal_epsilon)
    print(list_of_optimal_status)
#     result_optimal = "{},{},{},{}\n".format(optimal_number_of_success, self.total, optimal_learning_rate, optimal_discount_factor ,optimal_epsilon)            
#     with open("optimal.csv", "a") as myfile:
#             myfile.write(result_optimal) 
    thefile = open('optimal_result.csv', 'w')
    for item in list_of_optimal_status:
        thefile.write("%s\n" % item)

if __name__ == '__main__':
    run()
    #run for finding the optimized learning rate, discount  factor, epsilon        
#      run_find_optimal()
    
