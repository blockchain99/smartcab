import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy
from pprint import pprint


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here  
        self.max_trials = 100  
#         self.max_trials = 10  
         # agent's number of reaching the target for  given trials.
        self.success = 0
        #each trial count for a given trials.
        self.total = 0     
        # penalties to get for an agent
        self.penalties = 0
        #number of movement of  agent to get to the destination.
        self.num_moves = 0
        #total reward
        self.num_total_reward = 0
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_waypoint = None

    def update(self, t):
        # Gather inputs
        
        self.next_waypoint = numpy.random.choice(['forward', 'left','right',None])
        inputs = self.env.sense(self)
        
        deadline = self.env.get_deadline(self)
		
        #added input variables
        inputs_oncoming = inputs['oncoming']
        inputs_left = inputs['left']
        inputs_right = inputs['right']		
          
        self.state = {self.next_waypoint,inputs['light']}

        print("** Before action , inputs : \n {}. Deadline : '{}'. Time :{}"\
              "\n State prior to taking action : {}".format(inputs, deadline, t, self.state))
        result_before = "{},{},{},{},{}\n".format(inputs, deadline, t, self.state ,self.env.done)
        with open("result_before1.csv", "a") as myfile:
                myfile.write(result_before)   
     
        action_planner =  self.next_waypoint
        #action is randomly chosen		
        action = random.choice(Environment.valid_actions)
        reward = self.env.act(self, action)   
        
        self.get_final_statistics(deadline, reward, self.max_trials)  
        
        if (self.env.done == True):
            print 'Reached  to the destination : {} ! '.format(Environment.destination)

        print "After action, LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward for taking the action = {}"\
            .format(deadline, inputs, action, reward)  
        #save output status into csv file. 	
        result = "{},{},{},{},{},{},{},{}\n".format(inputs_oncoming,inputs_left,\
                inputs_right,self.next_waypoint,inputs['light'], action, reward,t)
        with open("agentprint1.csv", "a") as myfile:
                myfile.write(result)  
    
        # return success count to reach the destination, move, total reward, total, penalties.
    def get_final_statistics(self, deadline, reward, max_trials):        
        self.num_moves += 1
        self.num_total_reward += reward
        if reward < 0:
            self.penalties+= 1 
            
        self.is_Final=False    
        if deadline == 0:
            self.is_Final = True         
        #if reward  is bigger  than 8,  which means getting  to the destination. So increase  success  by 1.  
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
            with open("success_total_agent1.csv", "a") as myfile:
                myfile.write(result) 
#########################################   
    
    def get_final_success(self):
        return {'self.success':self.success,
                'self.max_trials':self.max_trials}

def run():
    """Run the agent for a finite number of trials."""
   
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
#     e.set_primary_agent(a, enforce_deadline=False)  # set agent to track
    #enforce_deadline=True to compare success rate with that of QLeanring algorithm
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track,  
    # Now simulate it
#     sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    sim = Simulator(e, update_delay=0.00000001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=a.max_trials)  # press Esc or close pygame window to quit
    result = a.get_final_success()
    print("*"*60)
    print("Success and max_trials :\n")
    pprint(result)
    
    

if __name__ == '__main__':
    run()