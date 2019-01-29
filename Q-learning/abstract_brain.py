from abc import ABC, abstractmethod

#https://www.google.com/search?client=ubuntu&channel=fs&q=abstract+method&ie=utf-8&oe=utf-8

#Abstract class for Learning of the reinforcement brain it contains two methods associated with learning 
#being the choose action method and the memory reply method
class AbstractBrainLearning(ABC):
        

    @abstractmethod
    def choose_action(self): raise NotImplementedError
    
    @abstractmethod
    def memory_replay(self): 

        # This is for when experience replay is not used in the alg , just do a super call to this functions base implementation
        print("Experience Replay wasn't used in this Algorithm")
        raise NotImplementedError


class AbstractBrainPreProcess(ABC): 
#preprocesssing isn't used in run this so it isn'y required in this file but is put as a guide 
#preproccessing is a function that takes 1 frame and outputs the frame after it has been greyscaled,
#normalised and resized
    @abstractmethod        
    def Preproccesing(self, state): raise NotImplementedError
    
    # stacks four frames in deque to output a state, gives the environment a sense of motion
    @abstractmethod
    def four_frames_to_state(self, state, is_new_episode): raise NotImplementedError


#Neural network abstract class
class AbstractNeuralNetwork(ABC):

    @abstractmethod
    def build_network(self): raise NotImplementedError
            




                                                                                                     