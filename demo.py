from HMM import *
status=['a','b']
observation=['m','n']
trans_matrix=[[0.5,0.5],
              [0.5,0.5]]
initial_status=[0.5,0.5]
observation_probability_distribution=[[1,0],[0,1]]
a=HMM(status,observation,trans_matrix,initial_status,observation_probability_distribution)
print(a.forward_algorithm(['m','m']))