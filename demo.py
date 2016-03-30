from HMM import *
status=['a','b']
observation=['m','n']
trans_matrix=[[0.5,0.5],
              [0.5,0.5]]
initial_status=[0.5,0.5]
observation_probability_distribution=[[0.3,0.7],[0.3,0.7]]
a=HMM(status,observation,trans_matrix,initial_status,observation_probability_distribution)
print(a.forward_algorithm(['m','n']))
print(a.backward_algorithm(['m','n']))

print(a.viterbi_method1(['m','n'],1))