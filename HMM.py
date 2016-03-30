'''
(2)在给定模型 =(A, B, pi) 和观察序列 O＝O1O2 …OT 的情况下，如何选择在一定意义下“最优”的状态 序列 Q = q1 q2 … qT，使得该状态序列“最好地解 释”观察序列？
(3)给定一个观察序列O＝O1O2 …OT ，如何根据最大 似然估计来求模型的参数值？即如何调节模型的参 数，使得p(O|) 最大？
'''
class HMM:
    status_numbers=0
    observations_numbers=0
    Status=[]
    Observations=[]
    trans_matrix_SiToSj=[]
    probability_distribution_FromStatusObserve=[]
    initial_state_probability_distribution=[]
    def __init__(self,status,observation,trans_matrix,initial_status,observation_probability_distribution):
        self.status_numbers=len(status)
        self.Status=status

        self.observations_numbers=len(observation)
        self.Observations=observation

        if len(trans_matrix)!=self.status_numbers:
            print('Error：状态数与状态转移矩阵行数不等')
            return 0
        for i in range(self.status_numbers):
            if len(trans_matrix[i])!=self.status_numbers:
                print('Error:第',i,'行状态转移矩阵不等于状态数')
                return 0
            sum=0
            for j in range(self.status_numbers):
                sum+=trans_matrix[i][j]
            if sum!=1:
                print('Error:第',i,'行状态转移矩阵之和不等于1')
                return 0
        self.trans_matrix_SiToSj=trans_matrix

        if len(initial_status)!=self.status_numbers:
            print('Error:初始状态概率分布不等于状态数')
            return 0
        self.initial_state_probability_distribution=initial_status

        if len(observation_probability_distribution)!=self.status_numbers:
            print('Error:观测概率分布矩阵 不等于 状态数')
            return 0
        for i in range(self.status_numbers):
            if len(observation_probability_distribution[i])!=self.observations_numbers:
                print('Error:观测概率分布矩阵 不等于 观测数')
                return 0
            sum=0
            for j in range(self.observations_numbers):
                sum+=observation_probability_distribution[i][j]
            if sum!=1:
                print('Error:第',i,'行 观测概率分布矩阵 之和不等于 1')
                return 0
        self.probability_distribution_FromStatusObserve=observation_probability_distribution

    '''
        动态规划:前向算法
        在给定模型=(A, B, pi) 和观察序列 O＝O1O2 …OT 的情况下，快速计算概率 p(O|)
        时间复杂度O(N*N*T)
    '''
    def forward_algorithm(self,observe_sequence):
        a=[0.0 for i in range(self.status_numbers)]
        index=self.Observations.index(observe_sequence[0])
        for i in range(self.status_numbers):
            a[i]=self.initial_state_probability_distribution[i]*self.probability_distribution_FromStatusObserve[i][index]
        for i in range(1,len(observe_sequence)):#T
            buff=a.copy()
            index=self.Observations.index(observe_sequence[i])
            for j in range(self.status_numbers):#N
                sum=0
                for k in range(self.status_numbers):#N
                    sum+=buff[k]*self.trans_matrix_SiToSj[k][j]
                a[j]=sum*self.probability_distribution_FromStatusObserve[j][index]
        sum=0
        for i in range(len(a)):
            sum+=a[i]
        return sum

    '''
        动态规划:后向算法
        在给定模型=(A, B, pi) 和观察序列 O＝O1O2 …OT 的情况下，快速计算概率 p(O|)
        时间复杂度O(N*N*T)
    '''
    def backward_algorithm(self,observe_sequence):
        a=[1.0 for i in range(self.status_numbers)]
        s=len(observe_sequence)-1
        while s>=0:
            buff=a.copy()
            index=self.Observations.index(observe_sequence[s])
            for i in range(self.status_numbers):
                sum=0
                for j in range(self.status_numbers):
                    sum+=self.trans_matrix_SiToSj[i][j]*self.probability_distribution_FromStatusObserve[j][index]*buff[j]
                a[i]=sum
            s-=1

        index=self.Observations.index(observe_sequence[0])
        sum=0
        for i in range(self.status_numbers):
            sum+=self.initial_state_probability_distribution[i]*self.probability_distribution_FromStatusObserve[i][index]*buff[i]
        return sum