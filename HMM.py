'''
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

    '''
    在给定模型 =(A, B, pi) 和观察序列 O＝O1O2 …OT 的情况下，选择在一定意义下“最优”的状态 序列 Q = q1 q2 … qT，使得该状态序列“最好地解 释”观察序列
    第一种解释：
        (1) 模型在时间 t 到达状态 i, 并且输出O＝O1O2 …OT。 根据前向变量的定义，实现这一步的概率为at(i)。
        (2) 从时间 t，状态 Si 出发，模型输出O＝O1O2 …OT， 根据后向变量定义，实现这一步的概率为bt(i)。
        于是： p（qt=si,O|u）=at(i)*bt(i)
        相当于 给出观测序列，找的不是整体最大可能的路线 而是 找到t时刻最可能的状态
    返回t时刻最可能的状态
    O(N*N*T)
    '''
    def viterbi_method1(self,observe_sequence,t):
        #通过前向传播计算t
        a=[0.0 for i in range(self.status_numbers)]
        index=self.Observations.index(observe_sequence[0])
        for i in range(self.status_numbers):
            a[i]=self.initial_state_probability_distribution[i]*self.probability_distribution_FromStatusObserve[i][index]
        for i in range(1,t):#T
            buff=a.copy()
            index=self.Observations.index(observe_sequence[i])
            for j in range(self.status_numbers):#N
                sum=0
                for k in range(self.status_numbers):#N
                    sum+=buff[k]*self.trans_matrix_SiToSj[k][j]
                a[j]=sum*self.probability_distribution_FromStatusObserve[j][index]
        #通过后向传播计算Bt
        b=[1.0 for i in range(self.status_numbers)]
        s=len(observe_sequence)-1
        while s>=t:
            buff=b.copy()
            index=self.Observations.index(observe_sequence[s])
            for i in range(self.status_numbers):
                sum=0
                for j in range(self.status_numbers):
                    sum+=self.trans_matrix_SiToSj[i][j]*self.probability_distribution_FromStatusObserve[j][index]*buff[j]
                b[i]=sum
            s-=1
        #求argmax使得At*Bt最大
        max=0
        s=0
        for i in range(self.status_numbers):
            if a[i]*b[i]>max:
                max=a[i]*b[i]
                s=i
        return s
    '''
    在给定模型 =(A, B, pi) 和观察序列 O＝O1O2 …OT 的情况下，选择在一定意义下“最优”的状态 序列 Q = q1 q2 … qT，使得该状态序列“最好地解 释”观察序列
    第一种解释：
        在给定模型 和观察序列O的条件 下求概率最大的状态序列 ，找的是整体最大可能的路线
    返回最大可能的状态序列
    O(N*N*T)
    代码可以改进的地方：配合剪枝，减小搜索复杂度
                        方案1：迭代过程中只保留 概率大于阈值的可能路径
                        方案2：迭代过程中只保留 前k个概率最大的路径
    '''
    def viterbi_method2(self,observe_sequence):
        path=[[0 for i in range(self.status_numbers)] for j in range(len(observe_sequence))]
        a=[0.0 for i in range(self.status_numbers)]

        index=self.Observations.index(observe_sequence[0])
        for i in range(self.status_numbers):
            a[i]=self.initial_state_probability_distribution[i]*self.probability_distribution_FromStatusObserve[i][index]
        for t in range(1,len(observe_sequence)):
            buff=a.copy()
            index=self.Observations.index(observe_sequence[t])
            for j in range(self.status_numbers):
                max=0
                max_index=0
                for i in range(self.status_numbers):
                    if buff[i]*self.trans_matrix_SiToSj[i][j]>max:
                        max=buff[i]*self.trans_matrix_SiToSj[i][j]
                        max_index=i
                a[j]=buff[max_index]*self.trans_matrix_SiToSj[max_index][j]*self.probability_distribution_FromStatusObserve[j][index]
                path[t][j]=max_index
        max=0
        max_index=0
        for i in range(self.status_numbers):
            if a[i]>max:
                max=a[i]
                max_index=i
        status_sequence=[0 for i in range(len(observe_sequence))]
        t=len(observe_sequence)-1
        while t>=0:
            status_sequence[t]=self.Status[max_index]
            max_index=path[t][max_index]
            t-=1
        return status_sequence