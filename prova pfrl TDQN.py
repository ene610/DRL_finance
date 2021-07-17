# coding=utf-8

"""
Goal: Implementing a custom enhanced version of the DQN algorithm specialized
      to algorithmic trading.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import math
import random
import copy
import datetime

import numpy as np

from collections import deque
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation
from tradingEnv import TradingEnv

import tqdm as tqdm
import pfrl
import torch
import torch.nn
import gym
import numpy

###############################################################################
################################ Global variables #############################
###############################################################################

# Default parameters related to the DQN algorithm
gamma = 0.4
learningRate = 0.0001
targetNetworkUpdate = 1000
learningUpdatePeriod = 1

# Default parameters related to the Experience Replay mechanism
capacity = 100000
batchSize = 32
experiencesRequired = 1000

# Default parameters related to the Deep Neural Network
numberOfNeurons = 512
dropout = 0.2

# Default parameters related to the Epsilon-Greedy exploration technique
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000

# Default parameters regarding the sticky actions RL generalization technique
alpha = 0.1

# Default parameters related to preprocessing
filterOrder = 5

# Default paramters related to the clipping of both the gradient and the RL rewards
gradientClipping = 1
rewardClipping = 1

# Default parameter related to the L2 Regularization
L2Factor = 0.000001

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0


###############################################################################
################################ Class TDQN ###################################
###############################################################################

class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)

class PFRL_DDQN():

    def __init__(self):

        action_space = gym.spaces.Discrete(2)
        stateLength = 30
        observationSpace = 1 + (stateLength - 1) * 4
        observation_space = gym.spaces.Box(low=np.Inf, high=np.NINF, shape=(1, observationSpace), dtype=np.float32)

        obs_size = observation_space.low.size
        n_actions = action_space.n
        q_func = QFunction(obs_size, n_actions)

        # Use Adam to optimize q_func. eps=1e-2 is for stability.
        optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

        # Set the discount factor that discounts future rewards.
        gamma = 0.9

        # Use epsilon-greedy for exploration
        explorer = pfrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=action_space.sample)

        # DQN uses Experience Replay.
        # Specify a replay buffer and its capacity.
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

        # Since observations from CartPole-v0 is numpy.float64 while
        # As PyTorch only accepts numpy.float32 by default, specify
        # a converter as a feature extractor function phi.
        phi = lambda x: x.astype(numpy.float32, copy=False)

        # Set the device id to use GPU. To use CPU only, set it to -1.
        gpu = -1

        # Now create an agent that will interact with the environment.
        agent = pfrl.agents.DoubleDQN(
            q_func,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            replay_start_size=500,
            update_interval=1,
            target_update_interval=100,
            phi=phi,
            gpu=gpu,
        )

class TDQN:
    """
    GOAL: Implementing an intelligent trading agent based on the DQN
          Reinforcement Learning algorithm.

    VARIABLES:  - device: Hardware specification (CPU or GPU).
                - gamma: Discount factor of the DQN algorithm.
                - learningRate: Learning rate of the ADAM optimizer.
                - capacity: Capacity of the experience replay memory.
                - batchSize: Size of the batch to sample from the replay memory.
                - targetNetworkUpdate: Frequency at which the target neural
                                       network is updated.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - policyNetwork: Deep Neural Network representing the RL policy.
                - targetNetwork: Deep Neural Network representing a target
                                 for the policy Deep Neural Network.
                - optimizer: Deep Neural Network optimizer (ADAM).
                - replayMemory: Experience replay memory.
                - epsilonValue: Value of the Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - iterations: Counter of the number of iterations.

    METHODS:    - __init__: Initialization of the RL trading agent, by setting up
                            many variables and parameters.
                - getNormalizationCoefficients: Retrieve the coefficients required
                                                for the normalization of input data.
                - processState: Process the RL state received.
                - processReward: Clipping of the RL reward received.
                - updateTargetNetwork: Update the target network, by transfering
                                       the policy network parameters.
                - chooseAction: Choose a valid action based on the current state
                                observed, according to the RL policy learned.
                - chooseActionEpsilonGreedy: Choose a valid action based on the
                                             current state observed, according to
                                             the RL policy learned, following the
                                             Epsilon Greedy exploration mechanism.
                - learn: Sample a batch of experiences and learn from that info.

                #TODO cambiato
                - training: Train the trading DQN agent by interacting with its
                            trading environment.
                #TODO cambiato
                - testing: Test the DQN agent trading policy on a new trading environment.

                - plotExpectedPerformance: Plot the expected performance of the intelligent
                                   DRL trading agent.
                - saveModel: Save the RL policy model.
                - loadModel: Load the RL policy model.
                - plotTraining: Plot the training results (score evolution, etc.).
                - plotEpsilonAnnealing: Plot the annealing behaviour of the Epsilon
                                     (Epsilon-Greedy exploration technique).
    """

    def __init__(self, observationSpace, actionSpace, numberOfNeurons=numberOfNeurons, dropout=dropout,
                 gamma=gamma, learningRate=learningRate, targetNetworkUpdate=targetNetworkUpdate,
                 epsilonStart=epsilonStart, epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay,
                 capacity=capacity, batchSize=batchSize):
        """
        GOAL: Initializing the RL agent based on the DQN Reinforcement Learning
              algorithm, by setting up the DQN algorithm parameters as well as
              the DQN Deep Neural Network.

        INPUTS: - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - numberOfNeurons: Number of neurons per layer in the Deep Neural Network.
                - dropout: Droupout probability value (handling of overfitting).
                - gamma: Discount factor of the DQN algorithm.
                - learningRate: Learning rate of the ADAM optimizer.
                - targetNetworkUpdate: Update frequency of the target network.
                - epsilonStart: Initial (maximum) value of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - epsilonEnd: Final (minimum) value of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - epsilonDecay: Decay factor (exponential) of Epsilon, from the
                                Epsilon-Greedy exploration technique.
                - capacity: Capacity of the Experience Replay memory.
                - batchSize: Size of the batch to sample from the replay memory.

        OUTPUTS: /
        """

        # Initialise the random function with a new random seed
        random.seed(0)

        # Check availability of CUDA for the hardware (CPU or GPU)
        self.device = torch.device('cuda:' + str(GPUNumber) if torch.cuda.is_available() else 'cpu')

        # Initialization of the iterations counter
        self.iterations = 0

        self.agent = PFRL_DDQN()

        # Initialization of the tensorboard writer
        self.writer = SummaryWriter('runs/' + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))

    def getNormalizationCoefficients(self, tradingEnv):
        """
        GOAL: Retrieve the coefficients required for the normalization
              of input data.

        INPUTS: - tradingEnv: RL trading environement to process.

        OUTPUTS: - coefficients: Normalization coefficients.
        """

        # Retrieve the available trading data
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()

        # Retrieve the coefficients required for the normalization
        coefficients = []
        margin = 1
        # 1. Close price => returns (absolute) => maximum value (absolute)
        returns = [abs((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]) for i in range(1, len(closePrices))]
        coeffs = (0, np.max(returns) * margin)
        coefficients.append(coeffs)
        # 2. Low/High prices => Delta prices => maximum value
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice) * margin)
        coefficients.append(coeffs)
        # 3. Close/Low/High prices => Close price position => no normalization required
        coeffs = (0, 1)
        coefficients.append(coeffs)
        # 4. Volumes => minimum and maximum values
        coeffs = (np.min(volumes) / margin, np.max(volumes) * margin)
        coefficients.append(coeffs)

        return coefficients

    def processState(self, state, coefficients):
        """
        GOAL: Process the RL state returned by the environment
              (appropriate format and normalization).

        INPUTS: - state: RL state returned by the environment.

        OUTPUTS: - state: Processed RL state.
        """

        # Normalization of the RL state
        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

        # 1. Close price => returns => MinMax normalization
        returns = [(closePrices[i] - closePrices[i - 1]) / closePrices[i - 1] for i in range(1, len(closePrices))]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [((x - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]
        # 2. Low/High prices => Delta prices => MinMax normalization
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(1, len(lowPrices))]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [((x - coefficients[1][0]) / (coefficients[1][1] - coefficients[1][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]
        # 3. Close/Low/High prices => Close price position => No normalization required
        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i] - lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i] - lowPrices[i]) / deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [((x - coefficients[2][0]) / (coefficients[2][1] - coefficients[2][0])) for x in
                        closePricePosition]
        else:
            state[2] = [0.5 for x in closePricePosition]
        # 4. Volumes => MinMax normalization
        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [((x - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]

        # Process the state structure to obtain the appropriate format
        state = [item for sublist in state for item in sublist]

        return state

    def processReward(self, reward):
        """
        GOAL: Process the RL reward returned by the environment by clipping
              its value. Such technique has been shown to improve the stability
              the DQN algorithm.

        INPUTS: - reward: RL reward returned by the environment.

        OUTPUTS: - reward: Process RL reward.
        """

        return np.clip(reward, -rewardClipping, rewardClipping)

    def training(self, trainingEnv, trainingParameters=[], verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        """
        GOAL: Train the RL trading agent by interacting with its trading environment.

        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes).
                - verbose: Enable the printing of a training feedback.
                - rendering: Enable the training environment rendering.
                - plotTraining: Enable the plotting of the training results.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - trainingEnv: Training RL environment.
        """

        """
        # Compute and plot the expected performance of the trading policy
        trainingEnv = self.plotExpectedPerformance(trainingEnv, trainingParameters, iterations=50)
        return trainingEnv
        """

        # Apply data augmentation techniques to improve the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Initialization of some variables tracking the training and testing performances
        if plotTraining:
            # Training performance
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))
            # Testing performance
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.endingDate
            endingDate = '2020-1-1'
            money = trainingEnv.data['Money'][0]
            stateLength = trainingEnv.stateLength
            transactionCosts = trainingEnv.transactionCosts
            testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
            performanceTest = []

        try:
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")

            # Training phase for the number of episodes specified as parameter
            for episode in tqdm(range(trainingParameters[0]), disable=not (verbose)):

                # For each episode, train on the entire set of training environments
                for i in range(len(trainingEnvList)):

                    # Set the initial RL variables
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    previousAction = 0
                    done = 0
                    stepsCounter = 0

                    # Set the performance tracking veriables
                    if plotTraining:
                        totalReward = 0

                    # Interact with the training environment until termination
                    while done == 0:

                        # Choose an action according to the RL policy and the current RL state
                        #action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)
                        action = self.agent.act(state)
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = trainingEnvList[i].step(action)

                        # Update the RL state
                        state = nextState
                        previousAction = action
                        reset = False
                        self.agent.observe(state, reward, done, reset)
                        # Continuous tracking of the training performance
                        if plotTraining:
                            totalReward += reward

                    # Store the current training results
                    if plotTraining:
                        score[i][episode] = totalReward

                # Compute the current performance on both the training and testing sets
                #TODO cambiato qualcosina
                if plotTraining:
                    # Training set performance
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                    trainingEnv.reset()
                    # Testing set performance
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performance, episode)
                    testingEnv.reset()

        except KeyboardInterrupt:
            print()
            print("WARNING: Training prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Assess the algorithm performance on the training trading environment
        trainingEnv = self.testing(trainingEnv, trainingEnv)

        # If required, show the rendering of the trading environment
        if rendering:
            trainingEnv.render()

        # If required, plot the training results
        if plotTraining:
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot(performanceTrain)
            ax.plot(performanceTest)
            ax.legend(["Training", "Testing"])
            plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingPerformance', '.png']))
            # plt.show()
            for i in range(len(trainingEnvList)):
                self.plotTraining(score[i][:episode], marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            #TODO CAMBIATO
            analyser.displayPerformance('TDQN')

        # Closing of the tensorboard writer
        self.writer.close()

        return trainingEnv

    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        """
        GOAL: Test the RL agent trading policy on a new trading environment
              in order to assess the trading strategy performance.

        INPUTS: - trainingEnv: Training RL environment (known).
                - testingEnv: Unknown trading RL environment.
                - rendering: Enable the trading environment rendering.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.

        OUTPUTS: - testingEnv: Trading environment backtested.
        """

        # Apply data augmentation techniques to process the testing set
        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)

        # Initialization of some RL variables
        coefficients = self.getNormalizationCoefficients(trainingEnv)
        state = self.processState(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0

        # Interact with the environment until the episode termination
        with self.agent.eval_mode():
            while done == 0:
                # Choose an action according to the RL policy and the current RL state
                #action, _, QValues = self.chooseAction(state)
                QValues = [0,0]
                action = self.agent.act(state)
                # Interact with the environment with the chosen action
                nextState, r, done, _ = testingEnvSmoothed.step(action)

                testingEnv.step(action)

                # Update the new state
                state = self.processState(nextState, coefficients)
                self.agent.observe(state, r, done, False)
                # Storing of the Q values
                QValues0 = [0,0,0,0]
                QValues1 = [0,0,0,0]

        # If required, show the rendering of the trading environment
        if rendering:
            testingEnv.render()
            self.plotQValues(QValues0, QValues1, testingEnv.marketSymbol)

        # If required, print the strategy performance in a table
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('TDQN')

        return testingEnv

    def plotTraining(self, score, marketSymbol):
        """
        GOAL: Plot the training phase results
              (score, sum of rewards).

        INPUTS: - score: Array of total episode rewards.
                - marketSymbol: Stock market trading symbol.

        OUTPUTS: /
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
        ax1.plot(score)
        plt.savefig(''.join(['Figures/', str(marketSymbol), 'TrainingResults', '.png']))
        # plt.show()

    def plotQValues(self, QValues0, QValues1, marketSymbol):
        """
        Plot sequentially the Q values related to both actions.

        :param: - QValues0: Array of Q values linked to action 0.
                - QValues1: Array of Q values linked to action 1.
                - marketSymbol: Stock market trading symbol.

        :return: /
        """

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Q values', xlabel='Time')
        ax1.plot(QValues0)
        ax1.plot(QValues1)
        ax1.legend(['Short', 'Long'])
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_QValues_zeri', '.png']))
        # plt.show()

    def plotExpectedPerformance(self, trainingEnv, trainingParameters=[], iterations=10):
        """
        GOAL: Plot the expected performance of the intelligent DRL trading agent.

        INPUTS: - trainingEnv: Training RL environment (known).
                - trainingParameters: Additional parameters associated
                                      with the training phase (e.g. the number
                                      of episodes).
                - iterations: Number of training/testing iterations to compute
                              the expected performance.

        OUTPUTS: - trainingEnv: Training RL environment.
        """

        # Preprocessing of the training set
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)

        # Save the initial Deep Neural Network weights
        initialWeights = copy.deepcopy(self.policyNetwork.state_dict())

        # Initialization of some variables tracking both training and testing performances
        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))

        # Initialization of the testing trading environment
        marketSymbol = trainingEnv.marketSymbol
        startingDate = trainingEnv.endingDate
        endingDate = '2020-1-1'
        money = trainingEnv.data['Money'][0]
        stateLength = trainingEnv.stateLength
        transactionCosts = trainingEnv.transactionCosts
        testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)

        # Print the hardware selected for the training of the Deep Neural Network (either CPU or GPU)
        print("Hardware selected for training: " + str(self.device))

        try:

            # Apply the training/testing procedure for the number of iterations specified
            for iteration in range(iterations):

                # Print the progression
                print(''.join(
                    ["Expected performance evaluation progression: ", str(iteration + 1), "/", str(iterations)]))

                # Training phase for the number of episodes specified as parameter
                for episode in tqdm(range(trainingParameters[0])):

                    # For each episode, train on the entire set of training environments
                    for i in range(len(trainingEnvList)):

                        # Set the initial RL variables
                        coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                        trainingEnvList[i].reset()
                        startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                        trainingEnvList[i].setStartingPoint(startingPoint)
                        state = self.processState(trainingEnvList[i].state, coefficients)
                        previousAction = 0
                        done = 0
                        stepsCounter = 0

                        # Interact with the training environment until termination
                        while done == 0:

                            # Choose an action according to the RL policy and the current RL state
                            action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)

                            # Interact with the environment with the chosen action
                            nextState, reward, done, info = trainingEnvList[i].step(action)

                            # Process the RL variables retrieved and insert this new experience into the Experience Replay memory
                            reward = self.processReward(reward)
                            nextState = self.processState(nextState, coefficients)
                            self.replayMemory.push(state, action, reward, nextState, done)

                            # Trick for better exploration
                            otherAction = int(not bool(action))
                            otherReward = self.processReward(info['Reward'])
                            otherDone = info['Done']
                            otherNextState = self.processState(info['State'], coefficients)
                            self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)

                            # Execute the DQN learning procedure
                            stepsCounter += 1
                            if stepsCounter == learningUpdatePeriod:
                                self.learning()
                                stepsCounter = 0

                            # Update the RL state
                            state = nextState
                            previousAction = action

                    # Compute both training and testing  current performances
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performanceTrain[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performanceTrain[episode][iteration],
                                           episode)
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performanceTest[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performanceTest[episode][iteration],
                                           episode)

                # Restore the initial state of the intelligent RL agent
                if iteration < (iterations - 1):
                    trainingEnv.reset()
                    testingEnv.reset()
                    self.policyNetwork.load_state_dict(initialWeights)
                    self.targetNetwork.load_state_dict(initialWeights)
                    self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)
                    self.replayMemory.reset()
                    self.iterations = 0
                    stepsCounter = 0

            iteration += 1

        except KeyboardInterrupt:
            print()
            print("WARNING: Expected performance evaluation prematurely interrupted...")
            print()
            self.policyNetwork.eval()

        # Compute the expected performance of the intelligent DRL trading agent
        expectedPerformanceTrain = []
        expectedPerformanceTest = []
        stdPerformanceTrain = []
        stdPerformanceTest = []
        for episode in range(trainingParameters[0]):
            expectedPerformanceTrain.append(np.mean(performanceTrain[episode][:iteration]))
            expectedPerformanceTest.append(np.mean(performanceTest[episode][:iteration]))
            stdPerformanceTrain.append(np.std(performanceTrain[episode][:iteration]))
            stdPerformanceTest.append(np.std(performanceTest[episode][:iteration]))
        expectedPerformanceTrain = np.array(expectedPerformanceTrain)
        expectedPerformanceTest = np.array(expectedPerformanceTest)
        stdPerformanceTrain = np.array(stdPerformanceTrain)
        stdPerformanceTest = np.array(stdPerformanceTest)

        # Plot each training/testing iteration performance of the intelligent DRL trading agent
        for i in range(iteration):
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot([performanceTrain[e][i] for e in range(trainingParameters[0])])
            ax.plot([performanceTest[e][i] for e in range(trainingParameters[0])])
            ax.legend(["Training", "Testing"])
            plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingPerformance', str(i + 1), '.png']))
            # plt.show()

        # Plot the expected performance of the intelligent DRL trading agent
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
        ax.plot(expectedPerformanceTrain)
        ax.plot(expectedPerformanceTest)
        ax.fill_between(range(len(expectedPerformanceTrain)), expectedPerformanceTrain - stdPerformanceTrain,
                        expectedPerformanceTrain + stdPerformanceTrain, alpha=0.25)
        ax.fill_between(range(len(expectedPerformanceTest)), expectedPerformanceTest - stdPerformanceTest,
                        expectedPerformanceTest + stdPerformanceTest, alpha=0.25)
        ax.legend(["Training", "Testing"])
        plt.savefig(''.join(['Figures/', str(marketSymbol), '_TrainingTestingExpectedPerformance', '.png']))
        # plt.show()

        # Closing of the tensorboard writer
        self.writer.close()

        return trainingEnv

    def saveModel(self, fileName):
        """
        GOAL: Save the RL policy, which is the policy Deep Neural Network.

        INPUTS: - fileName: Name of the file.

        OUTPUTS: /
        """

        pass

    def loadModel(self, fileName):
        """
        GOAL: Load a RL policy, which is the policy Deep Neural Network.

        INPUTS: - fileName: Name of the file.

        OUTPUTS: /
        """
        pass

