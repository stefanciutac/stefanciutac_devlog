### **Introduction**

It's 2026\. I shouldn't need to exert myself having fun playing board games when a computer can have fun for me. Enter: *evoNN-uttt* (catchy, I know) — a neural network trained to beat me at the game of 'Ultimate Tic-Tac-Toe'. 

Aside from some brief experiments conducted a few months ago (consisting of optimising hyperparameters for a genetic algorithm that solved the pure knapsack problem), I do not have much experience with evolutionary algorithms (note: I will use the terms *evolutionary algorithm* and *genetic algorithm* interchangeably), so iterating on and optimising a library for running the algorithm efficiently should be an interesting learning experience.

### **Motivation**

I concede that there are not, upon initial reflection, any particularly paradigm-changing use-cases for such a bot; its development, however, should certainly be of great educational value to me.

Pedagogical benefits aside, I was inspired to begin this project by the great feats of Google Deepmind's *AlphaGo* and *AlphaZero* models, and the compelling scientific promise of their Nobel-prize-winning sibling: AlphaFold; through optimising game-playing bots, Deepmind gained valuable knowledge and experience, which were, no doubt, invaluable to their eventual development of AlphaFold. Similarly, I hope — through this series of experiments with a board-game-playing-bot — to develop versatile, modular, and (ideally) performant libraries on which I can build more useful projects in the future. For example, the development of a robust and modular library for efficiently training neural networks using genetic algorithms, might \- when plugged into some use-case-specific reward function \- allow noise-and-pertubation-resistant control policies for dynamic robotics to be generated.

### **Architecture**

Everything will be written in C++ so that performance is workable in spite of invariably amateurish implementation. I will generally try to write things from scratch rather than using too many external libraries, ~~because I have too much time on my hands~~ because I will learn more doing it this way \- ultimately, this entire project could just be plugging other people’s code into other people’s code, or better yet, running a brute-force minimax search, if all I cared about was the final result.

The `Eigen` library will be used for matrix operations, as implementing this by hand would cause a massive hit to performance. Aside from that, I will mostly be using the STL.

Overarchingly, the structure of the program looks something like this:

1. A *population* of random genomes (consisting of randomly-initialised parameters, but conforming to user-set hyperparameters) is initialised.  
2. One full *tournament* is run (currently a round-robin where all agents play all other agents), and the ranking of genomes (ranked by points obtained; 3-1-0 for a win-draw-loss respectively) returned.  
3. An elitist selection algorithm and mutations are used to construct a new generation.  
4. Steps 2 and 3 are repeated for *n* generations

It should be noted that for this first version of the code, training is done for a 3x3 board.

#### *Initialisation*

For the sake of simplicity, the `std::vector` of `Eigen::MatrixXd` is initialised completely randomly, using the `Eigen::MatrixXd::Random()` function. There is probably the potential to engineer the starting parameters to increase the rate of convergence, but that would require some more experimentation for a meaningful difference to be made \- and certainly it will not be the bottleneck for efficiency in this first version.

The `population_size`, number of training `generations`, neural net `configuration`, and mutation rates have been selected based on very cursory experimentation, and there is definitely significant room for optimisation on this front. I intend to add an outer loop that crudely sweeps through a combination of these, or perhaps even a meta-genetic algorithm that optimises the hyperparameters, since they are interdependent rather than independent.

#### *Error, Reward, and Evaluation*

The decision to choose a tournament ranking over some objective evaluation function (e.g. a normalised weighted average of an agent’s score in a 100-game match against 3 hard-coded bots of different strengths, or some measure of the average change in the agent’s win probability after each of 100 moves in 100 randomly-generated positions, calculated using a minimax search) sacrifices training efficiency and probably peak performance for the sake of generality. Strong performance achieved by optimising an algorithm of adversarial evolution is much more easily generalised to problems in robotic control policy generation, for example, where it is far more difficult to design effective objective fitness functions. In a similar vein, the issue of sparse feedback (win, loss, or draw, in this case) could have been avoided here by using a move-by-move change position evaluation to provide more granular feedback on agent performance, but this also does not generalise well for similar reasons. The round-robin format specifically, assuming sufficient genetic diversity, punishes solutions optimised for a particular kind of opponent and usually results in rankings that more accurately determine relative performance.

There are, however, some problems with the current implementation.

1. The ‘true’ round-robin matchmaking algorithm results in a lack of evolutionary pressure for promising solutions, as they tend to win against the vast population of random movers, resulting in their placing in the higher percentiles, so winning/losing against the relatively few other promising solutions does not make too large of a difference on their mutation rate, causing a slow rate of convergence. This is also very slow for larger populations, as each tournament is *O(n2)*.  
   1. Instead, the population should be split by performance, with round-robin pairing within each segment of the population. This solution also vastly increases efficiency, as it is embarrassingly parallelisable.  
   2. A swiss pairing system would also be effective, as it significantly reduces the total number of games played, and removes the need for a potentially convoluted algorithm for determining how exactly the population should be split by performance.  
2. The selection algorithm only receives a final ranking, and not individual point totals, so it is both difficult to differentiate between the performance of good solutions and difficult to ascertain where the boundary between statistically significant solutions and random movers lies.

#### *Selection*

Using the results from the small-scale experiments I conducted a few months ago on optimising selection in genetic algorithms for solving the knapsack problem, I have decided on the following approach.

1. Trivially, the mutation rate applied to an agent’s genome should be roughly inversely proportional to its performance.  
2. Below a certain \- fairly high \- performance percentile threshold (e.g. the 50th percentile), agents should have a devastatingly large mutation rate, so that each generation, a large, genetically diverse population is created, from which \- given a large enough total population size \- some small proportion will have statistically significant performance.  
3. Once an agent is identified as having performance that is statistically significant (i.e. in this case, notably better than a random mover), the selection algorithm should move the agent out of this ‘disposable’ segment of the population and into the ‘preserved’ segment, where the mutation rate is significantly lower; this way, once an agent becomes good enough to have potential, its genome is largely frozen, and changes to it are incremental \- several ‘levels’ of ‘preserved’ agents, with a mutation rate that decreases significantly between each one, results in ‘fairly good’ solutions that develop rather than stagnating, and a consistently progressing ‘best solution’.  
4. Finally, elitism is used to prevent regression; if perfect copies of the very best performers are passed into the next generation, not only is regression of the best solution simply impossible, but there is some degree of selection pressure always present.

The main departure from the selection algorithm used in the aforementioned experiment is that crossover is completely omitted, since the parameters of a neural network are interdependent and not at all additive; combining two good solutions almost always results in a terrible solution.

The current implementation is extremely rough-around-the-edges, with clear issues.

1. The percentile cutoffs for each ‘level’ or ‘segment’ of the population, and hence for the mutation rates applied, are static, rather than adapting dynamically to the performance of agents in the population; this would cause the fix in *Error, Reward, and Evaluation: 1a* to fail, as the agent pools into which the population would be split would not be guaranteed to contain agents of roughly equal performance.  
   1. Ideally, the mutation rate should be determined separately for each agent based on their performance.  
   2. For efficient round-robin matchmaking, some kind of algorithm (the workings of which are not yet known to me) would need to be devised to determine the agents that go in each player pool, likely reliant on the fix in *Error, Reward, and Evaluation: 2\.*  
2. By necessity, given the framework I have chosen to implement, the effective culling of solutions that do not meet the percentile cutoff destroys partial solutions that have the potential to improve over several generations; the convergence rate is not as high as it could be, because the performance barrier that a new solution has to reach within a single generation (i.e. the quality of solution that has to be arrived upon purely by chance) to be preserved is high, so the frequency of new solutions is therefore probably quite low compared to a different architecture.  
   1. A potential solution to this would, again, be the introduction of variability in mutation rate at the level of individual agents.  
   2. The segmented-pool round-robin fix proposed in *Error, Reward, and Evaluation: 1* would also offer an improvement here.  
3. No logs are generated or population performance statistics captured, so it is very difficult to benchmark or debug the system.

#### *Neural Network*

A multi-layer perceptron model was chosen for its simplicity of implementation, and thus simplicity of debugging. Sigmoid activation was chosen over ReLU, as if the latter were chosen, parameters initialised to negative values would all be set to 0, potentially leading to agents beginning with smaller effective neural networks as a result. Should a NEAT-like system replace the globally-static hidden layer configuration in future versions, ReLU mapping an input of 0 to an output of 0 allows the addition of new nodes to hidden layers with weights and biases initialised at 0, which is crucial for preventing the temporary-regression-induced selection against neuroevolution.

Initially, the raw output of the agents’ networks were played on the board, with games being immediately given to the opponent if an illegal move was made. However, this led to poor convergence, as much of the training time was being wasted on (unsuccessfully) learning where on the board it is acceptable to play. To solve this, action masking was introduced. Performance gains were, predictably, immediate.

Some thought has been given to the design of the input and output layers of the full, 81-square version of ultimate tic-tac-toe. The following ideas have been considered.

1. Recursion being used to vastly reduce the number of input and output nodes, with 10 in each; 9 for selecting the board/square, and a tenth node to indicate to the network whether it is in board-selection mode or square-selection mode. This is potentially fairly difficult for a network to learn, so it might detract from the rate of convergence.  
2. One node per square; 81 nodes in total. This is the simplest way of doing things, but the drawback is that good solutions must develop ‘awareness’ of the board, which may increase convergence time.  
3. Some variation of over–informing the network. For example, a node for each of the squares *and* 9 additional nodes encoding the state of each of the boards as input, and 81 nodes as output. Alternatively, the 81 squares *and* the status of every diagonal and side, including the meta-diagonals and meta-sides. Grouping the information the model is given in this way reduces the amount of ‘awareness’ or ‘understanding’ of the board it has to achieve internally.

There are other options, of course, but I have decided that 81 input nodes and 81 output nodes, one for each square, will be the chosen architecture of the neural network in future versions that deal with the full, 81-square game. Similarly to other decisions made about the system’s architecture, the ease with which performance is achieved at this particular task is secondary to the generality of the selection algorithm. Giving the network no more information than it strictly needs forces the development of a robust, efficient, and effective training algorithm, which is then more easily applied to tasks where information is sparse.

#### *Hyperparameters*

As discussed earlier, the hyperparameters in this version have not been optimised. However, after brief testing, it was determined that the following are vaguely effective.

1. Obviously, as great a population size and number of generations as is tractable. Fair performance was obtained with a population of 200 running for 500 generations.  
2. For the 3 mutation rates used in this version, values of `[0.0001, 0.001, 0.5]` were tested successfully for `elite_mutation_rate`, `good_mutation_rate`, and `bad_mutation_rate`, respectively.  
3. The neural network, the parameters of which were encoded by each solution, had two hidden layers with 20 nodes in each. For the development of more advanced strategy and the potential for perfect play, this network size is, even in 3x3 naughts and crosses, perhaps a little on the low side; a network that is too big can drastically increase the convergence time, and perfect play on a 3x3 board can still likely be achieved with this configuration, given a well-designed-and-tuned selection algorithm and adequate training time.  
   

Please refer to this project’s GitHub repository for the source code.

### **Performance**

#### *Efficiency*

The efficiency of the program leaves much to be desired. The training time could be reduced by an order of magnitude by multi-threading the tournament stage, especially if combined with the proposals in *Error, Reward, and Evaluation: 1* to reduce the total number of games played/necessary.

#### *Short-Training Test*

A population of 120 agents was trained for 500 generations (\~1 hour on an Intel core i5-8350U CPU @ 1.70GHz, on a single thread), with mutation rates of `[0.0001, 0.001, 0.5]` and a static global network configuration of `{9, 20, 20, 9)`. The solution ranked first after the final generation was tested.   
As mentioned before, this version of the code is naively lacking in any sort of objective metric that can be tracked or analysed, so the analysis of performance will be brief and hand-wavy.

1. Evident, albeit limited, development of strategy: the solution attempted to set up 3-in-a-row in most variations, but it often did not stop my attempts at the same.  
2. Severe tactical limitations: although it usually capitalised on 3-in-a-row left hanging, it sometimes did not, and seemed almost completely oblivious to the opponent’s opportunity for 3-in-a-row.

Overall, optimal play was discovered in a few particular variations, but play was largely suboptimal. Considering the architectural limitations identified earlier, the short training time, and the small population size, it is not surprising that the most successful solution was a naively aggressive one, which likely won many quick games against random movers and elite solutions alike; primitively defensive strategies would not win many games, and rounded strategies are difficult to develop over just 500 generations. It is also clear (and to be expected) that no ‘understanding’ of the game and its rules developed, as \- if the solution tested is assumed to be representative of the entire population \- encoding set variations in the neural network was the most successful strategy.

#### *Long-Training Test*

A population of 250 agents was trained for 10,000 generations (\~320 hours on an Intel core i5-10500 CPU @ 3.10GHz, on a single thread), with mutation rates of `[0.0001, 0.001, 0.5]` and a static global network configuration of `{9, 20, 20, 9)`. The solution ranked first after the final generation was tested.

\[TRAINING ONGOING\]