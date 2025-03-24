A random walk is a random process in the mathematical
space. It describes a path consisting of a succession of random
steps in the mathematical space.

Random walks
can be used to analyze and simulate the randomness of
objects and calculate the correlation among objects, which
are useful in solving practical problems.
In the mathematical space, a simple random walk model
is a random walk on a regular lattice, in which one point
can jump to another position at each step according to a
certain probability distribution.

When Random walk applies in a specific network, the transition probability between nodes is positively
relevant to their correlation strength. That is, the stronger their
association is, the greater the transition probability is. After
enough steps, we can obtain a random path that can describe
the network structure.
A random walk is known as a random process. It describes a
path consisting of a succession of random steps on some mathematical space, which can be denoted as {s_t, t=0,1,2,...},
where s_t is a random variable describing the poisition of the random walk after t steps.
The sequence can also be regarded as a special category of Markov chain.

## Point Clouds

3D data can usually be represented with different formats, including depth images, point clouds, meshes, and
volumetric grids. As a commonly used format, point cloud
representation preserves the original geometric information
in 3D space without any discretization. Therefore, it is the
preferred representation for many scene understanding related applications such as autonomous driving and robotics.

### 3D Shape Classification

Methods for this task usually learn the embedding of each
point first and then extract a global shape embedding
from the whole point cloud using an aggregation method.
Classification is finally achieved by feeding the global embedding into several fully connected layers. According to
the data type of input for neural networks, existing 3D
shape classification methods can be divided into multi-view
based, volumetric-based and point-based methods.
In contrast, pointbased methods directly work on raw point clouds without
any voxelization or projection.

#### Point-based Methods

According to the network architecture used for the feature
learning of each point, methods in this category can be divided into pointwise MLP, convolution-based, graph-based,
hierarchical data structure-based methods and other typical
methods.

##### Pointwise MLP Methods

These methods model each point independently with several shared Multi-Layer Perceptrons (MLPs) and then aggregate a
global feature using a symmetric aggregation
function.

##### PAT - Point Attention Transformers

Point Attention Transformers represents each point
by its own absolute position and relative positions with
respect to its neighbors and learns high dimensional features
through MLPs. Then, Group Shuffle Attention (GSA) is
used to capture relations between points, and a permutation
invariant, differentiable and trainable end-to-end Gumbel
Subset Sampling (GSS) layer is developed to learn hierarchical features.

##### Hierarchical Data Structure-based Methods

These networks are constructed based on different hierarchical data structures (e.g., octree and kd-tree). In these
methods, point features are learned hierarchically from leaf
nodes to the root node along a tree.

We propose a novel, black-box, unified, and general adversarial attack, which leads to misclassification in SOTA mesh
neural
networks.
At the base of our method lies the
concept of an imitating network. We propose to train a network to
imitate a given classification network. For each network we wish
to attack, our imitating network gets as input pairs of a point-cloud & a prediction vector for that point cloud
(i.e. quering all point clouds in the dataset).
We utilize the general appromixation low, and aim to learn the classification function of the given attacked network,
by learning to generate prediction vector of the given point-cloud (hence, our loss function consider the distribution
of the prediction vectors rather than one-hot label vector for used for classification)