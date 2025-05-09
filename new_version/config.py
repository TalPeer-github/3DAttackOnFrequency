args = {'lr','weight_decay'}

lr = 0.001
weight_decay = 0.0001
betas = (0.9, 0.999)
eps = 1e-3

train_size = 0.8
num_epochs = 100
num_iters = 100

num_walks = 40
walk_length = 100
walk_generation_method = ["classic","high_variance","combined"]
dataset_name = "ModelNet40"

def explanations():
    walk_length_details =  """
        Number of points in each walk. Longer walk improves CloudWalker performance up to certain length, 
        probably since longer walks have better coverage of the object, thus distinguish between different categories.
        Alas, when the walk is too long, the GRUs' ability to remember points in the sequence is reduced, 
        which negatively impacts performance. 
        """
    num_walks_details = """
    Accuracy of CloudWalker improves up to 48 walks where it saturates. Note that above 48 walks most of the points in 
    the point cloud are visited at least once, i.e. full coverage. 
    Authors noted that even very few number of walks resulted in good accuracy, means the number of walks may be 
    a tunable parameter that balances trade of between computational power and accuracy.
    """
    classic_random_method = """
    Randomly choosing among the point's unvisited neighbors with a uniform distribution. 
    """
    high_variance_method = """
    Calculates the change in variance for each neighbor of the most recently added point and 
    selecting the one that increases the variance the most.
    """
    combined = """
    Combining both classic & high variance methods by choosing the neighbor who increases the variance 30% of the times,
    and randomly otherwise. NOTE ABOUT THE HEURISTIC, not sufficient since reducing randomness
    """

