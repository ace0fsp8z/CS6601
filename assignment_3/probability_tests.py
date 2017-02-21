"""
Contains various local tests for Assignment 3.
"""
#Part 1a
def network_setup_test(power_plant):
    """Test that the power plant network has the proper number of nodes and edges."""
    nodes = power_plant.nodes
    if(len(nodes)==5):
        print('correct number of nodes')
        total_links = sum([len(n.children) for n in nodes] + [len(n.parents) for n in nodes])
        if(total_links == 10):
            print('correct number of edges between nodes')
        else:
            print('incorrect number of edges between nodes')
    else:
        print('incorrect number of nodes')

#Part 1b
def probability_setup_test(power_plant):
    """Test that all nodes in the power plant network have proper probability distributions.
    Note that all nodes have to be named predictably for tests to run correctly."""
    print('checking probability distribution for Temperature node...')
    # first test temperature distribution
    T_node = power_plant.get_node_by_name('temperature')
    if(T_node is not None):
        T_dist = T_node.dist.table
        if(len(T_dist) == 2):
            print('correct temperature distribution size')
            test_prob = T_dist[0]
            if(int(test_prob*100) == 80):
                print('correct temperature distribution')
            else:
                print('incorrect temperature distribution')
        else:
            print('incorrect temperature distribution size')

    # then faulty gauge distribution
    print('checking probability distribution for Faulty Gauge node...')
    F_G_node = power_plant.get_node_by_name('faulty gauge')
    if(F_G_node is not None):
        F_G_dist = F_G_node.dist.table
        rows, cols = F_G_dist.shape
        if(rows == 2 and cols == 2):
            print('correct faulty gauge distribution size')
            test_prob1 = F_G_dist[0][1]
            test_prob2 = F_G_dist[1][0]
            if(int(test_prob1*100) == 5 and int(test_prob2*100) == 20):
                print('correct faulty gauge distribution')
            else:
                print('incorrect faulty gauge distribution')
        else:
            print('incorrect faulty gauge distribution size')

    # faulty alarm distribution
    print('checking probability distribution for Faulty Alarm node...')
    F_A_node = power_plant.get_node_by_name('faulty alarm')
    if(F_A_node is not None):
        F_A_dist = F_A_node.dist.table
        if(len(F_A_dist) == 2):
            print('correct faulty alarm distribution size')
            test_prob = F_A_dist[0]
            if(int(test_prob*100) == 85):
                print('correct faulty alarm distribution')
            else:
                print('incorrect faulty alarm distribution')
        else:
            print('incorrect faulty alarm distribution size')

    # gauge distribution
    # can't test exact probabilities because
    # order of probabilities is not guaranteed
    print('checking only the probability distribution size for Gauge node...')
    G_node = power_plant.get_node_by_name('gauge')
    if(G_node is not None):
        G_dist = G_node.dist.table
        rows1, rows2, cols = G_dist.shape
        if(rows1 == 2 and rows2 == 2 and cols == 2):
            print('correct gauge distribution size')
        else:
            print('incorrect gauge distribution size')

    # alarm distribution
    print('checking only the probability distribution size for Alarm node...')
    A_node = power_plant.get_node_by_name('alarm')
    if(A_node is not None):
        A_dist = A_node.dist.table
        rows1, rows2, cols = A_dist.shape
        if(rows1 == 2 and rows2 == 2 and cols == 2):
            print('correct alarm distribution size')
        else:
            print('incorrect alarm distribution size')


#Part 2a Test
def games_network_test(games_net):
    """Test that the games network has the proper number of nodes and edges."""
    print('checking the total number of edges and nodes in the network...')
    nodes = games_net.nodes
    if(len(nodes)==6):
        print('correct number of nodes')
        total_links = sum([len(n.children) for n in nodes] + [len(n.parents) for n in nodes])
        if(total_links == 12 ):
            print('correct number of edges')
        else:
            print('incorrect number of edges')
    else:
        print('incorrect number of nodes')


    # Now testing that all nodes in the games network have proper probability distributions.
    # Note that all nodes have to be named predictably for tests to run correctly.

    # First testing team distributions.
    # You can check this for all teams i.e. A,B,C (by replacing the first line for 'B','C')

    print ('checking probability distribution for Team A...')
    A_node = games_net.get_node_by_name('A')
    if(A_node is not None):
        A_dist = A_node.dist.table
        if(len(A_dist) == 4):
            print('correct team distribution size')
            test_prob = A_dist[0]
            test_prob2 = A_dist[2]
            if(int(test_prob*100) == 15 and int(test_prob2*100)==30):
                print('correct team distribution')
            else:
                print('incorrect team distributions')
        else:
            print('incorrect team distribution size')
    else:
        print 'No node with the name A exists.'

    # Now testing match distributions.
    # You can check this for all matches i.e. AvB,BvC,CvA (by replacing the first line)
    print ('checking probability distribution for match AvB...')    
    AvB_node = games_net.get_node_by_name('AvB')
    if(AvB_node is not None):
        AvB_dist = AvB_node.dist.table
        rows1, rows2, cols = AvB_dist.shape
        if(rows1 == 4 and rows2 == 4 and cols == 3):
            print('correct match distribution size')
            flag1 = True
            flag2 = True
            flag3 = True
            for i in range(0, 4):
                for j in range(0,4):
                    x = AvB_dist[i,j,]
                    if i==j:
                        if x[0]!=x[1]:
                            flag1=False
                    if j>i:
                        if not(x[1]>x[0] and x[1]>x[2]):
                            flag2=False
                    if j<i:
                        if not (x[0]>x[1] and x[0]>x[2]):
                            flag3=False
            if (flag1 and flag2 and flag3):
                print('correct match distribution')
            else:
                if not flag1:
                    print('incorrect match distribution for equal skill levels')
                if ((not flag2) or (not flag3)):
                    print('incorrect match distribution: team with higher skill should have higher probability of winning')
        else:
            print('incorrect match distribution size')
    else:
        print 'No node with the name AvB exists.'

#Part 2b Test
def posterior_test(posterior):
    if (abs(posterior[0]-0.25)<0.01 and abs(posterior[1]-0.42)<0.01 and abs(posterior[2]-0.31)<0.01):
        print 'correct posterior'
    else:
        print 'incorrect posterior calculated'
        
