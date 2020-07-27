import numpy as np
from scipy.spatial.distance import cdist


class UnionFind:
    """
    Class that implements the union-find structure with
    union by rank and find with path compression
    Each node has an slope field initialized when constructed
    With each union, the degree is set to be the average of all unified nodes
    """

    def __init__(self, n, xy_centers, slopes, b_list,lengths,lines_coords):
        self.parent = list(range(n))
        self.rank = [0 for x in range(n)]
        self.size = [1 for x in range(n)]
        self.slopes = slopes
        self.xy_centers = xy_centers
        self.b_list = b_list
        self.lengths = lengths
        self.edges_coords = [[(line[0],line[1]),(line[2],line[3])] for line in lines_coords]

    def set_new_props_after_union(self, parent, child):
        new_size = self.size[parent] + self.size[child]
        new_slope = (self.slopes[parent] * self.size[parent] + self.slopes[child] * self.size[child]) / new_size
        new_x_center = self.xy_centers[parent][0] * self.size[parent] + self.xy_centers[child][0] * self.size[child]
        new_x_center /= new_size
        new_y_center = self.xy_centers[parent][1] * self.size[parent] + self.xy_centers[child][1] * self.size[child]
        new_y_center /= new_size
        new_b = new_y_center - new_slope * new_x_center
        new_length = self.get_length(parent) + self.get_length(child)

        # find 2 points with max distance between parent and child old points
        all_old_edges_coords = self.edges_coords[parent] + self.edges_coords[child]
        dists = np.array(cdist(all_old_edges_coords, all_old_edges_coords))
        index1 , index2 = np.unravel_index(dists.argmax(),dists.shape)
        self.edges_coords[parent] = [all_old_edges_coords[index1] , all_old_edges_coords[index2]]

        self.parent[child] = parent
        self.size[parent] += self.size[child]
        self.slopes[parent] = new_slope
        self.xy_centers[parent] = new_x_center, new_y_center
        self.b_list[parent] = new_b
        self.lengths[parent] = new_length

    def find(self, v):
        if not v == self.parent[v]:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, u, v):
        aRoot = self.find(u)
        bRoot = self.find(v)
        if aRoot == bRoot:
            return
        if self.rank[aRoot] > self.rank[bRoot]:
            parent = aRoot
            child = bRoot
        else:
            parent = bRoot
            child = aRoot
            if self.rank[aRoot] == self.rank[bRoot]:
                self.rank[bRoot] += 1
        self.set_new_props_after_union(parent, child)

    def get_length(self,u):
        return self.lengths[self.find(u)]

    def get_edges_coords(self, u):
        return self.edges_coords[self.find(u)]

    def get_edges_distance(self,u):
        dists = cdist(self.edges_coords[self.find(u)],self.edges_coords[self.find(u)])
        return dists[0][1]

    def print_slope(self,u):
        print("slope of: " + str(u) + " is: " + str(self.get_slope(u)))

    def get_slope(self,u):
        return self.slopes[self.find(u)]

    def get_xy_centers(self,u):
        return self.xy_centers[self.find(u)]

    def get_b(self,u):
        return self.b_list[self.find(u)]

    def print_parent(self,u):
        print("parent of: " + str(u) + " is: " + str(self.parent[u]))

    def printParents(self):
        n = len(self.parent)
        print("index:  ", list(range(n)), sep=' ')
        print("parent: ", self.parent, sep=' ')
        slope = [self.get_slope(x) for x in range(n)]
        print("slope:  ", self.slopes, sep='')

    def getDisjointSets(self):
        myDict = {}
        for node in range(len(self.slopes)):
            root = self.find(node)
            if not root in myDict:
                myDict[root] = set([node])
            else:
                myDict[root].add(node)
        return myDict
# ---------------------------------------------------------------


if __name__ == '__main__':
    # Part a)
    slopes = [30, 60, 90, 120, 150, 0, 45, 0, 90]
    b_list = list(range(9))
    xy_centers = [[1 for col in range(2)] for row in range(10)]


    uf = UnionFind(9,xy_centers, slopes, b_list)
    uf.union(2, 1)
    uf.union(4, 3)
    uf.union(6, 5)
    print("\nParent array after union(2,1), union(4,3) and union(6,5):")
    uf.printParents()

    # Part b)
    uf.union(2, 4)
    print("\nParent array after union(2,4)")
    uf.printParents()

    uf.union(1, 7)
    print("\nParent array after union(1,7)")
    uf.printParents()

    # Part c)
    uf.find(2)
    print("\nParent array after find(2)")
    uf.printParents()

    # Part d)
    myDict = {}
    for node in range(9):
        root = uf.find(node)
        if not root in myDict:
            myDict[root] = set([node])
        else:
            myDict[root].add(node)
    print("\nDisjoint sets: ")
    for mySet in myDict.values():
        print(mySet)