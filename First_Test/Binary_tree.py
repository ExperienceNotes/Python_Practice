from random import randint
class Node:
    def __init__(self,value=None):
        self.value = value
        self.left_child = None #smaller
        self.right_child = None #bigger

class binary_serach_tree:
    def __init__(self):
        self.root = None
    def insert(self,value):
        if self.root == None:
            self.root = Node(value)
        else:
            self._insert(value,self.root)
    def _insert(self,value,cur_node):
        if value < cur_node.value:
            if cur_node.left_child == None:
                cur_node.left_child = Node(value)
            else:
                self._insert(value,cur_node.left_child)
        elif value > cur_node.value:
            if cur_node.right_child == None:
                cur_node.right_child = Node(value)
            else:
                self._insert(value,cur_node.right_child)
        else:
            print("This value has existed")

    def print_tree(self):
        if self.root != None:
            self._print_tree(self.root)
    def _print_tree(self,cur_node):
        if cur_node != None:
            self._print_tree(cur_node.left_child)
            print(str(cur_node.value))
            self._print_tree(cur_node.right_child)
def fill_tree(tree,num_elems = 10,max_int=50):
    for _ in range(num_elems):
        cur_elem = randint(0,max_int) #隨機0~50(不含50)的值
        #print("random_num{}".format(cur_elem))
        tree.insert(cur_elem)
    return tree

tree = binary_serach_tree()
tree = fill_tree(tree)
tree.print_tree()