
num_actions = 12


class tNode:

  def __init__(self, dataval=None):
    
    self.dataval = dataval
    self.childs = []
    self.parent = None

  def add_node(self, child):
    self.childs.append(child)
    child.parent = self.dataval
  
  def print(self):
    print("\nINFO for NODE: "+ str(self.dataval))    
    print("+PARENT " + str(self.parent))
    if len(self.childs) == 0: print("-NO CHILD")

    for child in self.childs:
      print("-CHILD " + str(child.dataval))



class tTree:
  def __init__(self, headval = None):
    self.headval = headval

  def print(self, depth):
    print("\n Tree Print:")
    queue = [self.headval]
    
    printval = self.headval
    print (printval.dataval)
    
    #while printval != 0:
      #printval = self.headval.childs 
      #for child in printval.childs:
      
      #print("-" + printval.dataval)      
      

      
      




def main():
  Root = tNode("Root")
  Tree = tTree(Root)
  

  child1 = tNode("Child1")
  child2 = tNode("Child2")
  
  Root.print()
  Root.add_node(child1)
  Root.add_node(child2)
  Root.print()

  child11 = tNode("Child1-1")
  child12 = tNode("Child1-2")
  child21 = tNode("Child2-1")
  child22 = tNode("Child2-2")
  
  child1.print()
  child1.add_node(child11)
  child1.add_node(child12)
  child1.print()
  
  child2.print()
  child2.add_node(child21)
  child2.add_node(child22)  
  child2.print()

  Tree.print(depth = 1)
  #Tree.print(depth = 2)

  #print(Root)
  exit(0)





if __name__=="__main__":
  main()

exit(0)

