class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext


class UnorderedList(object):
    """docstring for UnorderedList"""
    def __init__(self):
        self.head = None

    def isEmpty(self):
        return self.head == None

    def add(self, item):
        # own method: add nodes to the beginning
        tem = Node(item);
        tem.next = self.head
        self.head = tem


    def add2(self, item):
        # add nodes to the beginning
        temp = Node(item)
        temp.setNext(self.head)
        self.head = temp

    def printlist(self):
        cur = self.head
        while cur:
            print cur.data
            cur = cur.next

    def size(self):
        """
        :return int size of list
        """
        cur = self.head
        count = 0
        while cur:
            cur = cur.next
            count += 1

        return count
    def rotate(self):
        # reverse(rotate) the linked list
        
    def search(self, item):
        cur = self.head
        found = False
        while cur and not found:
            if cur.data == item:
                return True
            else:
                prev = cur
                cur = cur.next

        if cur is None:
            print("Data not in list")
        
        return found

    def remove(self, item):
        cur = self.head
        pre = None
        found = False
        while cur and not found:
            if cur.data == item:
                found = True
            else:
                pre = cur
                cur = cur.next
        if cur == None:
            print("can't find this item!")
        if pre == None:
            self.head = cur.next
        else:
            pre.next = cur.next


    def removeNthfromEnd(self, n):
        # Remove nth node from the end of list and assume n is valid
        size = self.size()
        cur = self.head
        pre = None
        for i in range(size - n):
            pre = cur
            cur = cur.next
        if pre is None:
            self.head = cur.next
        else:
            pre.next = cur.next
            cur.next = None
#---------- following are test codes -----------#
mylist = UnorderedList()
mylist.add(1)
mylist.add(2)
mylist.add(3)
mylist.add(4)
mylist.add(5)
mylist.remove2(3)
mylist.printlist()