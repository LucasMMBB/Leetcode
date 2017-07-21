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

    def listNode(self):
        cur = self
        while cur:
            print cur.data
            cur = cur.next

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
        pre = None
        while head:
            temp = head.next
            head.next = pre
            pre = head
            head = temp

        return pre
            
        
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

    def hasCycle(self):
        node = self.head
        slow = fast = node

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if id(slow) == id(fast):
                return True

        return False

    def midElement(self):
        slow = self.head
        fast = self.head.next
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        return slow.data
#---------- Some useful functions --------------#
a = UnorderedList()
a.add(5)
a.add(4)
a.add(3)
a.add(2)
a.add(1)
print a.midElement()