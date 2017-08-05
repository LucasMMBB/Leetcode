"""
/*\
|*| Leetcode answers in Python
|*| Author: Maoxu
|*| Description: share my answers to whoever needs
|*| Create date: April 3rd, 2017
\*/
"""
class Solution(object):
	# 28. Implement strStr
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        m = len(haystack)
        n = len(needle)

        for i in range(m+1):
        	for j in range(n+1):

        		if j == n:
        			return i

        		if i + j == m:
        			return -1

        		if haystack[i+j] != needle[j]:
        			break

    # 88. Merge Sorted Array
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i = m - 1;
        j = n - 1;
        k = m + n - 1;
        while k >= 0 and i >= 0 and j >= 0:
        	if nums1[i] > nums2[j]:
        		nums1[k] = nums1[i]
        		i -= 1
        	else:
        		nums1[k] = nums2[j]
        		j -= 1

        	k -= 1

        if j >= 0:
        	nums1[:j+1] = nums2[:j+1]

    # SMALLEST DIFFERENCE
    def smallestDifference(self, A, B):
    	A.sort()
    	B.sort()

    	i = j = 0
    	diff = 2147483647

    	while i < len(A) and j < len(B):
    		if A[i] > B[j]:
    			diff = min(A[i] - B[i], diff)
    			j += 1
    		else:
    			diff = min(B[j] - A[i], diff)
    			i += 1

    	return diff

# 86. Partition List

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        dummy1, dummy2 = ListNode(0), ListNode(0)
        cur1, cur2 = dummy1, dummy2

        while head:
        	if head.val < x:
        		cur1.next = head
        		cur1 = cur1.next
        	else:
        		cur2.next = head
        		cur2 = cur2.next

        	head = head.next

        cur2.next = None
        cur1.next = dummy2.next

        return dummy1.next

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head or not head.next or k==0:
        	return head

        size = self.getSize(head)
        k %= size

        dummy = ListNode(0)
        dummy.next = head

        p = head

        for i in range(k):
        	p = p.next

        for i in range(size - k - 1):
        	p = p.next
        	head = head.next

        p.next = dummy.next
        dummy.next = head.next
        head.next = None

        return dummy.next

    def getSize(self, head):
    	i = 0
    	while head:
    		head = head.next
    		i += 1
    	return i


# 234. Palindrome Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head == None:
        	return True

        mid = self.getMid(head)

        right = mid.next
        mid.next = None

        return self.compare(head, self.rotate(right))

     def getMid(self, head):
     	slow = head
     	fast = head.next

     	while fast and fast.next:
     		slow = slow.next
     		fast = fast.next.next

     	return slow


     def rotate(self, head):
     	pre = None
     	while head:
     		temp = head.next
     		head.next = pre
     		pre = head
     		head = temp

     	return pre

     def compare(sefl, h1, h2):
     	while h1 and h2:
     		if h1.val != h2.val:
     			return False

     		h1 = h1.next
     		h2 = h2.next

     	return True
# 19. Remove nth node from end of list
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy = cur = head
        for i in range(n):
            head = head.next
        
        if head is None:
            dummy = dummy.next
            return dummy
        while head.next:
            cur = cur.next
            head = head.next

        cur.next = cur.next.next
        return dummy

# 237.Delete Node in a Linked List
class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

# 203. Remove Linked List Elements
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = pre = ListNode(0)
        pre.next = cur = head
        if cur is None:
            return None
        
        if cur.val is val and cur.next is None:
            return None

        while cur:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre = cur

            cur = cur.next
        return dummy.next

# 92. Reverse Linked List ||
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
    	dummy = cur = head
    	left, right = m, n

# 2. Add Two Numbers
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = cur = ListNode(0)
        carry = 0

        while l1 or l2:
            a = l1.val if l1 else 0
            b = l2.val if l2 else 0

            sm = a + b + carry
            cur.next = ListNode(sm%10)
            cur = cur.next
            carry = sm/10
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next


        if carry > 0:
            cur.next = ListNode(1)

        return dummy.next


# 445. Add Two Numbers ||
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l1 = self.rotate(l1)
        l2 = self.rotate(l2)
        dummy = cur = ListNode(0)
        carry = 0

        while l1 or l2:
            a = l1.val if l1 else 0
            b = l2.val if l2 else 0

            sm = a + b + carry
            cur.next = ListNode(sm%10)
            cur = cur.next
            carry = sm/10
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next


        if carry > 0:
            cur.next = ListNode(1)

        return self.rotate(dummy.next)
    def rotate(self, head):
        # reverse(rotate) the linked list
        pre = None
        while head:
            temp = head.next
            head.next = pre
            pre = head
            head = temp

        return pre


# 147. INSERTION SORT LIST
# Method: insertion sort list
# Time: O(n^2)
# Space: O(1)
class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None:
            return head

        dummy = ListNode(0)
        dummy.next = head
        while head and head.next:
            if head.val <= head.next.val:
                head = head.next
            else:
                cur = dummy
                while cur.next.val <= head.next.val:
                    cur = cur.next
                tmp = head.next
                head.next = tmp.next
                tmp.next = cur.next
                cur.next = tmp
        return dummy.next



# 148. SORT LIST
# Method: merge sort

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

# 141. Linked List Cycle
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if id(slow) == id(fast):
                return True

        return False

# 142. Linked List Cycle ||
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = fast = head

        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None

# 274. H-Index
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """

# 275. H-Index ||
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """

# 230 Kth Smallest Element in a BST
# Method one: DC + Binary Search
"""
: Definition for a binary tree node.
: class TreeNode(object):
:     def __init__(self, x):
:         self.val = x
:         self.left = None
:         self.right = None
"""
class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        n = self.count(root.left)
        if n + 1 == k:
            return root.val
        elif n + 1 < k:
            return self.kthSmallest(root.right, k - n -1)
        else:
            return self.kthSmallest(root.left, k)

    def count(self, root):
        # count total node number of the root
        if root == None:
            return 0

        return self.count(root.left) + self.count(root.right) + 1

# 162. Find Peak Element
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l,r = 0, len(nums) - 1

        while l <= r:
            mid = l + (r - l) / 2

            if(mid == 0 or nums[mid] > nums[mid - 1]) and (mid==len(nums)-1 or nums[mid]>nums[mid + 1]):
                return mid
            elif mid > 0 and nums[mid] < nums[mid - 1]:
                r = mid -1
            else:
                l = mid + 1
        return -1

# 448. Find All Numbers Disappeared in an Array
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        ans = []
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            if(nums[index] > 0):
                nums[index] = -nums[index]

        for i in range(len(nums)):
            if nums[i] > 0:
                ans.append(i + 1)
        return ans

# 442. Find All Duplicates in an Array
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for i in range( len(nums) ):
            index = abs(nums[i]) - 1
            if nums[index] > 0:
                nums[index] = -nums[index]
            else:
                res.append(index + 1)
        return res

# 628. Maximum Product of Three Numbers
class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        a = nums[-1] * nums[-2] * nums[-3]
        b = nums[0] * nums[1] * nums[-1]
        return max(a,b)


# 35. Search Insert Position
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l,r = 0, len(nums)
        while l < r:
            mid = l + (l - r) / 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return l        


# 217. Contains Duplicate
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        pyset = set()
        for i in nums:
            if i not in pyset:
                pyset.add(i)
            else:
                return True
        return False


# 219. Contains Duplicate ||
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        tem = {}
        for i in range(len(nums)):
            if tem.has_key(nums[i]) and i - tem[nums[i]] <= k:
                return True
            tem[nums[i]] = i
        return False

# 220. Contains Duplicate |||
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        if k < 1 or t < 0:
            return False

        dic = collections.OrderedDict()

        for i in range(len(nums)):
            key = nums[i] / max(1, t)

            for m in (key, key - 1, key + 1):
                if m in dic and abs(nums[i] - dic[m]) <= t:
                    return True
            dic[key] = nums[i]

            if i >= k:
                dic.popitem(last = False)

        return False

# 111. Minimum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :method: Breadth First Search
        :type root: TreeNode
        :rtype: int
        """
        queue = [root]
        level = 1

        while queue:
            for i in range(len(queue)):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

                if not node.left and not node.right:
                    return level
            level += 1
        return level
    def minDepth_m2(self, root):
        '''
        :method: DFS
        :time: O(n), Space: O(logn)
        '''
        if not root:
            return 0

        if not root.left and not root.right:
            return 1
        elif root.left == None:
            return self.minDepth_m2(root.right) + 1
        elif root.right == None:
            return self.minDepth_m2(root.left) + 1
        else:
            return min(self.minDepth_m2(root.left), self.minDepth_m2(root.right)) + 1

  
# 102.Binary Tree Level Order Traversal
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]

        while queue:
            size = len(queue)
            lres = []
            for i in range(size):
                node = queue.pop(0)
                lres.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(lres)
        return res


# 637. Average of Levels in Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        queue = [root]
        res = []

        while queue:
            size = len(queue)
            sm = 0
            for i in range(size):
                node = queue.pop(0)
                sm += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            avg = sm / range(size)
            res.append(avg)
        return res

# 107. Binary Tree Level Order Traversal ||
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]

        while queue:
            size = len(queue)
            lres = []
            for i in range(size):
                node = queue.pop(0)
                lres.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0, lres)
        return res


# 103. Binary Tree Zigzag Level Order Traversal
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]
        tag = 1
        while queue:
            size = len(queue)
            lres = []
            for i in range(size):
                node = queue.pop(0)
                if tag == 1:
                    lres.append(node.val)
                else:
                    lres.insert(0, node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(lres)
            tag = -tag
        return res

# 199. Binary Tree Right Side View
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :method: BFS
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        queue = [root]

        while queue:
            size = len(queue)
            lres = []
            for i in range(size):
                node = queue.pop(0)
                lres.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(lres.pop())
        return res

    def rightSideView_2(self, root):
        #method: DFS
        res = []
        self.helper(res, root, 0)
        return res

    def helper(self, res, root, level):
        if not root:
            return

        if level == len(res):
            res.append(root.val)
        
        self.helper(res, root.right, level + 1)
        self.helper(res, root.left, leve + 1)

# 116. Populating Next Right Pointers in Each Node
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if root is None:
            return

        queue = [root]
        level = 0

        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.pop(0)
                if i == size - 1:
                    node.next = None
                else:
                    node.next = queue[0]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return

    def connect(self, root):
        # method: lnked list in each level
        if root is None:
            return
        queue = [root]
        level = 0

        while queue:
            size = len(queue)
            last = None
            for i in range(size):
                node = queue.pop(0)
                if last:
                    last.next = node
                last = node
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            last.next = None
        return

# 117. Populating Next Right Pointers in Each Node ||
class Solution:
    def connect(self, root):
        # method: lnked list in each level
        if root is None:
            return
        queue = [root]
        level = 0

        while queue:
            size = len(queue)
            last = None
            for i in range(size):
                node = queue.pop(0)
                if last:
                    last.next = node
                last = node
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            last.next = None
        return


# 136. Single Number
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        st = Set()
        for num in nums:
            if num in st:
                st.remove(num)
            else:
                st.add(num)
        return st.pop()

# 100. Same Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :method: recursion
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == None and q == None:
            return True
        elif p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False    

    def isSameTree(self, p, q):
        # method: no recursion
        if p == None and q == None:
            return True

        stack = [(p, q)]
        while stack:
            lnode, rnode = stack.pop(0)

            if lnode == None and rnode == None:
                continue
            elif lnode and rnode and lnode.val == rnode.val:
                stack.append((lnode.left, rnode.left))
                stack.append((lnode.right, rnode.right))
            else:
                return False
        return True


# 101. Symmetric Tree
class Solution(object):
    def isSymmetric(self, root):
        """
        :method: no recursion
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True

        left = root.left
        right = root.right
        if left == None and right == None:
            return True
        stack = [[left, right]]
        while stack:
            lnode, rnode = stack.pop(0)

            if lnode == None and rnode == None:
                continue
            elif lnode and rnode and lnode.val == rnode.val:
                stack.append([lnode.left, rnode.right])
                stack.append([lnode.right, rnode.left])
            else:
                return False

        return True

    def isSymmetric(self, root):
        if root == None:
            return True

    def helper(self, left, right):
        if left == None and right == None:
            return True
        elif left and right and left.val == right.val:
            return self.helper(left.left, right.right) and self.helper(left.right, right.left)
        else:
            return False

# 110. Balanced Binary Tree
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.height(root) != -1
    def height(self, root):
        if root == None:
            return 0
        l = self.height(root.left)
        r = self.height(root.right)

        if l == -1 or r == -1 or abs(l - r) > 1:
            return -1
        return max(l, r) + 1

# 108. Convert Sorted Array to Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if len(nums) == 0:
            return None
        mid = (0 + len(nums) - 1) / 2
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBST(nums[0:mid])
        node.right = self.sortedArrayToBST(nums[mid + 1 : len(nums)])
        return node   


# 109. Convert Sorted List to Binary Search Tree
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if head == None:
            return None
        rt = self.midElem(head)
        midValue = rt[0]
        node = TreeNode(midValue)
        node.left = self.sortedListToBST(rt[1])
        node.right = self.sortedListToBST(rt[2])
        return node

    def midElem(self, head):
        if head == None:
            return None
        prev = None
        slow = head
        fast = head.next
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next

        if prev:
            prev.next = None
            left = head
        else:
            left = None
            
        right = slow.next
        return [slow.val, left, right]


# 112. Path Sum
class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
            return False

        if root.left == None and root.right == None and sum - root.val == 0:
            return True;

        return hasPathSum(root.left, sum - root.val) or hasPathSum(root.right, sum - root.val)


# 113. Path Sum ||
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        path = []
        self.helper(root, sum, result, path)
        return result

    def helper(self, root, sum, res, path):
        if root is None:
            return

        path.append(root.val)
        if root.left is None and root.right is None:
            if sum == root.val:    
                newlist = p
                res.append(newlist)
            return

        if root.left is not None:
            self.helper(root.left, sum - root.val, res, p)
            p.pop(len(p) - 1)

        if root.right is not None:
            self.helper(root.right, sum - root.val, res, p)
            p.pop(len(p) - 1)

    def pathSum_m2(self, root, sum):
        # method: recursion with python
        if root is None:
            return []
        res = []
        self.dfs(root, sum, [], res)

    def dfs(self, root, sum, ls, res):
        if root is None:
            return
        if root.left is None and root.right is None and sum == root.val:
            ls.append(root.val)
            res.append(ls)
        if root.left:
            self.dfs(root.left, sum - root.val, ls + [root.val], res)
        if root.right:
            self.dfs(root.right, sum - root.val, ls + [root.val], res)

    def pathSum_m3(self, root, sum):
        if not root:
            return
        if not root.left and not root.right and sum == root.val:
            return [[root.val]]
        tmp = self.pathSum_m3(root.left, sum - root.val) + self.pathSum_m3(root.right, sum - root.val)
        return [[root.val] + i for i in tmp]


# 170. Two Sum ||| - Data structure design
class TwoSum(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashtable = dict()
        

    def add(self, number):
        """
        Add the number to an internal data structure..
        :type number: int
        :rtype: void
        """
        if number in self.hashtable:
            self.hashtable[number] += 1
        else:
            self.hashtable[number] = 1


    def find(self, value):
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        :type value: int
        :rtype: bool
        """
        for key in self.hashtable:
            if key * 2 == value:
                if self.hashtable[key] >= 2:
                    return True
            else:
                if value - key in self.hashtable:
                    return True
        return False

# 105. Construct Binary Tree from Preorder and Inorder
class Solution:
    def buildTree(self, preorder, inorder):
        if len(inorder) > 0:
            mid = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[mid])
            root.left = self.buildTree(preorder, inorder[:mid])
            root.right = self.buildTree(preorder, inorder[mid + 1:])
            return root


# 106. Construct Binary Tree from Inorder and Postorder
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if len(postorder) > 0:
            mid = inorder.index(postorder.pop())
            root = TreeNode(inorder[mid])
            root.left = self.buildTree(inorder[:mid], postorder)
            root.right = self.buildTree(inorder[mid + 1 :], postorder)
            return root
        return


# 545. Boundary of Binary Tree
class Solution(object):
    def isLeaf(self, root):
        '''
        :rtype: bool
        '''
        if not root:
            return False
        return root.left == None and root.right == None

    def addLeaves(self, res, root):
        if self.isLeaf(root):
            res.append(root.val)
        else:
            if root.left:
                self.addLeaves(res, root.left)
            if root.right:
                self.addLeaves(res, root.right)

    def boundaryOfBinaryTree(self, root):
        res = []
        if not root:
            return res

        if not self.isLeaf(root):
            res.append(root.val)

        t = root.left
        while t:
            if not self.isLeaf(t):
                res.append(t.val)

            if t.left:
                t = t.left
            else:
                t = t.right
        self.addLeaves(res, root)
        t = root.right
        stack = []
        while t:
            if not self.isLeaf(t)
                stack.insert(0, t.val)
            if t.right:
                t = t.right
            else:
                t = t.left
        res += stack
        return res


# 202. Happy Number
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        set = {1}
        while n not in set:
            set.add(n)
            sum = 0
            while n > 0:
                sum += (n%10)**2
                n = n / 10
            if sum == 1:
                return True
            n = sum
        return n == 1

    def isHappy(self, n):
        set = {1}
        while n not in set:
            set.add(n)
            n = sum(int(x)**2 for x in str(n))
        return n == 1


# 258. Add Digits
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        if num is None:
            return None
        numStr = str(num)
        num = 0
        while len(numStr) != 1:
            for i in range(len(numStr)):
                num += int(numStr[i])
            numStr = str(num)

        return num


# 263. Ugly Number
class Solution(object):
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num <= 0: return False
        while num % 2 == 0:
            num /= 2

        while num % 3 == 0:
            num /= 3

        while num % 5 == 0:
            num /= 5

        return num == 1


# 264. Ugly Number ||
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        ugly = [1]
        i2 = i3 = i5 = 0
        for i in range(n - 1):
            next_i2, next_i3, next_i5 = ugly[i2]*2, ugly[i3]*3, ugly[i5]*5
            next_ugly_no = min(next_i2, next_i3, next_i5)
            if next_ugly_no == next_i2:
                i2 += 1

            if next_ugly_no == next_i3:
                i3 += 1

            if next_ugly_no == next_i5:
                i5 += 1

            ugly.append(next_ugly_no)

        return ugly[-1]

# 242. Valid Anagram
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t): return False
        if len(l) == 0: return True
        set = {}
        for i in range(len(s)):
            k = s[i]
            if k not in set:
                set[k] = 1
            else:
                set[k] += 1

        # check and pop
        for i in range(len(t)):
            k = t[i]
            if k not in set:
                return False
            else:
                if set[k] == 1:
                    set.pop(k)
                else:
                    set[k] -= 1
        return set == {}

    def isAnagram_m2(self, s, t):
        return sorted(s) == sorted(t)
    
    def isAnagram_m3(self, s, t):
        if len(s) != len(t): return False
        if len(t) == 0: return True
        res = [0 for i in range(26)]
        for i in range(len(s)):
            res[ord(s[i]) - ord('a')] += 1
            res[ord(t[i]) - ord('a')] -= 1
        for i in range(26):
            if res[i] != 0:
                return False
        return True

# 49. Group Angarams
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dic = {}
        res = []
        for word in strs:
            st = str(sorted(word))
            if st in dic:
                dic[st].append(word)
            else:
                dic[st] = [word]
        for key in dic:
            if(len(dic[key]) >= 1):
                res.append(dic[key])

        return res

# 249. Group Shifted Strings
class Solution(object):
    def groupStrings(self, strings):
        """
        :type strings: List[str]
        :rtype: List[List[str]]
        """
        dic = {}
        res = []
        for i in range(len(strings)):
            key = self.helper(strings[i])
            if key not in dic:
                dic[key] = [strings[i]]
            else:
                dic[key].append(strings[i])
        for key in dic:
            res.append(dic[key])
        return res

    def helper(self, str):
        if len(str) == 0:
            return ""
        if str[0] == 'a':
            return str
        else:
            newstr = ''
            for i in range(len(str)):
                tmp = ord(str[i]) - ord(str[0]) + ord('a')
                if tmp < 97:
                    tmp += 26
                newstr += chr(tmp)
                
            return newstr

    def isShift(self, s, t):
        # method: 
        if len(s) != len(t) or len(s) == 0 or len(t) == 0:
            return False
        for i in range(len(s - 1)):
            if ord(s[i]) - ord(t[i]) != ord(s[i+1]) - ord(t[i+1]):
                return False
        return True 

# 349. Intersection of Two Arrays
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
# 204. Count Primes
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 2: return 0 # 1 is not a prime

        res = [True] * n

        for p in range(2, int(n ** 5) + 1):
            if A[i]:
                for j in range(p ** 2, n, p):
                    A[j] = False

        return sum(res) - 2

# 349. Intersection of Two Arrays
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if len(nums1) == 0 or len(nums2) == 0 or nums1 == None or nums2 == None:
            return []
        res = {}
        set = set(nums1)
        for num in nums2:
            if num in set:
                res.add(num)
        return list(res)

# 350. Intersection of Two Arrays ||
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        # method 1: hashset
        # time: O(n), Space: O(n)
        if len(nums1) == 0 or len(nums2) == 0 or nums1 == None or nums2 == None:
            return []
        dic = {}
        res = []
        for num in nums1:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
        for num in nums2:
            if num in dic and dic[num] > 0:
                res.append(num)
                dic[num] -= 1
        return res


# 94. Binary Tree Inorder Traversal
class Solution(object):
    def __init__(self):
        self.res = []
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root:
            self.inorderTraversal(self.left)
            self.res.append(root.val)
            self.inorderTraversal(self.right)
        return self.res

    def inorderTraversal(self, root):
        # method2: iterative
        stack = []
        res = []
        stack.append((root,False))

        while stack:
            node, flag = stack.pop()

            if node:
                if flag:
                    res.append(node.val)
                else:
                    stack.append((node.right, False))
                    stack.append((node, True))
                    stack.append((node.left, False))

        return res

# 144. Binary Tree Preorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def __init__(self):
        self.res = [] # a global value(property) to store results
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root:
            self.res.append(root.val)
            self.preorderTraversal(root.left)
            self.preorderTraversal(root.right)

        return self.res

    def preorderTraversal_2(self, root):
        # method 2: iteratively
        res = []
        stack = [(root, False)]
        while stack:
            node, flag = stack.pop()
            if node:
                if flag:
                    res.append(node.val)
                else:
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                    stack.append((node, True))

        return res
    def preorderTraversal_3(self, root):
        # method 3: another recursive
        res = []
        if root:
            res.append(root.val)
            res += self.preorderTraversal_3(root.left)
            res += self.preorderTraversal_3(root.right)
        return res



# 145. Binary Postorder Traversal
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        stack = [(root, False)]
        while stack:
            node, flag = stack.pop()
            if node:
                if flag:
                    res.append(node.val)
                else:
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
        return res

# 155. Min Stack
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        curMin = self.getMin()
        if curMin == None or x < curMin:
            curMin = x
        self.stack.append((x, curMin))

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()
        

    def top(self):
        """
        :rtype: int
        """
        if len(self.stack) == 0:
            return None
        else:
            return self.list[-1][0]
        

    def getMin(self):
        """
        :rtype: int
        """
        if len(self.stack) != 0:
            return self.stack[-1][1]
        else:
            return None


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


# 48. Rotate Image
class Solution(object):
    def rotate(self, nums):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        nums.reverse()
        length = len(nums)
        for i in range(length):
            for j in range(i):
                nums[i][j], nums[j][i] = nums[j][i], nums[i][j]
                
    def rotate_left(self, nums):
        nums.reverse()
        length = len(nums)
        for i in range(length):
            for j in range(length - i):
                nums[i][j], nums[length - 1 - j][length - 1 - i] = nums[length - 1 - j][length - 1 - i], nums[i][j]


# 17. Letter Combinations of a Phone Number
# method: backtracking
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        dic = {
            '1': "",
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }
        res = [""]
        for digit in digits:
            lst = dic[digit]
            temp = []
            for char in lst:
                for str in res:
                    temp.append(str + char)
            res = temp
        return res       

# 39. Combination Sum
# method: Backtracking
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        
#-------------- TO DO LIST -----------------


# 239. Sliding Window Maximum
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """




# 313. Super Ugly Number
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """

    
    def isSuperUgly(self, n, primes):
        for prime in primes:
            while num % prime == 0:
                num /= prime
            if num == 1:
                return True
        return False


# 437. Path Sum |||


# 133. Clone Graph
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):


# 222. Count Complete Tree Nodes
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
