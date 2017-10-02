# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        res = []
        stack = [(root, 0, 0)]
        while stack:
            node, length, flag = stack.pop()
            if not node.left and not node.right:
                res.append(length)
            if node.left and node.right:
                if node.val == node.left.val and node.val == node.right.val:
                    stack.append((node.left, length + 2))
                    stack.append((node.right, length + 2))
                else:
                    if node.left:
                        if node.val == node.left.val:
                            stack.append((node.left, length + 2))
                        else:
                            res.append(length)
                            stack.append((node.left, 0))
                    if node.right:
                        if node.val == node.right.val:
                            stack.append((node.right, length + 2))
                        else:
                            res.append(length)
                            stack.append((node.right, 0))
            if node.left:
                if node.val == node.left.val:
                    stack.append((node.left, length + 1))
                else:
                    res.append(length)
                    stack.append((node.left, 0))
            if node.right:
                if node.val == node.right.val:
                    stack.append((node.right, length + 1))
                else:
                    res.append(length)
                    stack.append((node.right, 0))
        return max(res)