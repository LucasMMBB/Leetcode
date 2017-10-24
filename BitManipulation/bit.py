# 191. Number of 1 Bits
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        return bin(n).count("1")

    def hammingWeight_m2(self, n):
        count = 0
        while n != 0:
        	n = n & (n - 1)
        	count += 1
        return count