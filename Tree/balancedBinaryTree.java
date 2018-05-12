 public class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
 }

// leetcode question 110
class Solution {
    public boolean isBalanced(TreeNode root) {
        if(root == null){return true;}
        return maxDepth(root)!=-1;
    }
    
    public int maxDepth(TreeNode root){
        if(root == null) {return 0;}
        int left = maxDepth(root.left);
        if(left == -1) {return -1;}
        int right = maxDepth(root.right);
        if(right == -1) {return -1;}
        
        if(Math.abs(left - right) > 1){return -1;}
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
