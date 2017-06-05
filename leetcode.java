/*\
|*| Leetcode answers
|*| Author: Maoxu
|*| Description: share my answers to whoever needs
|*| Create date: April 3rd, 2017
\*/
// 71. Simplify path
public class Solution {
    public String simplifyPath(String path) {

     	if(path == null || path.length() == 0){return new String();}

     	path = path.trim();
     	Deque<String> deque = new LinkedList<String>();

     	for(String cur: path.splict("/")){
     		if(cur.equals("..")){
     			if(!deque.isEmpty()){
     				deque.pollLast();
     			}
     		}else if(!cur.equals(".")&&!cur.equals("")){
     			deque.offerLast(cur);
     		}
     	}

     	StringBuilder sb = new StringBuilder();
     	while(!deque.isEmpty()){
     		sb.append("/").append(deque.pollFirst());
     	}

     	return sb.length == 0? new String("/") : sb.toString();
    }
}

// 84. Largest Rectangle in Histogram
public class Solution {
    public int largestRectangleArea(int[] heights) {

        if(heights == null || heights.length == 0){return 0;}

        Deque<Integer> stack = new LinkedList<Integer>();//store the index
        int max = 0;

        for(int i=0;i<=heights.length;i++){
        	//Each elem will be push once and poll once
        	//1. Check whether this elem can be pushed into the stack
        	int curVal = i == heights.length?0:heights[i];
        	while(!stack.isEmpty() && heights[stack.peekLast()]>=curVal){
        		int height = heights[stack.pollLast()];
        		int leftBound = stack.isEmpty()?0:stack.peekLast() + 1;
        		int rightBound = i;
        		max = Math.max(max, height*(rightBound - leftBound));
        	}

        	//2. Push the elem into the stack
        	stack.addLast(i);
        }

        return max;

    }
}

// 42. Trapping rain water
public class Solution {
    public int trap(int[] arr) {
    	if(arr == null || arr.length <=2 ) {return 0;}

    	//Two scanners
    	int left = 0;
    	int right = arr.length - 1;
    	int sum = 0;

    	//Two walls
    	int leftMax = 0;
    	int rightMax = 0;
    	while(left<=right){
    		//Move lower wall first: Guarantee middle region can trap water
    		if(leftMax<=rightMax){
    			leftMax = Math.max(leftMax, arr[left]);
	    		if(arr[left]<leftMax){
	    			sum+=leftMax - arr[left];
	    			left++;
	    		}
    		}else{
    			rightMax = Math.max(rightMax, arr[right]);
    			if(arr[right]<rightMax){
    				sum += rightMax - arr[right];
    				right--
    			}
    		}
    	}
    	return sum;
	}
}


// 75. sort colors
public class Solution {

    public void sortColors(int[] nums) {
	
        if(nums == null || nums.length <= 1){
		  return;    
	      //System.out.println("fuck you!!");
		}
	    int lb = 0;
	    int rb = nums.length - 1;
	    int i = 0; // scanner index
	    while(i<=rb){
	      if(nums[i]==2){
	      	swag(nums, i, rb);
	        rb-=1;
	      }else if(nums[i]==0){
	      	swag(nums, i, lb);
	        lb+=1;
	        i+=1;
	      }else{
	      	i+=1;
	      }
	    }
    }// sortCOlors ends

 	public void sortColors_M2(int[] nums) {
	    if(nums == null || nums.length <= 1){
	      System.out.println("fuck you!!");
	    }
	    
	    int countzero = 0;
	    int countone = 0;
	    int counttwo = 0;
	    
	    for(int i = 0;i<nums.length;i++){
	      if(nums[i]==0){
	      	countzero+=1;
	      }else if(nums[i]==1){
	      	countone+=1;
	      }else{
	      	counttwo+=1;
	      }
	    }
	    
	    for(int j=0;j<nums.length;j++){
	      if(j<countzero){
	      	nums[j] = 0;
	      }else if(j>=countzero && j<(countzero+countone)){
	      	nums[j] = 1;
	      }else{
	      	nums[j] = 2;
	      }
	    }
    
    	printArray(nums);
    }   

    private void printArray(int[] anArray) {
      for (int i = 0; i < anArray.length; i++) {
         if (i > 0) {
            System.out.print(", ");
         }
         System.out.print(anArray[i]);
      }
   }



    private void swag(int[] arr, int a, int b){
	    if(arr[a]!=arr[b]){
	   		int temp = arr[b];
	      	arr[b]=arr[a];
	      	arr[a]=temp;
	    }
  }

}

// 134. Gas Station
public class Solution {

    // Time Comflexity: O(n)
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int start = 0;
        int total = 0;
        int tank = 0;
        for(int i=0;i<gas.length;i++){
            tank += gas[i] - gas[i];
            if( tank < 0 ){
                start = i + 1;
                total += tank;
                tank = 0;
            }
        }
        return (total+tank<0)?-1:start;
    }// canCompleteCircuit ends here
}

// 20. Valid Parentheses
public class Solution {
    public boolean isValid(String s) {
        if( (s == null) || s.length() == 0){
            return true;
        }

        if( s.length()%2 == 1 ){
            return false;
        }

        char[] str = s.toCharArray();
        Deque<Character> stack = new LinkedList<Character>();

        for(char ch : str){
            // Case 1: left parenthese --> put into stack
            if(ch == '(' || ch == '[' || ch == '{'){
                stack.offerLast(ch);
            } else {
                //Case 2: right parenthese --> check stack.peer()
                if(stack.isEmpty()){return false;}
                char left = stack.pollLast();
                if( (ch == ')' && left == '(') ||
                    (ch == ']' && left == '[') ||
                    (ch == '}' && left == '{') ){
                    continue;
                }else{
                    return false;
                }

            }

        }

        return stack.isEmpty();


    }
}


// 22. Generate Parentheses
// 32. Longest Valid Parentheses
public class Solution {
    public int longestValidParentheses(String s) {
        if( (s == null) || s.length() == 0 || s.length() == 1){
            return 0;
        }

        int mx = 0;
        char[] str = s.toCharArray();
        Deque<Integer> stack = new LinkedList<Integer>();

        for(int i=0;i<s.length();i++){
            // Case 1: left parenthese --> put into stack
            if(str[i] == '('){
                stack.offerLast(i);
            } else {
                //Case 2: right parenthese --> check stack.peer()
                if(stack.isEmpty()){ 
                    stack.offerLast(i);
                }else{
                    if(str[stack.peekLast()] == ')'){
                      stack.offerLast(i);
                    }else{
                      stack.removeLast();
                    }
                }
            }
        } // for ends
        
        if(stack.isEmpty()){return s.length();}
        else{
            int a = s.length();int b = 0;
            while(!stack.isEmpty()){
                b = stack.peekLast();
                stack.removeLast();
                mx = Math.max(mx,a-b-1);
                a = b;
            }// while ends
            mx = Math.max(mx,a);
            return mx;
        }
    }// function ends

    public int longestValidParentheses_better(String s){
        if( (s == null) || s.length() == 0 || s.length() == 1){
            return 0;
        }

        int mx = 0; int left = -1;
        char[] str = s.toCharArray();
        Deque<Integer> stack = new LinkedList<Integer>();

        for(int i=0;i<s.length();i++){
            // Case 1: left parenthese --> put into stack
            if(str[i] == '('){
                stack.offerLast(i);
            } else {
                //Case 2: right parenthese --> check stack.peer()
                if(stack.isEmpty()){ 
                    stack.offerLast(i);
                }else{
                    if(str[stack.peekLast()] == ')'){
                      stack.offerLast(i);
                    }else{
                      stack.removeLast();
                    }
                }
            }
            // calulate max length while processing data
            if(stack.isEmpty()) mx = Math.max(mx, i - left);
            else mx = Math.max(mx, i-stack.peekLast());

        } // for ends
        return mx;
    }

}

// 278. First Bad Version
/* The isBadVersion API is defined in the parent class VersionControl.
      boolean isBadVersion(int version); */

public class Solution extends VersionControl {
    public int firstBadVersion(int n) {
        if(n <= 0){return Integer.MIN_VALUE;}

        int start = 1; int end = n;

        while(start < end - 1){
            int mid =  start + (end - start)/2; // In case of overflow

            if(!isBadVersion(mid)){
                // Case 1: Good version  --> go right
                start = mid;
            }else{
                // Case 2: Bad version --> go left
                end = mid;
            }
        }// while ends
        return isBadVersion(start) ? start : end;
    }
}

// 74. Search a 2D Matrix
public class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix == null || matrix.length == 0){return false;}
        if(matrix[0] == null || matrix[0].length == 0){return false;}

        int m = matrix.length;
        int n = matrix[0].length;

        int start = 0;
        int end = m*n-1;

        // One-D binary search
        while(end >= start){
            int mid = start + (end -  start)/2;
            if(matrix[mid/n][mid%n]==target){
                return true;
            }else if(matrix[mid/n][mid%n] > target){
                end = mid -1;
            }else{
                start = mid +1;
            }
        }// while ends

        return false;
    }
}

// 240. Search a 2D Matrix TWO
public class Solution {
    public boolean searchMatrix_TWO(int[][] matrix, int target) {
        // Corned Case Checked
        if(matrix == null || matrix.length == 0){return false;}
        if(matrix[0] == null || matrix[0].length == 0){return false;}

        int m = matrix.length;
        int n = matrix[0].length;

        return binarySearch(matrix, target, 0, 0, m-1, n-1);
    }

    private boolean binarySearch(int[][] matrix, int target, int startX, int startY, int endX, int endY){
        if(startX > endX || startY > endY){return false;}

        int midX = startX + (endX - startX) / 2;
        int midY = startY + (endY - startY) / 2;

        if(matrix[midX][midY] == target){
            // case 1: found
            return true;
        }else if (matrix[midX][midY] > target){
            // case 2: larger than target, go into left or up
            return binarySearch(matrix, target, startX, startY, endX, midY - 1) ||
                   binarySearch(matrix, target, startX, startY, midX - 1, endY);
        }else{
            // case 3: less than target, go into right or down
            return binarySearch(matrix, target, startX, midY + 1, endX, endY) ||
                   binarySearch(matrix, target, midX + 1, startY, endX, endY);
        }// if ends
    }

    public boolean searchMatrix_Three(int[][] matrix, int target){
        
        if(matrix == null){return false;}
        int row = matrix.length;
        if(row == 0 || matrix[0] == null){return false;}
        int col = matrix[0].length;
        if(col == 0){return false;}

        // Start from the top-right point
        int curRow = 0;
        int curCol = col - 1;

        while(curRow < row && curCol >= 0){
            if(matrix[curRow][curCol] == target){
                // Case 1: found
                return true;
            }else if(matrix[curRow][curCol] > target){
                // Case 2: larger than target --> go left
                curCol --;
            }else{
                // Case 3: less than target --> go right
                curRow ++;
            }
        }// while ends
        return false;

    }
}

// 367. Valid Perfect Square
public class Solution {
    public boolean isPerfectSquare(int num) {
        // time complexity is O(sqrt(n))
        int i = 1;
        while(num > 0){
            num -= i;
            i += 2;
        }
        return num == 0;       
    }

    public boolean isPerfectSquare_m2(int num){
        int low = 1, high = num;
        while(low<=high){
            long mid = low + (high - low) / 2;
            if(mid * mid == num){
                return true;
            }else if( mid*mid < num ){
                low = (int)mid + 1;
            }else{
                high = (int)mid - 1;
            }
        }// while ends
        return false;
    }
}

// 69. Sqrt(x)
public class Solution {
    public int mySqrt(int x) {
        if( x == 0 ){return 0;}
        int left = 1, right = Integer.MAX_VALUE;
        while(true){
            int mid = left + ( right - left ) / 2;
            if(mid > x / mid){
                right = mid -1;
            }else{
                if(mid + 1 > x/(mid+1)){
                    return mid;
                }
                left = mid + 1;
            }
        }// while ends
    }
}


// 231. Power of Two
public class Solution {
    public boolean isPowerOfTwo(int n) {
        if(n <= 0){return false;}
        int bitCount = 0;

        for(int i = 0; i < 32; i++){
            if( (n&1) == 1){
                bitCount++;
            }
            n = n >> 1;
        }

        return bitCount == 1;
    }

    public boolean isPowerOfTwo_m2(int n){
        // Example 8 == 1000, 8-1 = 0111, so 1000 & 0111 = 0; Time complexity is O(1)
        return n > 0 && ( n & ( n - 1 ) ) == 0;
    }
}


// 326. Power of Three
public class Solution {
    public boolean isPowerOfThree(int n) {
        if(n<=0){return false;}
        while(n%3==0){
            n = n/3;
        }
        return n == 1;       
    }
    public boolean isPowerOfThree_M2(int n) {
        // Time complexity: O(1), space complexity: O(1)
        return n > 0 && 1162261467 % n == 0;
    }

}


// 342. Power of Four
public class Solution {
    public boolean isPowerOfFour(int num) {
        if(n<=0){return false;}
        while(n%4==0){
            n = n/4; // or n /= 4
        }
        return n == 1;
    }
    //1.073741824E9
    public boolean isPowerOfFour_M2(int n) {
        // Time complexity: O(1), space complexity: O(1)
        return n > 0 && 1073741824 % n == 0;
    }
}


// 190. Reverse Bits
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int result = 0;
        for(int i=0;i<32;i++){
            result += (n&1);
            n >>= 1;
            if(i<31){result<<=1;}
        }
        return result;   
    }
}

// 191. Number of 1 Bits
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        if(n==0){return 0;}
        int count = 0;
        for(int i=0;i<32;i++){
            if( (n&1) == 1){
                count++;
            }
            n>>>=1;
        }
        return count;        
    }
}


// 338. Counting Bits
public class Solution {
    public int[] countBits(int num) {
        int[] f = new int[num + 1];
        for(int i=1;i<=num;i++){
            f[i]=f[i>>1]+(i&1);
        }
        return f;
    }
}

// 102. Binary Tree Level Order Traversal
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        
    }
}

// 33. Search in Rotated Sorted Array
public class Solution {
    public int search(int[] nums, int target) {
        if(nums == null || nums.length == 0) return -1;
        for(int i=0;i<nums.length;i++){
            if(nums[i] == target){
                return i;
            }
        }
        return -1;
    }

    public int search2(int[] nums, int target){
        if(nums == null || nums.length == 0)
            return -1;
        int start = 0;
        int end = nums.length - 1;

        while(start + 1 < end){
            int mid = start + (end-start)/2;

            if(nums[mid] == target)
                return mid;

            if(nums[mid] > nums[start]){
                if(target >= nums[start] && target <= nums[mid]){
                    end = mid;
                }else{
                    start = mid;
                }
            }else if(nums[mid] < nums[start]){
                if(target >= nums[mid] && target <= nums[end]){
                    start = mid;
                }else{
                    end = mid;
                }
            }else{
                start ++;// duplicates
            }
        }
        if(nums[start] == target)
            return start;
        if(nums[end] ==  target)
            return end;
        return -1;
    }// method ends
}


// 81. Search in Rotated Sorted Array 2
// Regular: O(log(n) and worst is O(n)
public class Solution {
    public boolean search(int[] nums, int target) {
        if(nums == null || nums.length == 0)
            return false;
        int start = 0;
        int end = nums.length - 1;

        while(start + 1 < end){
            int mid = start + (end-start)/2;

            if(nums[mid] == target)
                return true;

            if(nums[mid] > nums[start]){
                if(target >= nums[start] && target <= nums[mid]){
                    end = mid;
                }else{
                    start = mid;
                }
            }else if(nums[mid] < nums[start]){
                if(target >= nums[mid] && target <= nums[end]){
                    start = mid;
                }else{
                    end = mid;
                }
            }else{
                start ++;// duplicates
            }
        }
        if(nums[start] == target || nums[end] == target)
            return true;
        return false;
    }
}

// 153. Find Minimum in Rotated Sorted Array
// Regular: O(Log(n))
public class Solution {
    public int findMin(int[] nums) {
        if(nums == null || nums.length == 0)
            return -1;
        int start = 0;
        int end = nums.length - 1;

        while(start + 1 < end){
            int mid = start + (end-start)/2;

            if(nums[mid] > nums[start]){
                if(nums[mid]<nums[end]){
                    return nums[start];
                }else{
                    start = mid;
                }
            }else if(nums[mid] < nums[start]){
                end = mid;
            }else{
                start ++;// duplicates
            }
        }
        if(nums[start] <=  nums[end]){
            return nums[start];
        }
        else{
            return nums[end];
        }
    }
}


// 46. Permutations
public class Solution {
    public List<List<Integer>> permute(int[] nums) {
        
    }
}

// 31. Next Permutation
public class Solution {
    public void nextPermutation(int[] nums) {
        
    }
}