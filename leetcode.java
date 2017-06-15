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


// 31. Next Permutation
/**
 * Algorithme:
 *      1. Start scanning from the right
 *      2. Find a digit D1 which is not in ascending order from right
 *      3. If all the digiits from right to left are ascending then print None
 *      4. Find a digit D2 which is right of D1, such that it is the smallest number greater than D1
 *      5. Swap D1 and D2
 *      6. Now sort the digits right of D1's original position
 */
public class Solution {
    public void nextPermutation(int[] nums) {
        if(nums.length <= 1)
            return;

        int i = nums.length - 1;

        while(i>=1){
            if(nums[i]>nums[i-1])
                break;
            i--;
        }
        
        if(i!=0)
            swap(nums,i-1);

        reverse(nums, i);
    }

    private void swap(int[] a, int i){
        for(int j = a.length - 1; j>i; j--){
            if(a[j]>a[i]){
                int t = a[j];
                a[j] = a[i];
                a[i] = t;
                break;
            }
        }
    }// swap ends

    private void reverse(int[] a, int i){
        int first = i;
        int last = a.length - 1;
        while(first<last){
            int t = a[first];
            a[first] = a[last];
            a[last] = t;
            first ++;
            last --;
        }
    }
}



// 46. Permutations
public class Solution {
    public List<List<Integer>> permute(int[] nums) {
        // method 1
        // corner case checked
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> list = new ArrayList<Integer>();
        dfsHelper(nums, list, res);
        return res;
    }

    private void dfsHelper(int nums[], List<Integer> list, List<List<Integer>> result){
        if(list.size()==nums.length){
            result.add(new ArrayList<Integer>(list));
            return;
        }

        for(int i=0;i<nums.length;i++){
            if(!list.contains(nums[i])){
                list.add(nums[i]);
                dfsHelper(nums, list, result);// next position
                // empty last position for next iteration
                list.remove(list.size()-1);
            }
        }
    }

    /*
        basic method. O(n!)
    */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if(nums == null || nums.length == 0)
            return res;

        res.add(new ArrayList<Integer>());

        for(int i = 0; i < nums.length; i++){
            List<List<Integer>> nextRes = new ArrayList<List<Integer>>();
            for(List<Integer> list: res){
                // for each list in res
                for(int j=0; j<list.size()+1; j++){
                    // copy a list to nextList
                    List<Integer> nextList = new ArrayList<Integer>(list);
                    nextList.add(j, nums[i]);
                    nextRes.add(nextList);
                }
            }
            res = nextRes;// Move to next level
        }
        return res;
    
    }

    public List<List<Integer>> permute2(int[] nums) {
        // method 3: best way so far
        // corner case checked
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> list = new ArrayList<Integer>();
        dfsHelper2(nums, list, res);
        return res;
    }

}


// 47. Permutation 2
public class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        // Corner case CHecked
        /*
        METHOD: DFS, and using a HashSet<Integer> in each pos to remove duplicates
        */
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        dfsHelper(res, nums, 0);
        return res;
    }

    private void dfsHelper(List<List<Integer>>, int[] nums, int pos){

        if(pos == nums.length){
            List<Integer> list =  new ArrayList<Integer>();
            for(int num: nums){
                list.add(num);
            }
            res.add(list);
            return;
        }

        Set<Integer> used = new HashSet<Integer>();
        for(int i = pos; i < nums.length; i++){
            if(used.add(nums[i])){
                swap(nums, i, pos);
                dfsHelper(res, nums, pos+1);
                swap(nums, i, pos);
            }
        }

    }
    private int[] swap(int[] nums, int a, int b){
        int t = nums[a];
        nums[a] = nums[b];
        nums[b] = t;
        return nums;
    }
}


// 60. Permutation Sequence
public class Solution {
    public String getPermutation(int n, int k) {
        int pos = 0;
        List<Integer> numbers = new ArrayList<>();
        int[] factorial = new int [n+1];
        StringBuilder sb = new StringBuilder();

        // create an array of factorial lookup
        int sum = 1;
        factorial[0] = 1;
        for(int i=1;i<=n;i++){
            sum*=i;
            factorial[i]=sum;
        }

        // create a list of numbers to get indices
        for(int i=1;i<=n;i++){
            numbers.add(i);
        }

        k--;

        for(int i=1;i<=n;i++){
            int index = k/factorial[n-i];
            sb.append(String.valueOf(numbers.get(index)));
            numbers.remove(index);
            k-=index*factorial[n-i];
        }
        return String.valueOf(sb);
    }
}

// 44. Wildcard Matching
public class Solution {
    public boolean isMatch(String str, String pattern) {
        int s=0, p=0, match=0, ss = -1;
        while(s<str.length()){
            if(p<pattern.length() && 
                (pattern.charAt(p)=='?'
                || str.charAt(s) == pattern.charAt(p))){
                // advancing both pointers
                s++;p++;
            }else if(p<pattern.length()
                && pattern.charAt(p) == '*'){
                // * found, only advancing pattern pointer
                ss = p;
                match = s;
                p++;
            }else if(ss != -1){
                // last pattern pointer was *, advancing string pointer
                p = ss + 1;
                match++;
                s = match;
            }else{
                return false;
            }
        }

        while(p<pattern.length() && pattern.charAt(p) == '*')
            p++;

        return p ==  pattern.length();
    }
}

// 10. Regular Expression Matching
public class Solution {
    public boolean isMatch(String s, String p) {

            if (s == null || p == null) {
                return false;
            }
            boolean[][] dp = new boolean[s.length()+1][p.length()+1];
            dp[0][0] = true;
            for (int i = 0; i < p.length(); i++) {
                if (p.charAt(i) == '*' && dp[0][i-1]) {
                    dp[0][i+1] = true;
                }
            }
            for (int i = 0 ; i < s.length(); i++) {
                for (int j = 0; j < p.length(); j++) {
                    if (p.charAt(j) == '.') {
                        dp[i+1][j+1] = dp[i][j];
                    }
                    if (p.charAt(j) == s.charAt(i)) {
                        dp[i+1][j+1] = dp[i][j];
                    }
                    if (p.charAt(j) == '*') {
                        if (p.charAt(j-1) != s.charAt(i) && p.charAt(j-1) != '.') {
                            dp[i+1][j+1] = dp[i+1][j-1];
                        } else {
                            dp[i+1][j+1] = (dp[i+1][j] || dp[i][j+1] || dp[i+1][j-1]);
                        }
                    }
                }
            }
            return dp[s.length()][p.length()];

    }
}


// 53. Maximum Subarray
// Time and space complexity O(n) 
public class Solution {
    public int maxSubArray(int[] nums) {
        int maxsf = nums[0]; // max sum so far
        int maxi = nums[0]; // max sum is in current index

        for (int i=1; i<nums.length; i++) {
            maxi = Math.max(maxi+nums[i], nums[i]);
            maxsf = Math.max(maxi, maxsf);
        }
        return maxsf;
    }
}


// 121. Best Time to Buy and Sell Stock
/**
 * Kadane's algorithm - O(n)
 */
public class Solution {
    public int maxProfit(int[] prices) {
        int maxCur = 0, maxSoFar = 0;
        for(int i = 1; i < prices.length; i++) {
            maxCur = Math.max(0, maxCur += prices[i] - prices[i-1]);
            maxSoFar = Math.max(maxCur, maxSoFar);
        }
        return maxSoFar;
    }

    // Sample test case {4,7,1,8,20,15}
    public int maxProfit_2(int[] prices){
        int profit = 0;
        int minp = Integer.MAX_VALUE;//MINIMUM PRICE

        for (int i=0; i<prices.length; i++) {
            profit = Math.max(profit, prices[i]-minp);
            minp = Math.min(prices[i], minp);
        }
        return profit;
    }
}


// 122. Best Time to Buy and Sell Stock ||
/**
 * Time complexity: O(n)
 * Space complexity: O(1): constant space required.
 * Use Greedy algorithm(
 * Sample test case {100, 80, 120, 130, 70, 60, 100, 125}
 * The sequences of buy-sell is : 
 *      80(b)->120(s)->120(b)->130(s)->60(b)->100(s)->100(b)->125(s)
 */
public class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0)
            return 0;
        int profit = 0;
        for (int i = 0; i < prices.length-1; i++) {
            if(prices[i] < prices[i + 1])
                profit += prices[i + 1] - prices[i];
        }
        return profit;
    }
}

// 123. Best Time to Buy and Sell Stock |||
/**
 * Time complexity: O(n), Space complexity: O(1)
 */
public class Solution {
    public int maxProfit(int[] prices) {
        int buyOne = Integer.MIN_VALUE, buyTwo = Integer.MIN_VALUE;
        int sellOne = 0, sellTwo = 0;

        for(int i : prices){
            sellTwo = Math.max(sellTwo, buyTwo + i);
            buyTwo = Math.max(buyTwo, sellOne - i);
            sellOne = Math.max(sellOne, buyOne + i);
            buyOne = Math.max(buyOne, -i);
        }
        return sellTwo;
    }
}

// 188. Best Time to Buy and Sell Stock IV
public class Solution {
    public int maxProfit(int k, int[] prices) {
        // To be done after more understanding the Dynamic Programming and Recursion
    }

}


// 152. Maximum Product Subarray
/**
 * Time complexity: O(n); Space complextiy: O(1) - const required
 */
public class Solution {
    public int maxProduct(int[] nums) {
        if(nums == null || nums.length == 0)
            return 0;

        int max = nums[0]; // maximum product summ so far
        int maxCurPre = nums[0], minCurPre = nums[0];
        int maxCur = 1, minCur = 1;
        for(int i=1; i < nums.length; i++){
            maxCur = Math.max(Math.max(maxCurPre * nums[i], minCurPre * nums[i]), nums[i]);
            minCur = Math.min(Math.min(maxCurPre * nums[i], minCurPre * nums[i]), nums[i]);
            max = Math.max(max, maxCur);
            maxCurPre = maxCur;
            minCurPre = minCur;
        }
        return max;
    }
}


// 198. House Robber
/**
 * Dynamic Programming
 * Time complexity: O(n);  Space complexity: O(1)
 */
public class Solution {
    public int rob(int[] nums) {
        if(nums == null || nums.length == 0)
            return 0;
        int maxSum = nums[0], maxSum0 = 0, maxSum1 = nums[0];
        for(int i = 1; i < nums.length; i++){
            maxSum = Math.max( maxSum1, nums[i] + maxSum0);
            maxSum0 = maxSum1;
            maxSum1 = maxSum;
        }
        return maxSum;
    }
}


// 213. House Robber ||
public class Solution {
    public int rob(int[] nums) {
        if(nums == null || nums.length == 0)
            return 0;
        return Math.max(robHelper(nums, 1, nums.length-1), robHelper(nums, 0, nums.length-2));
    }

    public int robHelper(int[] nums, int s, int e) {// i/e: start/end indices
        if(nums == null || nums.length == 0)
            return 0;
        if(nums.length == 1)
            return nums[0];
        int maxSum = nums[s], maxSum0 = 0, maxSum1 = nums[s];
        for(int i = s+1; i < e + 1; i++){
            maxSum = Math.max( maxSum1, nums[i] + maxSum0);
            maxSum0 = maxSum1;
            maxSum1 = maxSum;
        }
        return maxSum;
    }
}


// 62. Unique Paths
/**
 * Dynamic programming
 * Time complexity: O(mn); space: O(mn)
 */
public class Solution {
    public int uniquePaths(int m, int n) {
        if(m == 1 || n == 1)
            return 1;
        int[][] res = new int[m][n];
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(i == 0 || j == 0){
                    res[i][j] = 1;
                }else{
                    res[i][j] = res[i-1][j] + res[i][j-1];
                }
            }
        }
        return res[m-1][n-1];
    }

    public int uniquePaths_2(int m, int n){
        // Math method
    }
}


//63. Unique Paths ||
/**
 * Dynamic programming
 * Time complexity: O(n*m); Space: O(n*m)
 */
public class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {

        int rows = obstacleGrid.length, cols = obstacleGrid[0].length;
        if(obstacleGrid == null || obstacleGrid[0][0] == 1 || obstacleGrid.length == 0 || rows == 0 || cols == 0)
            return 0;
        int [][] res = new int[rows][cols];
        for(int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                // condition
                if(obstacleGrid[r][c] == 1){
                    res[r][c] = -1;
                }else{
                    if(r == 0 && c == 0){
                        res[0][0] = 1;
                    }else if(r == 0 && c != 0){
                        if(obstacleGrid[r][c-1] == -1){
                            res[r][c] = -1;
                        }else{
                            res[r][c] = res[r][c-1];
                        }
                    }else if(r != 0 && c == 0){
                        if(obstacleGrid[r-1][c] == -1){
                            res[r][c] = -1;
                        }else{
                            res[r][c] = res[r-1][c];
                        }
                    }else{
                        if(res[r-1][c] == -1 && res[r][c-1] == -1){
                            res[r][c] = -1;
                        }else if(res[r-1][c] == -1 && res[r][c-1] != -1){
                            res[r][c] = res[r][c-1];
                        }else if(res[r-1][c] != -1 && res[r][c-1] == -1){
                            res[r][c] = res[r-1][c];
                        }else{
                            res[r][c] = res[r-1][c] + res[r][c-1];
                        }
                    }
                }
            }// for ends
        }// for ends
        if(res[rows-1][cols-1] == -1){
            return 0;
        }else {
            return res[rows-1][cols-1];
        }

    }
}


// 64. Minimum Path Sum
/**
 * Dynamic programming
 * Time complexity: O(m*n); Space complexity: O(m*n)
 */
public class Solution {
    public int minPathSum(int[][] grid) {
        int row = grid.length, col = grid[0].length;
        if(grid == null || grid.length == 0 || row == 0 || col == 0)
            return 0;
        int[][] res = new int[row][col];
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                if(i == 0 && j == 0)
                    res[i][j] = grid[i][j];
                else if(i != 0 && j == 0)
                    res[i][j] = res[i-1][j] + grid[i][j];
                else if(i == 0 && j != 0)
                    res[i][j] = res[i][j-1] + grid[i][j];
                else
                    res[i][j] = Math.min(res[i-1][j] + grid[i][j], res[i][j-1] + grid[i][j]);
            }// for ends
        }// for ends

        return res[row-1][col-1];
    }
}


// 174. Dungeon Game
/**
 * Dynamic programming
 * Time complexity: O(m*n); Space complexity: O(m*n)
 */
public class Solution {
    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length, n = dungeon[0].length;// rows and cols
        if(dungeon.length == 0 || dungeon == null)
            return 0;

        int[][] ih = new int[m][n];
        for(int i = m - 1; i >= 0; i--){
            for(int j = n - 1; j >= 0; j--){
                if(i == m-1 && j == n-1)
                    ih[i][j] = Math.max(1, 1 - dungeon[i][j]);
                else if(i == m-1 && j != n-1)
                    ih[i][j] = Math.max(1, ih[i][j+1] - dungeon[i][j]);
                else if(i != m-1 && j == n-1)
                    ih[i][j] = Math.max(1, ih[i+1][j] - dungeon[i][j]);
                else
                    ih[i][j] = Math.max(1, Math.min(ih[i+1][j], ih[i][j+1]) - dungeon[i][j]);
            }// for ends
        }// for ends
        return ih[0][0];
    }
}


// 4. Median of Two Sorted Arrays
/**
 * method1: Merge sort time complexity: O(n+m)
 * method2: Comparing methods recursively
 */
public class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        double mOne = findMedian(nums1), mTwo = findMedian(nums2);
        if(mOne == mTwo){
            return mOne;
        }else if(mOne < mTwo){
            if(m == 1 && n == 1){
                return findMedianSortedArrays(Arrays.copyOfRange(nums1, m/2, m-1), Arrays.copyOfRange(nums2, 0, n/2));
            }
        }else{
                return findMedianSortedArrays(Arrays.copyOfRange(nums1, 0, m/2), Arrays.copyOfRange(nums2, n/2, n-1));
            }
        }

    }
    
    public double findMedianSortedArrays_m2(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        if(m < n)
            return findMedianSortedArrays_m2(nums2, nums1); // make sure nums2 is shorter
        if(m == 0)
            return ((double)nums1[(m-1)/2] + (double)nums1[m/2])/2;

        int lo = 0, hi = n * 2;

        while(lo <= hi){
            int mid2 = (lo + hi) / 2; // Try cut 2
            int mid1 = m + n -mid2; // Calculate Cut 1 accordingly

            // get L1, R1, L2, R2 respectively
            double l1 = (mid1 == 0) ? Integer.MIN_VALUE : nums1[(mid1 - 1) / 2];
            double l2 = (mid2 == 0) ? Integer.MIN_VALUE : nums2[(mid2-1)/2];
            double r1 = (mid1 == m*2) ? Integer.MAX_VALUE : nums1[(mid1)/2];
            double r2 = (mid2 == n*2) ? Integer.MAX_VALUE : nums2[(mid2)/2];

            if(l1 > r2)
                lo = mid2 + 1;
            else if(l2 > r1)
                hi = mid2 + 1;
            else
                return (Math.max(l1,l2) + Math.min(r1,r2)) / 2; // Otherwise, that's the right cut
        }

    }

    public double findMedian(int[] nums){
        double median;
        if(nums.length%2 == 0)
            median = (double)( nums[nums.length/2] + nums[nums.length/2-1] )/2;
        else
            median = (double)nums[nums.length/2]; // 5/2 = 2   
            
        return median;
    }
}


// 26. Remove Duplicates from Sorted Array
public class Solution {
    public int removeDuplicates(int[] nums) {

       int length = nums.length;
       if(length <= 1) return length;
       int len = 1;
       
       for(int i = 0, j = 0; i < nums.length - 1; i++){
          if(nums[i] != nums[i+1]){
            len += 1;
            nums[j+1] = nums[i+1];
            j++;
          }
       }
       return len; 

    }
}


// 27. Remove Element
public class Solution {
    public int removeElement(int[] nums, int val) {
        int len = 0;
        for(int i = 0; i < nums.length; i++){
            if(nums[i] != val){
                nums[len] = nums[i];
                len ++;
            }
        } // for ends
        return len;
    }
}

// 283. Move Zeroes
/**
 * Array, Two pointers
 * Space: O(1); Time: O(n)
 */
public class Solution {
    public void moveZeroes(int[] nums) {
        if(nums == null || nums.length == 0) return;
        int j = 0;
        for(int i = 0; i < nums.length; i++){
            if(nums[i] != 0)
                nums[j++] = nums[i];
        }// for ends

        while(j < nums.length){
            nums[j++] = 0;
        }
    }
}

// 287. Find the Duplicate Number
public class Solution {
    public int findDuplicate(int[] nums) {
        if(nums.length > 1){
            int slow = nums[0];
            int fast = nums[nums[0]];
            while (slow != fast){
                slow = nums[slow];
                fast = nums[nums[fast]];
            }

            fast = 0;
            while(fast != slow){
                fast = nums[fast];
                slow = nums[slow];
            }
            return slow;
        }// if ends
        return -1;
    }
}

// 268. Missing Number
/**
 * Time complexity: O(n); Space complexity: O(1)
 */
public class Solution {
    public int missingNumber_m1(int[] nums) {
        // method1: sum
        int len = nums.length;
        int sum = len*(len+1)/2;
        for(int i=0; i<len; i++)
            sum -=nums[i];
        return sum;
    }

    public int missingNumber_m2(int[] nums) {
        // method2: XOR
        int res = 0;
        for (int i = 0; i<nums.length; i++) {
            res ^= i;
            res ^= nums[i];
        }

        return res;
    }

    public int missingNumber_m3(int[] nums) {
        // method3: Binary search
    }
}

// 41. First Missing Positive
public class Solution {
    public int firstMissingPositive(int[] nums) {
        for(int i = 0; i < nums.length; i++){
            while(nums[i] > 0 && nums[i] <= n && nums[nums[i]-1] != nums[i])
                swap(nums[i], nums[nums[i]-1]);
        }

        for(int i = 0; i < nums.length; i++){
            if(nums[i] != i + 1)
                return i + 1;
        }

        return nums.length + 1;
    }

    public void swap(int a, int b){
        int temp = b;
        b = a;
        a = temp;
    }// swap ends

}


// 11. Container with Most Water
/**
 * Two pointers: left, right on each side of array
 * Time: O(n); Space: O(1)
 */
public class Solution {
    public int maxArea(int[] height) {
        if(height == null || height.length < 2)
            return 0;
        int lf = 0; // left pointer
        int rt = height.length - 1; // right pointer
        int area = Math.min(height[lf], height[rt]) * (rt - lf);

        while(lf < rt){
            if(height[lf] < height[rt]){
                lf++;
            }else if(height[lf] > height[rt]){
                rt--;
            }else{
                lf++; rt--;
            }
            int prev = area;
            int cur = Math.min(height[lf], height[rt]) * (rt - lf);
            area = Math.max(prev, cur);
        }// while ends
        return area;
    }
}