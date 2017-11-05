/*\
|*| Leetcode answers in Java
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


// 42. Trapping Rain Water
/**
 * Method1: Two pointers: 
 * Method2: Brute force: Time complexity: O(n^2)  Space complexity: O(1) extra space
 * Method3: Stack
 */
public class Solution {
    public int trap(int[] height) {
    	// find the wall position for this array,which is of the maximum of array
    	if(height.length == 0 || height ==null)
    		return 0;
    	int maxid = 0;
    	
    	for(int i = 1; i < height.length; i++){
    		if(height[i] > height[maxid])
    			maxid = i;
    	}// for ends

    	int water = 0, l = 0, r = height.length - 1;// left, right indices

    	while( l < r && height[l] < height[l + 1])
    		l++;
    	while( l < r && height[r] < height[r - 1])
    		r--;

    	for(int i = l + 1; i < maxid; i++){
    		if(height[i] < height[l])
    			water += height[l] - height[i];
    		else
    			l = i;
    	}// for ends

    	for (int i = r - 1; i > maxid ; i--) {
    		if(height[i] < height[r])
    			water += height[r] - height[i];
    		else
    			r = i;
    	}// for ends

    	return water;
    }

    // Method2: Brute force
    public int trap_m2(int[] height) {
        int ans = 0;
        int len = height.length;
        for(int i = 1; i < len - 1; i++){
            int max_left = 0, max_right = 0;
            for (int j = i; j >= 0; j--) {
                // search the left part
                max_left = Math.max(max_left, height[j]);
            }
            for (int j = i; j < len; j++) {
                max_right = Math.max(max_right, height[j]);
            }
            ans += Math.min(max_left, max_right) - height[i];
        }// for ends
        return ans;
    }
}


// 15. 3Sum
public class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();

        if(nums == null || nums.length < 3)
        	return res;
        
        // sort array
        Arrays.sort(nums);
        int size = nums.length;

        for(int i = 0; i < size - 2; i++){
        	int target = nums[i];
        	int left = i + 1; // left index
        	int right = size - 1; // right index

        	while(left < right){
        		int sum = nums[left] + nums[right];

        		if(sum == -1 * target){
        			List<Integer> l = new ArrayList<Integer>();
        			l.add(target);
        			l.add(nums[left]);
        			l.add(nums[right]);
        			res.add(l);
        			
        			int leftValue = nums[left];
        			int rightValue = nums[right];
        			left++;
        			right--;

        			// skip duplicates for both sides
	        		while(left < size && nums[left] == leftValue){
	        			left++;
	        		}
	        		while(right > i && nums[right] == rightValue){
	        			right--;
	        		}

        		}else if(sum < -1 * target){
        			left++;
        		}else{
        			right--;
        		}
        	}// while ends

        	// skip duplicates
        	while(i+1 < size && nums[i+1] ==  target){
        		i++;
        	}

        }// for ends
        return res;
    }
}


// 16. 3Sum closest
public class Solution {
    public int threeSumClosest(int[] nums, int target) {
 		
 		int size = nums.length;
 		if(nums == null || nums.length < 3)
        	return 0;

        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[2];

        for(int i = 0; i < size; i++){
        	int left = i + 1;
        	int right = size - 1;

        	while(left < right){
        		int sum = nums[i] + nums[left] + nums[right];

        		if(Math.abs(sum - target) < Math.abs(res - target))
        			res = sum;

        		if(sum == target){
        			return sum;
        		}else if(sum > target){
        			right--;
        		}else{
        			left++;
        		}// end if
        	}// while ends 
        }// for ends

        return res;
    }
}

// 259. 3Sum Smaller
public class Solution {
    public int threeSumSmaller(int[] nums, int target) {
        
        int size = nums.length;
        if(nums == null || nums.length < 3)
        	return 0;

        Arrays.sort(nums);
        int res = 0;
        for(int i = 0; i < size; i++){
        	int left = i + 1;
        	int right = size - 1;

        	while(left < right){
        		int sum = nums[i] + nums[left] + nums[right];

        		if(sum <  target){
        			res += right - left;
        			left++;
        		}else{
        			right--;
        		}

        	}//while ends
        }// for ends
        return res;
    }
}


// 18. 4Sum
/**
 * Time complexity: O(n^3)
 * Space complexity: O(n)
 */
public class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        if(nums == null || nums.length < 4){
        	return null;
        }
        List<List<Integer>> res =  new ArrayList<List<Integer>>();
        int size = nums.length;
        Arrays.sort(nums);

        for(int i = 0; i < size; i++){
        	int numI = nums[i];
        	for(int j = 1; j < size; j++){
        		List<Integer> innerList = new ArrayList<Integer>();

        		int left = j + 1; // left index
        		int right = size - 1; // right index

        		int numJ = nums[j];

        		while(left < right){
        			int sum = nums[i] + nums[j] + nums[left] + nums[right];

        			if(sum == target){
        				innerList.add(nums[i]);
        				innerList.add(nums[j]);
        				innerList.add(nums[left]);
        				innerList.add(nums[right]);
        				res.add(innerList);
        				left++;
        				right--;

        				// remove duplicates
	        			int leftValue = nums[left];
	        			int rightValue = nums[right];
	        			left++;
	        			right--;
		        		while(left < size && nums[left] == leftValue){
		        			left++;
		        		}
		        		while(right > j && nums[right] == rightValue){
		        			right--;
		        		}

        			}else if(sum > target){
        				right--;
        			}else{
        				left++;
        			}


        		}// while ends here

        		// remove duplicates in second for loop
        		while(j < size - 1 && nums[j + 1] == numJ){
        			j++;
        		}
        	}// for ends

        	//remove duplicates in first for loop
        	while(i + 1 < size && nums[i + 1] == numI){
        		i++;
        	}
        }// for ends
        return res;
    }
}


// 1. Two Sum
/**
 * Using hashmap to store nums as key and indices as values
 * Time complexity: O(n), space: O(n)
 */
public class Solution {
    public int[] twoSum(int[] nums, int target) {
    	Map<Integer, Integer> hashmap = new HashMap<>();
    	for(int i = 0; i < nums.length; i++){
    		int element = target - nums[i];
    		if(hashmap.containsKey(element)){
    			return new int[] {hashmap.get(element), i};
    		}
    		hashmap.put(nums[i], i);
    	}
    	throw new IllegalArgumentException("No two sum solution");
    }

    public int[] twoSum_m2(int[] nums, int target) {
    	Map<Integer, Integer> hashmap = new HashMap<>();
    	int[] res = new int[2];
    	// put elements into hashmap
    	for(int i = 0; i < nums.length; i++){
    		hashmap.put(nums[i], i);
    	}

    	for(int i = 0; i < nums.length; i++){
    		int keyToBeFind = target - nums[i];
    		if(hashmap.containsKey(keyToBeFind) && hashmap.get(keyToBeFind) != i){
    			res[0] = i;
    			res[1] = hashmap.containsKey(keyToBeFind);
    			return res;
    		}
    	}
    	return res;
    	//throw new IllegalArgumentException("No two sum solution");
    }
}

//167. Two Sum || - Input array is sorted
public class Solution {
    public int[] twoSum(int[] numbers, int target) {
        if(numbers == null || numbers.length < 2)
        	return null;
        int[] res = new int[2];
        int size = numbers.length;
        int left = 0, right = size - 1;
        Arrays.sort(numbers);
        while(left < right){
        	if((numbers[left] + numbers[right]) == target){
        		res[0] = left;
        		res[1] = right;
        		return res;
        	}else if((numbers[left] + numbers[right]) < target){
        		left++;
        	}else{
        		right--;
        	}
        }
        return res;
    }
}

// 280. Wiggle sort
/**
 * 1. if index is odd, nums[i] >= nums[i+1]
 * 2. if index is even, nums[i] <= nums[i-1]
 * Time complexity: O(n), space complexity: O(1)
 */
public class Solution {
    public void wiggleSort(int[] nums) {
        for(int i = 0; i < nums.length - 1; i++){
        	if ( (i%2)==0 && (nums[i] > nums[i+1]) ){// even
        		swap(nums, i, i+1);
        	}
        	if ( (i%2)==1 && (nums[i] < nums[i+1]) ){// odd
        		swap(nums, i, i+1);
        	}
        }// for ends
    }

    public void wiggleSort_m2(int[] nums){
    	// this is an obvious method.
    	/**
    	 * First sort it in ascending, then swap elements pair-wise starting from second element.
    	 * Time complexity: O(nlog(n)): sorting cost
    	 */
    	Arrays.sort(nums);
    	for (int i = 1; i < nums.length; i+=2) {
    		swap(nums, i, i + 1);
    	}
    }

    public int[] swap(int[] nums, int a, int b){
        int t = nums[a];
        nums[a] = nums[b];
        nums[b] = t;
        return nums;
    }
}


// 324. Wiggle Sort ||
/* Using python to finish this.
class Solution(object):
    def wiggleSort(self, nums):
        nums.sort()
        half = len(nums[::2]) - 1
        nums[::2], nums[1::2] = nums[half::-1], nums[:half:-1]
*/

// 215. Kth Largest Element in an Array
/**
 * Method1: Simplest method
 * Time complexity: O(NlogN), space complexity: O(1)
 * Method2: Other possibility is to use a min oriented priority queue that will
 * store the k-th largest values.
 * Method 3: quick select
 */
public class Solution {
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        int size = nums.length;
        return nums[size-k];
    }

    public int findKthLargest_m2(int[] nums, int k){
    	// method 2
    	final PriorityQueue<Integer> pq = new PriorityQueue<>();
    	for(int val : nums){
    		pq.offer(val);

    		if(pq.size() > k){
    			pq.poll();
    		}
    	}// for ends
    	return pq.peek();

    }

  public int findKthLargest_m3(int[] a, int k) {
    int n = a.length;
    int p = quickSelect(a, 0, n - 1, n - k + 1);
    return a[p];
  }
  
  // return the index of the kth smallest number
  public int quickSelect(int[] a, int lo, int hi, int k) {
    // use quick sort's idea
    // put nums that are <= pivot to the left
    // put nums that are  > pivot to the right
    int i = lo, j = hi, pivot = a[hi];
    while (i < j) {
      if (a[i++] > pivot) swap(a, --i, --j);
    }
    swap(a, i, hi);
    
    // count the nums that are <= pivot from lo
    int m = i - lo + 1;
    
    // pivot is the one!
    if (m == k)     return i;
    // pivot is too big, so it must be on the left
    else if (m > k) return quickSelect(a, lo, i - 1, k);
    // pivot is too small, so it must be on the right
    else            return quickSelect(a, i + 1, hi, k - m);
  }
  
  public void swap(int[] a, int i, int j) {
    int tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }

}


// 56. Merge Intervals
/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
public class Solution {
    public List<Interval> merge(List<Interval> intervals) {
        
    }
}



// 28. Implement strStr()
// Description: find needle in haystack and return first occurrence, or -1 if needle is not found
public class Solution {
    public int strStr(String haystack, String needle) {
        int sizeHay = haystack.length, sizeN = needle.length;
        for(int i = 0; i < sizeHay + 1; i++){
        	for(int j = 0; j < sizeN + 1; j++){

        		if(j == sizeN)
        			return i;

        		if(i + j == sizeHay)
        			return -1;

        		if(haystack.charAt(i + j) != needle(j))
        			break;
        	}//for ends
        }// for ends
    }

    public int strStr_m2(String haystack, String needle) {
        for(int i = 0; ; i++){
        	for(int j = 0; ; j++){

        		if(j == needle.length())
        			return i;

        		if(i + j == haystack.length())
        			return -1;

        		if(haystack.charAt(i + j) != needle.charAt(j))
        			break;
        	}//for ends
        }// for ends
    }
}


// 88. Merge Sorted Array
/**
 * Example: 12345, 357 merge => 1,2,3,3,4,5,5,7
 * Method: Two pointers from the end
 */
public class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
    	int i = m - 1; // index of nums1
    	int j = n - 1; // index of nums2
    	int k = m + n - 1; // index of merged array

    	while(i >= 0 && j >= 0){
    		if(nums1[i] > nums2[j]){
    			nums1[k] = nums1[i];
    			i--;
    		}else{
    			nums1[k] = nums2[j];
    			j--;
    		}
    		k--;
    	}// while ends

    	while(j >= 0){
    		nums1[k] = nums2[j];
    		j--;
            k--;
    	}
    }
}



// 459. Repeated Substring Pattern
public class Solution {
    public boolean repeatedSubstringPattern(String s) {
    	String ss = s.concat(s).substring(1, 2 * s.length() - 1);

    	if(ss.contains(s))
    		return true;
    	else
    		return false;
    }
}


// 125. Valid Palindrome
// Method: TWO POINTER
// Time complexity: O(n), Space complexity: O(1)
public class Solution {
    public boolean isPalindrome(String s) {

    	if (s.isEmpty()) {
        	return true;
        }
        int left = 0, right = s.length() - 1;
        char cHead, cTail;
        while(head <= tail) {
        	cHead = s.charAt(head);
        	cTail = s.charAt(tail);
        	if (!Character.isLetterOrDigit(cHead)) {
        		head++;
        	} else if(!Character.isLetterOrDigit(cTail)) {
        		tail--;
        	} else {
        		if (Character.toLowerCase(cHead) != Character.toLowerCase(cTail)) {
        			return false;
        		}
        		head++;
        		tail--;
        	}
        }
        
        return true;

    }
}


// 5. Longest Palindromic Substring
public class Solution {
    public String longestPalindrome(String s) {
        // method 1: Dynamic programming
        // Time and space complexity: O(n^2)
        if(s == null || s.length() == 0)
        	return "";

        int size = s.length(), max_len = 1, startIndex = 0;
        boolean[][] palindrome = new boolean[size][size];

        for(int i = 0; i < size; i++){
        	palindrome[i][i] = true;
        }

        // Finding palindromes of two characters
        for(int i = 0; i < size - 1; i++){
        	if(s.charAt(i) == s.charAt(i + 1)){
        		palindrome[i][i + 1] = true;
        		startIndex = i;
        		max_len = 2;
        	}
        }// for ends

        // Finding palindromes of more characters than two
        for(int cur_len = 3; cur_len <= size; cur_len++){
        	for(int i = 0; i < size - cur_len + 1; i++){
        		int j = i + cur_len - 1; // end index of substring
        		if(s.charAt(i) == s.charAt(j)
        			&& palindrome[i + 1][j - 1]){
        			palindrome[i][j] = true;
        			startIndex = i;
        			max_len = cur_len;
        		}
        	}// for ends
        }//for ends
        return s.substring(startIndex, max_len + startIndex);
    }

    public String longestPalindrome_m2(){
    	/**
    	 * Method 2: Manacher's algorithm
    	 * Time complexity: linear time
    	 */
    }
}

// 214. Shortest Palindrome
public class Solution {
    public String shortestPalindrome(String s) {

    }
}

//21. Merge Two Sorted Lists
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null)
        	return l2;
        if(l2 == null)
        	return l1;

        ListNode mergeHead;
        if(l1.val < l2.val){
        	mergeHead = l1;
        	mergeHead.next = mergeTwoLists(l1.next, l2);
        }else{
        	mergeHead = l2;
        	mergeHead.next = mergeTwoLists(l1, l2.next);
        }//if ends
        return mergeHead;
    }


    public ListNode mergeTwoLists_m2(ListNode l1, ListNode l2){
    	// while loop method
        if(l1 == null) {
            return l2;
        }
        if(l2 == null) {
            return l1;
        }
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if (l1 != null) {
            cur.next = l1;
        }
        if (l2 != null) {
            cur.next = l2;
        }
        return dummy.next;

    }
}


// 148. Sort List
public class Solution {
    public ListNode sortList(ListNode head) {
        
    }
}


// 66. PLUS ONE
public class Solution {
    public int[] plusOne(int[] digits) {

        int size = digits.length;
        int i = size - 1;

        while(i >= 0){
        	if(digits[i] == 9){
        		digits[i] = 0;
        	}else{
        		digits[i] = digits[i] + 1;
        		break;
        	}
            
            i--;
        }// while ends

        if(i == -1){
        	int[] res = new int[size + 1];
        	res[0] = 1;
        	for(int j = 1; j < size + 1; j++){
        		res[j] = 0;
        	}//for ends
        	return res;
        }else{
        	return digits;
        }// if ends
    }
}

// 415. Add Strings
public class Solution {
    public String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        for(int i = num1.length() - 1, j = num2.length() - 1; i >= 0 || j >= 0 || carry == 1; i--, j--){
            int x = i < 0 ? 0 : num1.charAt(i) - '0';
            int y = j < 0 ? 0 : num2.charAt(j) - '0';
            sb.append((x + y + carry) % 10);
            carry = (x + y + carry) / 10;
        }
        return sb.reverse().toString();
    }
}


// 86. Partition List
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class Solution {
    public ListNode partition(ListNode head, int x) {
        ListNode dummy1 = new ListNode(0), dummy2 = new ListNode(0);
        ListNode cur1 = dummy1, cur2 = dummy2;

        while(head != null){
        	if(head.val < x){
        		cur1.next = head;
        		cur1 = cur1.next;
        	}else{
        		cur2.next = head;
        		cur2 = cur2.next;
        	}// if ends
        	head = head.next;
        }// while ends

        cur2.next = null;
        cur1.next = dummy2.next;
        return dummy1.next;
    }
}

// 61. Rotate List
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || head.next == null || k == 0)
        	return head;

        int size = getSize(head);
        k = k % size;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode p = head;
        
        for(int i = 0; i < k; i++){
        	p = p.next;
        }// for ends

        for(int j = 0; j < size - k - 1; j++){
        	p = p.next;
        	head = head.next;
        }// for ends

        p.next = dummy.next;
        dummy.next = head.next;
        head.next = null;

        return dummy.next;
    }

    public int getSize(ListNode head){
    	int count = 0;
    	while(head != null){
    		head = head.next;
    		count++;
    	}// while ends
    	return count;
    }
}


// 189. ROTATE ARRAY
// Const space complexity. Time complexity: O(n)
public class Solution {
    public void rotate(int[] nums, int k) {
  		if(nums == null || nums.length == 0 || k == 0)
  			return;
  		int size = nums.length;
  		k %= size;

  		reverseArray(nums, size - k, size - 1);
  		reverseArray(nums, 0, size - k - 1);
  		reverseArray(nums, 0, size - 1);
    }

    public void reverseArray(int[] nums, int start, int end){
    	if(nums == null || nums.length == 0 || nums.length == 1)
    		return;

    	while(start < end){
    		int temp = nums[end];
    		nums[end] = nums[start];
    		nums[start] = temp;
            start++;
            end--;
    	}// while ends
    }
}

// 9. Palindrome Number
public class Solution {
    public boolean isPalindrome(int x) {
        if(x < 0 || (x != 0 && x%10 == 0))
        	return false;

        int rev = 0;
        while(x > rev){
        	rev = rev*10 + x%10;
        	x = x/10;
        }// while ends

        return (x == rev || x == rev/10);

    }
}
// 206. Reverse Linked List
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode cur = head, prev = null, next = null;

        while(cur != null){
        	next = cur.next;
        	cur.next = prev;
        	prev = cur;
        	cur = next;
        }// while ends
        return prev;

    }
}
// 234. Palindrome Linked List
/**
 * test case: 1 2 3 4 5
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class Solution {
    public boolean isPalindrome(ListNode head) {
        // get median
        if(head == null || head.next == null) return true;
        ListNode mid = getMedian(head);
        ListNode rightpart = mid.next;
        mid.next = null;
        return compareTwoList(head, reverseList(rightpart));
    }

    public ListNode getMedian(ListNode head){
    	ListNode slow = head;
    	ListNode fast = head.next;
    	while(fast != null && fast.next != null){
    		slow = slow.next;
    		fast = fast.next.next;
    	}// while ends
    	return slow;
    }
    public ListNode reverseList(ListNode head) {
        ListNode cur = head, prev = null, next = null;

        while(cur != null){
        	next = cur.next;
        	cur.next = prev;
        	prev = cur;
        	cur = next;
        }// while ends
        return prev;
        
    }
    public boolean compareTwoList(ListNode l1, ListNode l2){

    	while(l1 != null && l2 != null){
    		if(l1.val != l2.val)
    			return false;
    		l1 = l1.next;
    		l2 = l2.next;
    	}// while ends
    	return true;
    }
}

// 19. Remove Nth Node From End of List
public class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        
    }
}


// 237. Delete Node in a Linked List
/**
 * Time and Space: const
 * Method: copy next node value to current and point current to next.next node 
 */
public class Solution {
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}

// 203. Remove Linked List Elements
public class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(0), pre = new ListNode(0);
        ListNode cur = head;
        dummy = pre;
        pre.next = cur;

        if(cur == null)
            return null;

        while(cur != null){
            if(cur.val == val){
                pre.next = cur.next;
            }else{
                pre = cur;
            }// if ends
            cur = cur.next;
        }// while ends
        return dummy.next;
    }
}


// 2. Add two numbers
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0), cur = dummy;
        int carry = 0;
        while(l1 != null || l2 != null){
            if(l1 != null)
                int a = l1.val;
            else
                int a = 0;

            if(l2 != null)
                int b = l2.val;
            else
                int b = 0;

            int sum = a + b + carry;
            cur.next = new ListNode(sum % 10);
            cur = cur.next;
            if(l1 != null)
                l1 = l1.next;
            if(l2 != null)
                l2 = l2.next;

        }// while ends
        if(carry > 0)
            cur.next = new ListNode(1);
        return dummy.next;
    }
}

// 445. Add Two Numbers ||
public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Integer> s1 = new Stack<Integer>();
        Stack<Integer> s2 = new Stack<Integer>();

        while(l1 != null){
            s1.push(l1.val);
            l1 = l1.next;
        }// while ends

        while(l2 != null){
            s2.push(l2.val);
            l2 = l2.next;
        }// while ends

        int sum = 0, carry = 0;
        ListNode cur = null;
        while(!s1.empty() || !s2.empty()){
            if(!s1.empty()) sum += s1.pop();
            if(!s2.empty()) sum += s2.pop();
            sum += carry;
            ListNode temp = new ListNode(sum % 10);
            temp.next = cur;
            cur = temp;
            carry = sum / 10;
            sum = 0;
        }// while ends
        if(carry > 0){
            ListNode temp = new ListNode(carry);
            temp.next = cur;
            cur = temp;
        }
        return cur;
    }
}

// 147. Insertion sort list
public class Solution {
    public ListNode insertionSortList(ListNode head){

    }
}
// 148. Sort List
public class Solution {
    public ListNode sortList(ListNode head) {
      if(head == null || head.next == null) return head;

      // 1. cur the list to two halves
      ListNode prev = null, slow = head, fast = head;
      while(fast != null && fast.next != null){
        prev = slow;
        slow = slow.next;
        fast = fast.next.next;
      }// while ends
      prev.next = null;

      // 2. sort each half
      ListNode l1 = sortList(head);
      ListNode l2 = sortList(slow);

      // 3. merge l1 and l2
      return merge(l1, l2);
    }

    public ListNode merge(ListNode l1, ListNode l2){
        ListNode res = new ListNode(0), cur = res;
        while(l1 != null && l2 != null){
            if(l1.val <= l2.val){
                cur.next = new ListNode(l1.val);
                l1 = l1.next;
            }else{
                cur.next = new ListNode(l2.val);
                l2 = l2.next;
            }// if ends
            cur = cur.next;
        }// while ends
        if(l1 != null)
            cur.next = l1;
        if(l2 != null)
            cur.next = l2;
        return res.next;
    }
}

// 141. Linked List Cycle
public class Solution {
    public boolean hasCycle(ListNode head) {
       ListNode slow = fast = head;

       while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(fast == slow) return true;
       }// while ends 

       return false;
    }

    public boolean hasCycle_m2(ListNode head){
        // method 2: hashset
    }
}

// 142. Linked List Cycle ||
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;

        while(fast != None && fast.next != None){
            slow = slow.next;
            fast = fast.next.next;

            if( slow == fast){
                slow = head;
                while(slow != fast){
                    slow = slow.next;
                    fast = fast.next;
                }// while ends
                return slow;
            }
        }//while ends
        return null;
    }
}


// 274. H-Index
/**
 * Method One: Sort first, scan from end to front
 * Method Two: Binary Search
 */
public class Solution {
    public int hIndex(int[] citations) {
        int len = citations.length;
        Arrays.sort(citations);
        int tag = 0;
        for (int h = len; h > 0; h--) {
            for(int j = 1; j <= h; j++){
                if(citations[len - j] < h){
                    tag = 2;
                    break;
                }
                tag = 1;
            }// for ends
            if(tag == 1){
                if(len - h - 1 >= 0 && citations[len - h - 1] <= h)
                    return h;

                if(len - h -1 < 0)
                    return h;
            }
        }// for ends
        return 0;
    }

    public int hIndex_m2(int[] citations){
        // method two
        int n = citations.length;
        int[] buckets = new int[n + 1];
        for (int c : citations) {
            if(c >= n){
                buckets[n]++;
            }else{
                buckets[c]++;
            }
        }// for ends

        int count = 0;
        for (int i = n; i >= 0; i--) {
            count += buckets[i];
            if(count >= i)
                return i;
        }// for ends
        return 0;
    }
}

// 275. H-Index ||
public class Solution {
    public int hIndex(int[] citations) {
        int count = 0, len = citations.length;
        for (int h = 1; h < citations.length + 1; h++) {
            if(citations[len - h] >= h){
                count++;
            }else{
                break;
            }
        }
        return count;   
    }

    public int hIndex_m2(int[] citations){
        // method two: Binary Search
        // time: O(log(n))
        int n = citations.length;
        int l = 0, r = n - 1;

        while(l <= r){
            mid = l + (r - l) / 2;

            if(citations[mid] == n - mid){
                return n - mid;
            }else if(citations[mid] < n - mid){
                l = mid + 1;
            }else{
                r = mid - 1;
            }// if ends
        }// while ends
        return n - l;
    }
}

// 230. Kth Smallest Element in BST
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
    public int kthSmallest(TreeNode root, int k) {
        int size = getSize(root.left);

        if(size + 1 == k){
            return root.val;
        }else if(size + 1 < k){
            return kthSmallest(root.right, k - size - 1);
        }else{
            return kthSmallest(root.left, k);
        }// if ends
    }

    public int getSize(TreeNode root){
        if(root == null)
            return 0;

        return getSize(root.left) + getSize(root.right) + 1;
    }
}

// 162. Find Peak Element
public class Solution {
    public int findPeakElement(int[] nums) {
        int len = nums.length;

        for(int i = 0; i < len - 1; i++){
            if(nums[i] > nums[i + 1])
                return i;
        }// for ends
        return len - 1;
    }
}

// 448. Find All Numbers Disappeared in an Array
// O(n)
public class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
       List<Integer> res = new ArrayList<Integer>();
       for (int i = 0; i < nums.length; i++) {
           int index = Math.abs(nums[i]) - 1;
           if(nums[index] > 0) nums[index] = -nums[index];
       }
       for (int i = 0; i < nums.length; i++) {
           if(nums[i] > 0){
            res.add(i + 1);
           }
       }
       return res;
    }
}

// 442. Finad All duplicate numbers in an array
public class Solution {
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<Integer>();
        for (int i = 0; i < nums.length; i++) {
            int index = Math.abs(nums[i]) - 1;
            if(nums[index] > 0)
                nums[index] = -nums[index];
            else
                res.add(index + 1);
        }
        return res;
    }
}
// 628. Maximum Product of Three Numbers
public class Solution {
    public int maximumProduct(int[] nums) {
        Arrays.sort(nums);
        int a1 = nums[nums.length - 1] * nums[nums.length - 2] * nums[nums.length - 3];
        int a2 = nums[0] * nums[1] * nums[nums.length - 1];
        return Math.max(a1, a2);
    }

    public int maximumProduct_m2(int[] nums){
        int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE;
        for(int n: nums){
            // min1 < min2
            if(n <= min1){
                min2 = min1;
                min1 = n;
            } else if (n <= min2){
                min2 = n;
            }

            // max1 > max2 > max3
            if(n > max1){
                max3 = max2;
                max2 = max1;
                max1 = n;
            }else if( n >= max2){
                max3 = max2;
                max2 = n;
            }else if(n >= max3){
                max3 = n;
            }
        }// for ends
        return Math.max(min1*min2*max1, max3*max2*max1);
    }
}


// 35. Search Insert Position
public class Solution {
    public int searchInsert(int[] nums, int target) {
        int l = 0, r = nums.length - 1;
        while(l <= r){
            int mid = l + (r - l) / 2;
            if(nums[mid] ==  target) return mid;
            else if(nums[mid] > target) r = mid - 1;
            else l = mid + 1;
        }
        return l;
    }
}


// 217. Contains Duplicate
public class Solution {
    public boolean containsDuplicate(int[] nums) {
        HashSet set = new HashSet<Integer>();
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }
        return set.size() != nums.length;
    }

    public boolean containsDuplicate_m2(int[] nums){
        HashSet<Integer> set = new HashSet<Integer>();
        for(int num: nums){
            if(set.contains(num)){
                return true;
            }
            set.add(num);
        }
        return false;
    }

    public boolean containsDuplicate_m3(int[] nums){
        // method 3: sort
        // Time: O(nlogn), Space: O(1)
        Arrays.sort(nums);
        for(int i = 0; i < nums.length - 1; i ++){
            if(nums[i] == nums[i + 1])
                return true;
        }// for ends
        return false;
    }
}


// 219. Contains Duplicate ||
public class Solution {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        // hashset
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            if(map.containsKey(nums[i])){
                int sub = Math.abs(i - map.get(nums[i]));
                if(sub <= k)
                    return true;
            }else{
                map.put(nums[i], i);
            }// if ends
        }// for ends
        return false;
    }
}

// 111. Minimum Depth of Binary Tree
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
    public int minDepth(TreeNode root) {
        if(root == null) return 0;
        int depth = 1;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while(!q.isEmpty()){
            int size = q.size();
            // for each level
            for(int i=0;i<size;i++){
                TreeNode node = q.poll();
                if(node.left == null && node.right == null){
                    return depth;
                }
                if(node.left != null){
                    q.offer(node.left);
                }
                if(node.right != null){
                    q.offer(node.right);
                }
            }
            depth++;
        }
        return depth;
    }

    public int minDepth(TreeNode root){
        if(root == null)
            return 0;

        if(root.left != null && root.right != null){
            return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
        }else if(root.left != null && root.right == null){
            return minDepth(root.left) + 1;
        }else if(root.left == null && root.right != null){
            return minDepth(root.right) + 1;
        }else{
            return 1;
        }// if ends
    }
}


// 102. Binary Tree Level Order Traversal
public class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if(root == null) return res;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while(!q.isEmpty()){
            int size = q.size();
            List<Integer> lres = new ArrayList<Integer>();
            for(int i = 0; i < size; i++){
                TreeNode node = q.poll();
                lres.add(node.val);
                if(node.left != null){
                    q.offer(node.left);
                }
                if(node.right != null){
                    q.offer(node.right);
                }
            }
            res.add(lres);
        }// while ends
        return res; 
    }
}


// 637.Average of Levels in Binary Tree
public class Solution {
    public List<Double> averageOfLevels(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if(root == null) return res;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while(!q.isEmpty()){
            int size = q.size();
            List<Integer> lres = new ArrayList<Integer>();
            for(int i = 0; i < size; i++){
                TreeNode node = q.poll();
                lres.add(node.val);
                if(node.left != null){
                    q.offer(node.left);
                }
                if(node.right != null){
                    q.offer(node.right);
                }
            }
            res.add(lres);
        }// while ends
        return res;      
    }
} 


// 107. Binary Tree Level Order Traversal ||
public class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if(root == null) return res;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        while(!queue.isEmpty()){
            int len = queue.size();
            List<Integer> lres = new ArrayList<Integer>();
            for(int i = 0; i < len; i++){
                TreeNode node = queue.poll();
                lres.add(node.val);
                if(node.left != null){
                    queue.offer(node.left);
                }
                if(node.right != null){
                    queue.offer(node.right);
                }
            }// for ends
            res.add(0, lres);
        }// while ends
        return res;

    }
}


// 136. Single Number
public class Solution {
    public int singleNumber(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for (int num: nums) {
             if(!set.contains(num)){
                set.add(num);
             }else{
                set.remove(num);
             }// if ends
         }// for ends
         Iterator<Integer> itr = set.iterator();
         return itr.next();
    }

    public int singleNumber_m2(int[] nums){
        // bit manipulation: xor
        int res = 0;
        for(int num: nums)
            res ^= num;
        return res;
    }
}

// 100. Same Tree
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
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null){
            return true;
        }else if(p != null && q != null && p.val == q.val){
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }else{
            return false;
        }// if ends
    }

    public boolean isSameTree(TreeNode p, TreeNode q){
        if(p == null && q == null){
            return true;
        }
        List<List<TreeNode>> stack = new LinkedList<List<TreeNode>>();
        List<TreeNode> list = new ArrayList<TreeNode>();
        list.add(p);
        list.add(q);
        stack.add(list);
        while(!stack.isEmpty()){
            List<TreeNode> tem = stack.remove(0);
            TreeNode l = tem.remove(0); ///get first elment
            TreeNode r = tem.remove(0);
            if(l == null && r == null){
                continue;
            }else if(l != null && r != null && l.val == r.val){
                List<TreeNode> llist = new ArrayList<TreeNode>();
                llist.add(l.left);
                llist.add(r.left);
                stack.add(llist);
                List<TreeNode> rlist = new ArrayList<TreeNode>();
                rlist.add(l.right);
                rlist.add(r.right);
                stack.add(rlist);
            }else{
                return false;
            }
        }//  while ends
        return true;  
    }
}


// 110. Balanced Binary Tree
public class Solution {
    public boolean isBalanced(TreeNode root) {
        return height(root) != -1;
    }

    public int height(TreeNode root){
        if(root == null)
            return 0;
        // left and right height
        int l = height(root.left);
        int r = height(root.right);

        if(l == -1 || r == -1 || Math.abs(l - r) > 1)
            return -1; // -1 means that tree is not balanced

        return Math.max(l, r) + 1;
    }
}


// 104. Maximum Depth of Binary Tree
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
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;

        if(root.left != null || root.right != null)
            return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
        else
            return 1;

    }
}


// 108. Convert Sorted Array to Binary Search Tree
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
    public TreeNode sortedArrayToBST(int[] nums) {
        if(nums.length == 0 || nums == null) return null;
        TreeNode tree = helper(nums, 0, nums.length - 1);
        return tree;
    }

    public TreeNode helper(int[] nums, int l, int r){
        if(l >  r) return null;

        int mid = l + (r - l) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = helper(nums, l, mid - 1);
        node.right = helper(nums, mid + 1, r);
        return node;
    }
}

// 109. Convert Sorted List to Binary Search Tree
public class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        if(head == null) return null;
        return toBST(head, null);
    }

    public TreeNode toBST(ListNode head, ListNode tail){
        if(head == tail) return null;
        ListNode slow = head;
        ListNode fast = head.next;
        while(fast != tail && fast.next != tail){
            slow = slow.next;
            fast = fast.next.next;
        }
        TreeNode node = new TreeNode(slow.val);
        node.left = toBST(head, slow);
        node.right = toBST(slow.next, tail);
        return node;
    }
}


// 112. Path Sum
public class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) return false;

        if(root.left == null && root.right == null && sum - root.val == 0){
            return true;
        }else if(sum - root.val < 0){
            return false;
        }
        
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }
}


// 113. Path Sum ||
public class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        dfs(root, sum, res, path);
        return res;
    }

    public void dfs(TreeNode root, int sum, List<List<Integer>> res, List<Integer> path){
        if(root == null) return;
        path.add(root.val);

        if(root.left == null && root.right == null){
            if(root.val == sum)
                res.add(new ArrayList<Integer>(path));
            return
        }

        if(root.left != null){
            dfs(root.left, sum - root.val, res, path);
            path.remove(path.size() - 1);
        }

        if(root.right != null){
            dfs(root.right, sum - root.val, res, path);
            path.remove(path.size() - 1);
        }
    }
}


//170. Two Sum ||| - Data structure design
public class TwoSum {

    // Map interface
    Map<Integer, Integer> map = new HashMap<Integer, Integer>();
    /** Initialize your data structure here. */
    public TwoSum() {
        
    }
    
    /** Add the number to an internal data structure.. */
    public void add(int number) {
        if(map.containsKey(number)){
            map.put(number, map.get(number) + 1);
        }else{
            map.put(number, 1);
        }
    }
    
    /** Find if there exists any pair of numbers which sum is equal to the value. */
    public boolean find(int value) {
        for(Map.Entry<Integer, Integer> entry : map.entrySet()){
            int k = entry.getKey();
            int v = entry.getValue();
            if(k*2 == value){
                if(v >= 2)return true;
            }else{
                if(map.containsKey(value - k)) return true;
            }
        }// for ends
        return false;
    }
}

// 105. Construct Binary Tree from Preorder and Inorder
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
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return helper(0, 0, inorder.length - 1, preorder, inorder);
    }

    public TreeNode helper(int preStart, int inStart, int inEnd, in[] preorder, in[] inorder){
        if(preStart > preorder.length - 1 || inStart > inEnd)
            return null;

        TreeNode root = new TreeNode(preorder[preStart]);
        int inIndex = 0;
        for(int i = inStart; i <= inEnd; i++){
            if(inorder[i] == root.val)
                inIndex = i;
        }// for ends
        root.left = helper(preStart + 1, inStart, inIndex + 1, preorder, inorder);
        root.right = helper(preStart + inIndex - inStart + 1, inIndex + 1, inEnd, preorder, inorder);
        return root;
    }
}

// 545. Boundary of Binary Tree
public class Solution {
    public List<Integer> boundaryOfBinaryTree(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        if(root == null)return res;
        if(!isLeaf(root)) res.add(root.val);

        TreeNode left = root.left;
        while(left != null){
            if(!isLeaf(left))res.add(left.val);
            if(left.left != null)
                left = left.left;
            else{
                left = left.right;
            }
        }// while ends

        addLeaves(res, root);

        TreeNode right = root.right;
        Stack<Integer> st = new Stack<Integer>();
        while(right != null){
            if(!isLeaf(right)) st.push(right.val);
            if(right.right != null){
                right = right.right;
            }else{
                right = right.left;
            }
        }// while ends
        while(!st.empty()){
            res.add(st.pop());
        }
        return res;
    }

    public boolean isLeaf(TreeNode root){
        if(root == null) return false;
        return root.left == null && root.right == null;
    }

    public void addLeaves(List<Integer> res, TreeNode root){
        if(isLeaf(root)){
            res.add(root.val);
        }else{
            if(root.left != null)
                addLeaves(res, root.left);
            if(root.right != null)
                addLeaves(res, root.right);
        }
    }
}

// 202. Happy Number
public class Solution {
    public boolean isHappy(int n) {
        Set<Integer> set = new HashSet<Integer>();
        set.add(1);
        while(!set.contains(n)){
            set.add(n);
            int sum = 0;
            while(n > 0){
                sum += Math.pow(n % 10, 2);
                n = n / 10;
            }// while ends
            n = sum;
        }// while ends
        return n == 1;
    }
}


// 258. Add Digits
public class Solution {
    public int addDigits(int num) {
        return num == 0 ? 0 : (num % 9 == 0 ? 9 : (num % 9)); 
    }
}

// 242.Valid Angaram
public class Solution {
    // method 1: sort
    // time: O(nlogn)
    public boolean isAnagram(String s, String t) {
        if(s == null || t == null || s.length() != t.length())
            return false;

        return sortString(s).equals( sortString(t) );
    }

    public String sortString(String s){
        char[] sChar = s.toCharArray();
        Arrays.sort(sChar);
        return new String(sChar);
    }

    public boolean isAnagram_2(String s, String t){
        if (s.length() != t.length()) {
            return false;
        }
        int[] counter = new int[26];
        for (int i = 0; i < s.length(); i++) {
            counter[s.charAt(i) - 'a']++;
            counter[t.charAt(i) - 'a']--;
        }
        for (int count : counter) {
            if (count != 0) {
                return false;
            }
        }
        return true;
    }
}


// 204. Count Primes
public class Solution {
    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n];
        int count = 0;
        for(int i = 2; i < n; i++){
            if(notPrime[i] == false){
                count++;
                for(int j = 2; j*i < n; j++){
                    notPrime[j*i] = true;
                }// for ends
            }// if ends
        }// for ends
        return count;  
    }
}

// 349. Intersection of Two Arrays
public class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet<Integer>();
        Set<Integer> intersect = new HashSet<Integer>();
        for(int i = 0; i < nums1.length; i++){
            set.add(nums1[i]);
        }// for ends
        for(int i = 0; i < nums2.length; i++){
            if(set.contains(nums2[i]))
                intersect.add(nums2[i]);
        }// for ends
        int[] res = new int[intersect.size()];
        int i = 0;
        for(Integer num : intersect)
            res[i++] = num;
        return res;    
    }
}

// 350. Intersection of Two Arrays
public class Solution{
    public int[] intersect(int[] nums1, int[] nums2){
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        List<Integer> res = new ArrayList<Integer>();
        for(int num : nums1){
            if(map.containsKey(num)){
                map.put(num, map.get(num) + 1);
            }else{
                map.put(num, 1);
            }// if ends
        }// for ends
        for(int num : nums2){
            if(map.containsKey(num) && map.get(num) > 0){
                res.add(num);
                map.put(num, map.get(num) - 1);
            }// if ends
        }// for ends

        int[] result = new int[res.size()];
        for(int i = 0; i < res.size(); i++){
            result[i] = res.get(i);
        }
        return result;
    }
}
// 94. Binary Tree Inorder Traversal
public class Solution{
    public List<Integer> inorderTraversal(TreeNode root){
        List<Integer> res = new ArrayList<Integer>();
        Deque<TreeNode> stack = new ArrayDeque<TreeNode>();
        TreeNode p = root;
        while(!stack.isEmpty() || p != null){
            if(p != null){
                stack.push(p);
                p = p.left;
            }else{
                TreeNode node = stack.pop();
                res.add(node.val);
                p = node.right;
            }// if ends
        }// while ends
        return res;
    }
}

// 155. Min Stack
public class MinStack {
    int min = Integer.MAX_VALUE;
    Stack<Integer> stack = new Stack<Integer>();
    /** initialize your data structure here. */
    public MinStack() {
    }
    
    public void push(int x) {
        if(x <= min){
            stack.push(min);
            min = x;
        }
        stack.push(x);
    }
    
    public void pop() {
        if(stack.pop() == min){
            min = stack.pop();
        }
    }
    
    public int top() {
        stack.peek();
    }
    
    public int getMin() {
        return min
    }
}
// 144. Binary Tree Preorder Traversal
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
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        if(root == null) return res;
        Stack<TreeNode> st = new Stack<TreeNode>();
        st.add(root);
        while(!st.empty()){
            TreeNode node = st.pop();
            if(node != null){
                res.add(node.val);
                st.push(node.right);
                st.push(node.left);
            }// if ends
        }// while ends
        return res;
    }
}


// 39. Combination Sum
public class Solution {
    public List<List<Integer>> combinationSum(int[] nums, int target) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(list, new ArrayList<>(), nums, target, 0);
        return list;
    }

    private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] nums, int remain, int start){
        if (remain < 0) return;
        else if(remain == 0) list.add(new ArrayList<>(tempList));
        else{
            for (int i = start; i < nums.length; i++){
                tempList.add(nums[i]);
                // not i + 1 since we can reuse same elements
                backtrack(list, tempList, nums, remain-nums[i], i);
                tempList.remove(tempList.size()-1);
            }
        }
    }
}

// 401. Binary Watch
public class Solution {
    public List<String> readBinaryWatch(int num) {
        List<String> times = new ArrayList<>();
        for (int h = 0; h < 12; h++){
            for (int m = 0; m < 60; m++){
                if (Integer.bitCount(h * 64 + m) == num)
                    times.add(String.format("%d:%02d", h, m));
            }
        }
        return times;
    }
}

// 79. Word Search
class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null) return false;
        boolean[][] used = new boolean[board.length][board[0].length];
        for (int row = 0; row < board.length; row++){
            for (int col = 0; col < board[0].length; col++){
                if(existHelper(board, used, word.toCharArray(), 0, col, row)){
                    return true;
                }
            }
        }
        return false;       
    }

    public boolean existHelper(char[][] board, boolean[][] used, char[] word, int idx, int col, int row){
        if (idx == word.length) return true;
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length){
            return false;
        }
        if (used[row][col] ==  true || board[row][col] != word[idx]) return false;
        used[row][col] = true;

        boolean exist = existHelper(board, used, word, idx+1, col + 1, row);
        if(exist) return true;

        exist = existHelper(board, used, word, idx+1, col - 1, row);
        if(exist) return true;

        exist = existHelper(board, used, word, idx+1, col, row+1);
        if(exist) return true;

        exist = existHelper(board, used, word, idx+1, col, row - 1);
        if(exist) return true;

        used[row][col] = false;
        return false;

    }
}

// 292. Nim Game
class Solution {
    public boolean canWinNim(int n) {
        return (n % 4) != 0;
    }
}

// 293. Flip Game
class Solution {
    public List<String> generatePossibleNextMoves(String s) {
        List<String> list = new ArrayList<String>();
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == '+' && s.charAt(i - 1) == '+') {
                list.add(s.substring(0, i - 1) + "--" + s.substring(i + 1, s.length()));
            }
        }
        return list;
    }
} 


// 169. Majority Element
class Solution {
    public int majorityElement(int[] nums) {
        // Hashtable
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int res = 0;

        for(int num: nums){
            if(!map.containsKey(num)){
                map.put(num, 1);
            }else{
                map.put(num, map.get(num)+1);
            }

            if(map.get(num) > nums.length / 2){
                res = num;
                break;
            }

        }

        return res;
    }
}


// 229. Majority Element ||
class Solution {
    public List<Integer> majorityElement(int[] nums) {
        int cnt1 = 0, cnt2 = 0;
        int m1 = 0, m2 = 1;

        List<Integer> res = new ArrayList<Integer>();
        int len = nums.length;

        for(int i = 0; i < len; i++){
            if(nums[i] == m1){
                cnt1++;
            }else if(nums[i] == m2){
                cnt2++;
            }else if(cnt1 == 0){
                m1 = nums[i];
                cnt1 = 1;
            }else if(cnt2 == 0){
                m2 = nums[i];
                cnt2 = 1;
            }else{
                cnt1--;
                cnt2--;
            }
        }

        cnt1 = 0;
        cnt2 = 0;
        for(int i = 0; i < nums.length; i++){
            if(nums[i] == m1){
                cnt1++;
            }
            if(nums[i] == m2){
                cnt2++;
            }
        }

        if(cnt1 > len / 3)
            res.add(m1);
        if(cnt2 > len / 3)
            res.add(m2);

        return res;
    }
}


// 118. Pascal's Triangle
class Solution {
    public List<List<Integer>> generate(int numRows) {
        // caculate element value:
        // k(i)(j) = k(i-1)(j-1) + k(i-1)(j) except for the first and last element
        List<List<Integer>> triangle = new ArrayList<List<Integer>>();
        if(numRows <= 0)
            return triangle;
        int tmp;
        for(int i = 0; i < numRows; i++){
            List<Integer> row = new ArrayList<Integer>();
            for(int j = 0; j < i + 1; j++){
                if(j == 0 || j == i){
                    row.add(1);
                }else{
                    tmp = triangle.get(i-1).get(j-1) + triangle.get(i-1).get(j);
                    row.add(tmp);
                }
            }
            triangle.add(row);
        }
        return triangle;  

    }

}


// 55. Jump Game
class Solution {
    public boolean canJump(int[] nums) {
        int max = 0;
        for(int i = 0; i < nums.length; i++){
            if(i > max) return false;
            max = Math.max(nums[i] + i, max);
        }
        return true;
    }
}

// 226. Invert Binary Tree
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;

        TreeNode right = invertTree(root.right);
        TreeNode left = invertTree(root.left);

        root.left = right;
        root.right = left;

        return root;
    }

    public TreeNode invertTree(TreeNode root){
        if(root==null) return null;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;

            if(node.left != null) queue.add(node.left);
            if(node.right != null) queue.add(node.right);

        return root;
        }
    }
}




// ------- To do List ----------

// 103. Binary Tree Zigzag Level Order Traversal
public class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        
    }
}

// 199. Binary Tree Right Side View
public class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        
    }
}
// 220. Contains Duplicate |||
public class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        
    }
}

// 222. Count Complete Tree Nodes
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
    public int countNodes(TreeNode root) {
        
    }
}

// 92. Reverse Linked List ||
public class Solution {
    public ListNode reverseBetween(ListNode head, int m, int n) {
        
    }
}
