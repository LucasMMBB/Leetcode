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
}