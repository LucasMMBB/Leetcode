public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int ones = 0;
        while(n != 0){
            ones = ones + (n & 1);
            n = n >>> 1; //  filled with zeros
        }
        return ones;
    }

    // method 2
    // time: const O(1), space: O(1)
    public int hammingWeight_m2(int n){
        int bits = 0;
        int mask = 1;
        for(int i = 0; i <  32; i++){
            if((n & mask) != 0){
                bits++;
            }
            mask <<= 1;
        }
        return bits;
    }

    // method 3
    public int hammingWeight_m3(int n){
        int sum = 0;
        while(n != 0){
            n &= (n - 1);
            sum++;
        }
        return sum;
    }
}