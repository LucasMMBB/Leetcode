package util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class TwoSum {

    private ArrayList<Integer> values;

    /** Initialize data structure here. **/
    public TwoSum(){
        this.values = new ArrayList<>();
    }

    /** Add the number to an internal data structure **/
    public void add(int number){
        this.values.add(number);
    }

    /** Find if there exists any pair of numbers which sum is equal to the value **/
    public boolean find(int value){
        Set<Integer> set = new HashSet<>();
        for(int num : values){
            if(set.contains(value - num)){
                return true;
            } else {
                set.add(num);
            }
        }
        return false;
    }

    /** Getter and setter **/
    public ArrayList<Integer> getValues() {
        return values;
    }

    public void setValues(ArrayList<Integer> values) {
        this.values = values;
    }
}
