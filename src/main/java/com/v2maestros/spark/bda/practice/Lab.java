package com.v2maestros.spark.bda.practice;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.Row;

public class Lab {

	public static void main(String[] args) {
		
		System.out.println(getAgeRange(105));
	}
	
	
	public static int getAgeRange(int num) {
				int d = num % 10;
				int resAge = 0; 
				if(num % 10 < 5) {
					resAge = (num - d);
				}else {
					resAge = (num + (10 - d));
				}
				return resAge;
	}
	
	public static int getAverage(Row r, int startInd, int endInd) {
		
		List<Integer> arr = new ArrayList<>();
		
		for(int i = startInd; i < endInd ; i++) {
			arr.add(Math.abs(r.getInt(i))); 
		}
		
		return (int) arr.stream().mapToInt(e -> e).average().getAsDouble();
		
		
	}
		
	
}
