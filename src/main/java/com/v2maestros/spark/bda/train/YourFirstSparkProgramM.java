package com.v2maestros.spark.bda.train;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class YourFirstSparkProgramM {
	
	public static void main(String[] args) {
		
		//get spark context
		JavaSparkContext spContext = getSparkContext("ManojApp", "local[2]"); // local with 2 cores for 2 paartitions
		
		//Read a file to RDD
		JavaRDD<String> tweetRDD = spContext.textFile("data/movietweets.csv");
		
		//print lines from RDD
		for(String s: tweetRDD.take(10)) {
			System.out.println(s);
		}
		
		//print the count of records in RDD with count() action
		System.out.println(tweetRDD.count());
		
		
		//make data in upper case
		JavaRDD<String> upper = tweetRDD.map(s -> s.toUpperCase());
		//upper.saveAsTextFile("data/mytextTweet.txt");
		tweetRDD.foreach(System.out::println);
		for(String s: upper.take(5)) {
			System.out.println(s);
		}
		
	}
	
	
	//get spark context object
	public static JavaSparkContext getSparkContext(String appName, String sparkMaster) {
		 JavaSparkContext context = null;
		 //create spark configuration
		 SparkConf spConf = new SparkConf().setAppName(appName).setMaster(sparkMaster);
		 
		 //create context with configuration. Actually starts spark context.
		 context = new JavaSparkContext(spConf);
		 return context;
	}
	
	 

}
