/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Code Samples : Your first Spark Program
*****************************************************************************/
package com.v2maestros.spark.bda.train;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class YourFirstSparkProgram {


	public static void main(String[] args) {
		
		//Setup configuration
		String appName = "firstApp";
		//String sparkMaster = "local[2]";
		String sparkMaster = "spark://192.168.99.1:7077";

		JavaSparkContext spContext = null;
	
		SparkConf conf = new SparkConf()
				.setAppName(appName)
				.setMaster(sparkMaster);
		
		//Create Spark Context from configuration
		spContext = new JavaSparkContext(conf); //actualy starts spark context
		
		//Read a file into an RDD
		JavaRDD<String> tweetsRDD = spContext.textFile("data/movietweets.csv");
		
		//Print first five lines
		for ( String s : tweetsRDD.take(5)) {
			System.out.println(s);
		}
		
		//Print count.
		System.out.println("Total tweets in file : " + tweetsRDD.count());
		
		
		//Convert to upper case
		JavaRDD<String> ucRDD = tweetsRDD.map( str -> str.toUpperCase());
		//Print upper case lines
		for ( String s : ucRDD.take(5)) {
			System.out.println(s);
		}
		
		
		//keep program running an look at spark website which gets activate when the program is running	 
		while(true) {
			try {
				Thread.sleep(10000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}


	}

}

/* Starting your own spark cluster on windows.
 * 
 * Step1 : First go to "SPARK_HOME\bin" Dir and run below commands

spark-class org.apache.spark.deploy.master.Master 
Use the IP and Port provided.

spark-class org.apache.spark.deploy.worker.Worker spark://IP:PORT


*/