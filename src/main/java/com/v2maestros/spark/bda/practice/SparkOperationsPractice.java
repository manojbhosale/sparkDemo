
/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Practice Exercises : Spark Operations
*****************************************************************************/

package com.v2maestros.spark.bda.practice;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.LongAccumulator;

import com.v2maestros.spark.bda.common.ExerciseUtils;
import com.v2maestros.spark.bda.common.SparkConnection;

import scala.Tuple2;

public class SparkOperationsPractice {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		/*----------------------------------------------------------------------
		 * 		Common Setup
		 ---------------------------------------------------------------------*/
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		JavaSparkContext spContext = SparkConnection.getContext();
		
		/*----------------------------------------------------------------------
		  		# Loading and Storing Data

		 ---------------------------------------------------------------------*/
  		
		// 1. Your course resource has a CSV file "iris.csv". 
		//Load that file into an RDD called irisRDD
		//Cache the RDD and count the number of lines
		
		System.out.println("\n---------- Loading and Storing Data -----------------\n");

		JavaRDD<String> irisRDD = spContext.textFile("data/iris.csv");
		irisRDD.cache();
		System.out.println(" Iris RDD count : " + irisRDD.count());
		ExerciseUtils.printStringRDD(irisRDD, 5);
		
		/*----------------------------------------------------------------------
  				# Spark Transformations

		 ---------------------------------------------------------------------*/
  		
		 // 1. Create a new RDD from irisRDD with the following transformations
		 // 		- The name of the Species should be all Capitals
		 // 		- The numeric values should be rounded off (as integers)
		
		System.out.println("\n---------- Spark Transformations -----------------\n");
		
		JavaRDD<String> xformedIris = irisRDD.map(new Function<String, String>() {

			public String call(String s) {
				
				//Dont touch the header
				if ( s.startsWith("Sepal")) {
					return s;
				}
				
				//Split the string to its elements
				String[] attList = s.split(",");
				
				//Make species upper case
				attList[4] = attList[4].toUpperCase();
				
				//Convert to integer values
				for (int i=0; i < attList.length-1; i++) {
					attList[i] = String.valueOf( Math.round( Double.valueOf(attList[i])));
				}
				
				//Concatenate array to string
				String result = attList[0];
		        for (int i=1; i<attList.length; i++) {
		            result = result + "," + attList[i];
		        }
				return result;
			}
		});
		
		ExerciseUtils.printStringRDD(xformedIris, 5);
		
		// 2. Filter irisRDD for lines that contain versicolor and count them.
		
		JavaRDD<String> versiRDD = irisRDD.filter( str -> str.contains("versicolor"));
		System.out.println("Versicolor records = " + versiRDD.count());
		
		/*----------------------------------------------------------------------
				# Spark Actions
 		---------------------------------------------------------------------*/
		
		System.out.println("\n---------- Spark Actions -----------------\n");
		
		//Find the average Sepal.Length for all flowers in the irisRDD
		
		String totalMPG = irisRDD.reduce(new Function2<String, String, String>() {

			public String call(String arg0, String arg1) {
				
				double firstVal = 0.0;
				double secondVal = 0.0;

				// First parameter - might be a numeric or string. handle appropriately
				firstVal = (isNumeric(arg0) ? Double.valueOf(arg0) : getSepalLength(arg0));
				// Second parameter.
				secondVal = (isNumeric(arg1) ? Double.valueOf(arg1) : getSepalLength(arg1));

				return Double.valueOf(firstVal + secondVal).toString();

			}
			
			private double getSepalLength(String str) {
				String[] attList = str.split(",");
				if (isNumeric(attList[0])) {
					return Double.valueOf(attList[0]);
				} else {
					return 0.0;
				}
			}
			
			private boolean isNumeric(String s) {
				return s.matches("[-+]?\\d*\\.?\\d+");
			}
		});
 
		double avgSepalLength = ( Double.valueOf(totalMPG) / ( irisRDD.count() - 1) ) ;
		System.out.println("Average Sepal Length = " + avgSepalLength );
		
		/*----------------------------------------------------------------------
				# Key-value RDDs
		---------------------------------------------------------------------*/
		System.out.println("\n---------- Key Value RDDs -----------------\n");
		
		// Convert the irisRDD into a key-value RDD with Species as key and Sepal.Length
		// as the value.
		// Then find the maximum of Sepal.Length by each Species.
		
		//First split as KV
		JavaPairRDD<String, Double[]> irisKV 
			= irisRDD.mapToPair(new PairFunction<String, String, Double[]>() {

			@Override
			public Tuple2<String, Double[]> call(String arg0) throws Exception {

				String[] attList = arg0.split(",");
				// Handle header line
				Double[] sepalLength = { (attList[0].equals("SepalLength") ? 0
						: Double.valueOf(attList[0])), 1.0 };
				return new Tuple2<String, Double[]>(attList[4], sepalLength);
			}

		});
		
		//Find sum of Sepal Length and Count by Key
		JavaPairRDD<String, Double[]> irisSumKV 
				= irisKV.reduceByKey(new Function2<Double[], Double[], Double[]>() {
					
					public Double[] call(Double[] arg0, Double[] arg1) throws Exception {

						Double[] retval = { arg0[0] + arg1[0], arg0[1] + arg1[1] };
						return retval;
					}
				});
		
		// Now find average
		JavaPairRDD<String, Double> irisAvgKV 
				= irisSumKV.mapValues(x -> x[0] / x[1]);

		System.out.println("KV RDD Practice - Tuples after averaging :");
		for (Tuple2<String, Double> kvList : irisAvgKV.take(5)) {
			System.out.println(kvList);
		}
		
		/*----------------------------------------------------------------------
				# Advanced Spark
		---------------------------------------------------------------------*/
		System.out.println("\n---------- Advanced Spark -----------------\n");
		
		// Find the number of records in irisRDD, whose Sepal.Length is 
		// greater than the Average Sepal Length we found in the earlier practice
		//
		// Note: Use Broadcast and Accumulator variables for this practice
		
		LongAccumulator gtavgCount = spContext.sc().longAccumulator();

		Broadcast<Double> avgLength = spContext.broadcast(avgSepalLength);
		
		JavaRDD<String> irisOut = irisRDD.map(new Function<String, String>() {
			public String call(String s) {

				if (! s.startsWith("Sepal") ) {
					String[] attList = s.split(",");
					Double sepalLength = Double.valueOf(attList[0]);
					if ( sepalLength > avgLength.value()) {
						gtavgCount.add(1);
					}
				}
				
				return s;
			}
		});
		irisOut.count();
		System.out.println("No. of records with Sepal Length > Avg : " + gtavgCount.sum());
		
		/*----------------------------------------------------------------------
				Hope you had some good practice !
		---------------------------------------------------------------------*/
	}

}
