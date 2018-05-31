
/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Operations
*****************************************************************************/

package com.v2maestros.spark.bda.train;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.LongAccumulator;

import com.v2maestros.spark.bda.common.DataResource;
import com.v2maestros.spark.bda.common.ExerciseUtils;
import com.v2maestros.spark.bda.common.SparkConnection;

import scala.Tuple2;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

public class SparkOperationsDemo {

	public static void main(String[] args) {

		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		JavaSparkContext spContext = SparkConnection.getContext();

		/*-------------------------------------------------------------------
		 * Loading and Storing Data
		 -------------------------------------------------------------------*/

		// Create RDD from a list
		JavaRDD<Integer> collData = DataResource.getCollData();
        //Lazy-evaluation. creation happens only during action.
		System.out.println("Total records = " + collData.count());

		// Create a RDD from a file
		JavaRDD<String> autoAllData 
            = spContext.textFile("data/auto-data.csv");
		System.out.println("Total Records in Autodata : " 
            + autoAllData.count());
		System.out.println("Spark Operations : Load from CSV");
		ExerciseUtils.printStringRDD(autoAllData, 5);

		// Storing RDDs
		// Make sure the file does not exist.
		autoAllData.saveAsTextFile("data/auto-data-modified.csv");

		// RDDs can also be converted back to Java Lists with collect(). Then
		// Java lists can be stored using any data sink and any technique
		// available in standard Java.
        List<String> autoList = autoAllData.collect();

		/*-------------------------------------------------------------------
		 * Spark Transformation
		 -------------------------------------------------------------------*/

		// Map Example: Change CSV to TSV
		JavaRDD<String> tsvData = autoAllData
            .map(str -> str.replace(",", "\t"));
		System.out.println("Spark Operations : MAP");
		ExerciseUtils.printStringRDD(tsvData, 5);

		// Remove first header line
		String header = autoAllData.first();
		JavaRDD<String> autoData 
            = autoAllData.filter(s -> !s.equals(header));

		// Filter Example : Filter data for only "toyota" with inline lambda
		// function
		JavaRDD<String> toyotaData 
            = autoData.filter(str -> str.contains("toyota"));
		System.out.println("Spark Operations : FILTER");
		ExerciseUtils.printStringRDD(toyotaData, 5);

		// Distinct Example
		System.out.println("Spark Distinct Example : " 
					+ collData.distinct().collect());

		// FlatMap with inline function
		System.out.println("Spark Operations : FLAT MAP");

		JavaRDD<String> words 
            = toyotaData.flatMap(new FlatMapFunction<String, String>() {

			public Iterator<String> call(String s) {
				return Arrays.asList(s.split(",")).iterator();
			}
		});

		System.out.println("Toyota words count :" + words.count());
		ExerciseUtils.printStringRDD(words, 10);

		// Using functions for Map
		JavaRDD<String> cleansedRDD = autoData.map(new cleanseRDD());
		System.out.println("Using Functions for Map : ");
		ExerciseUtils.printStringRDD(cleansedRDD, 5);

		// Set operations
		JavaRDD<String> words1 = spContext.parallelize(
				Arrays.asList("hello", "war", "peace", "world"));
		JavaRDD<String> words2 = spContext.parallelize(
				Arrays.asList("war", "peace", "universe"));

		System.out.println("Example for Set operations : Union:");
		ExerciseUtils.printStringRDD(words1.union(words2), 10);
		System.out.println("Example for Set operations : Intersection");
		ExerciseUtils.printStringRDD(words1.intersection(words2), 10);

		/*-------------------------------------------------------------------
		 * Spark Actions
		 -------------------------------------------------------------------*/
		// Find sum of numbers in collData
		int collCount = collData.reduce((x, y) -> x + y);
		System.out.println("Compute sum using reduce " + collCount);

		// Find shortest line in autoData with lambda function
		String shortest 
            = autoData.reduce(new Function2<String, String, String>() {
			public String call(String x, String y) {
				return (x.length() < y.length() ? x : y);
			}
		});
		System.out.println("The shortest string is " + shortest);

		// Use external function to compute Average MPG
		String totMPG = autoData.reduce(new totalMPG());
		System.out.println("Average MPG is " + 
				(Integer.valueOf(totMPG) / (autoData.count() )));

		/*-------------------------------------------------------------------
		 * Key Value / Pair RDDs
		 -------------------------------------------------------------------*/
		// Create a KV RDD with auto Brand and Horse Power.

		JavaPairRDD<String, Integer[]> autoKV 
                = autoData.mapToPair(new getKV());

		System.out.println("KV RDD Demo - raw tuples :");
		for (Tuple2<String, Integer[]> kvList : autoKV.take(5)) {
			System.out.println(kvList._1 + " - " 
					+ kvList._2[0] + " ,  " + kvList._2[1]);
		}

		// Compute Average HP by each key
		// Find summarize total HP and total Count
		JavaPairRDD<String, Integer[]> autoSumKV 
			= autoKV.reduceByKey(new computeTotalHP());

		System.out.println("KV RDD Demo - Tuples after summarizing :");
		for (Tuple2<String, Integer[]> kvList : autoSumKV.take(5)) {
			System.out.println(kvList._1 + " - " + 
					kvList._2[0] + " ,  " + kvList._2[1]);
		}

		// Now find average
		JavaPairRDD<String, Integer> autoAvgKV 
			= autoSumKV.mapValues(x -> x[0] / x[1]);

		System.out.println("KV RDD Demo - Tuples after averaging :");
		for (Tuple2<String, Integer> kvList : autoAvgKV.take(5)) {
			System.out.println(kvList);
		}

		/*-------------------------------------------------------------------
		 * Accumulators and Broadcast variables.
		 -------------------------------------------------------------------*/

		LongAccumulator sedanCount = spContext.sc().longAccumulator();
		LongAccumulator hatchbackCount = spContext.sc().longAccumulator();

		Broadcast<String> sedanText = spContext.broadcast("sedan");
		Broadcast<String> hatchbackText = spContext.broadcast("hatchback");

		JavaRDD<String> autoOut 
            = autoData.map(new Function<String, String>() {
			public String call(String x) {

				if (x.contains(sedanText.value())) {
					sedanCount.add(1);
				}
				if (x.contains(hatchbackText.value())) {
					hatchbackCount.add(1);
				}
				return x;
			}
		});

		// Execute an action to force the map. Otherwise accumulators are not
		// triggered.
		autoOut.count();

		System.out.println("Demo for Accumulators and Broadcasts : ");
		System.out.println("Sedan Count : " + sedanCount.value() +
				"  HatchBack Count : " + hatchbackCount.value());

		/*-------------------------------------------------------------------
		 * Partitioning and Persistence
		 -------------------------------------------------------------------*/
		autoData.cache();

		System.out.println("No. of partitions in autoData = " + 
				autoData.getNumPartitions());
		JavaRDD<String> wordsList = spContext.parallelize(
				Arrays.asList("hello", "war", "peace", "world"), 4);
		System.out.println("No. of partitions in wordsList = " + 
				wordsList.getNumPartitions());

		// Keep the program running so we can checkout things.
		ExerciseUtils.hold();

	}

}

/* class to cleanse the autoRDD */
class cleanseRDD implements Function<String, String> {

	@Override
	public String call(String autoStr) throws Exception {

		// Splite into attributes
		String[] attList = autoStr.split(",");
		// Change number of doors to a number
		attList[3] = (attList[3].equals("two")) ? "2" : "4";
		// Change drive to upper case
		attList[5] = attList[5].toUpperCase();

		// Return modified string
		return Arrays.toString(attList);
	}

}

// Extract Total MPG from auto string with reduce class
class totalMPG implements Function2<String, String, String> {

	@Override
	public String call(String arg0, String arg1) throws Exception {

		int firstVal = 0;
		int secondVal = 0;

		// First parameter - might be a numeric or string. handle appropriately
		firstVal = (isNumeric(arg0) ? 
            Integer.valueOf(arg0) : getMPGValue(arg0));
		// Second parameter.
		secondVal = (isNumeric(arg1) ? 
            Integer.valueOf(arg1) : getMPGValue(arg1));

		return Integer.valueOf(firstVal + secondVal).toString();
	}

	// Internal function to extract MPG
	private int getMPGValue(String str) {
		// System.out.println(str);
		String[] attList = str.split(",");
		if (isNumeric(attList[9])) {
			return Integer.valueOf(attList[9]);
		} else {
			return 0;
		}
	}

	// Internal function to check if value is numeric
	private boolean isNumeric(String s) {
		return s.matches("[-+]?\\d*\\.?\\d+");
	}

}

// KV Pair Class to make pairs
class getKV implements PairFunction<String, String, Integer[]> {

	@Override
	public Tuple2<String, Integer[]> call(String arg0) throws Exception {

		String[] attList = arg0.split(",");
		// Handle header line
		Integer[] hpVal = { (attList[7].equals("HP") ?
            0 : Integer.valueOf(attList[7])), 1 };
		return new Tuple2<String, Integer[]>(attList[0], hpVal);
	}

}

// Class to compute Average HP by make.
class computeTotalHP implements 
        Function2<Integer[], Integer[], Integer[]> {

	@Override
	public Integer[] call(Integer[] arg0, Integer[] arg1) 
                throws Exception {

		Integer[] retval = { arg0[0] + arg1[0], arg0[1] + arg1[1] };
		return retval;
	}

}
