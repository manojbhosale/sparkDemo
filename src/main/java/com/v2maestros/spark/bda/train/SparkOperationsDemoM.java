package com.v2maestros.spark.bda.train;

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

import com.v2maestros.spark.bda.common.SparkConnection;

import scala.Tuple2;

public class SparkOperationsDemoM {
	
	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR); //Suppress the info level msgs from spark, to make the console less crowded
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		JavaSparkContext context  = SparkConnection.getContext(); 
		
		JavaRDD<String> irisRDD = context.textFile("data/iris.csv"); //just creates a RDD by splitting the file in individual lines. Does not interpret the CSV format
		
		//collect to get list		
		//List<String> irisList = irisRDD.collect();
		//irisList.forEach(System.out::println);
		
		//System.out.println(irisRDD.count());
		//System.out.println(irisRDD.take(5));
		
		//MAP DEMO
		// Returns the transformed RDD
		JavaRDD<String> transformed = irisRDD.map(s -> s.contains("setosa") ? s.replace("setosa", "1") : s.contains("virginica") ? s.replace("virginica", "2") : s.replace("versicolor", "3"));
		
		
		//FILTER DEMO

		JavaRDD<String> setosaFiltered = irisRDD.filter(s -> s.contains("virginica"));
		
		
		//Parallelize
		JavaRDD<Integer> intRDD =  context.parallelize(Arrays.asList(1,2,3,4,5,5,3,1,11,12,1));
		JavaRDD<Integer> intRDD1 =  context.parallelize(Arrays.asList(11,12,1,13,14,15,16));
		
		//DISTINCT and filter
		JavaRDD<Integer> distRdd = intRDD.distinct();
		
		//UNION of the RDDs
		JavaRDD<Integer> unionRdd = intRDD.union(intRDD1);
		JavaRDD<Integer> intersectRdd = intRDD.intersection(intRDD1);
		
		//get first line of RDD
		String firstSetosa = setosaFiltered.first();
		System.out.println(firstSetosa);
		
		
		//flatMap
		JavaRDD<String> words = setosaFiltered.flatMap(new FlatMapFunction<String, String>() {

			@Override
			public Iterator<String> call(String arg0) throws Exception {
				return Arrays.asList(arg0.split(",")).iterator();
			}
			
			
		});
		
		
		System.out.println(words.count());
		
		//use of class implementing the Function class
		JavaRDD<String> stringNume = transformed.map(new CleanseRDDM());
		
		//Load tweets data
		JavaRDD<String> rdd = context.textFile("data/movietweets.csv");
		
		
		//REDUCE
		String combinedRdd = rdd.reduce((x, y)->  x + y );
		
		//largest String
		String largestString = rdd.reduce(new Function2<String, String, String>() {
			@Override
			public String call(String v1, String v2) throws Exception {
				return (v1.length() > v2.length() ? v1 : v2);
			}
		});
		
		System.out.println(largestString);
		
		//printRDD(stringNume, 100);
		
		
		//ASSIGNMENT 1: transformation . Transform all numeric values in irisRDD into float. Find out average of the sepal lengths
		String header = irisRDD.first();
		irisRDD = irisRDD.filter(m -> !m.equals(header));
		
		String totalSepalLength = irisRDD.reduce(new Function2<String, String, String>() {

			@Override
			public String call(String v1, String v2) throws Exception {
				Float val1 = 0f;
				Float val2 = 0f;
				val1 = isFloat(v1) ? Float.valueOf(v1) : getFloatValue(v1);
				val2 = isFloat(v2) ? Float.valueOf(v2) : getFloatValue(v2);
				return  Float.valueOf(val1 + val2).toString();
			}
			
		});

		
		//System.out.println(isFloat("5"));
		//printRDD(irisRDD, 5);
		System.out.println("Average sepal length :: "+Float.valueOf(totalSepalLength)/irisRDD.count());
		
		//ASSIGNMENT 2: create pair RDD where species is Key and Sepal Length is value
			//create pair RDD
			JavaPairRDD<String, Double> sepalpair = irisRDD.mapToPair(new GetPair());
			
			for(Tuple2<String, Double> tup : sepalpair.take(5)) {
				//System.out.println(tup._1+" "+tup._2);
			}
			
			//Find min Sepal length
			JavaPairRDD<String, Double> minvalues = sepalpair.reduceByKey(new GetMinSepalLengths());
			
			for(Tuple2<String, Double> tup : minvalues.take(5)) {
				//System.out.println(tup._1+" "+tup._2);
			}

			
		
		//printRDD(distRdd, 151); //take() does not throw an error if the provided size is greater than the size of RDD
		
		// Broadcasters and accumulators
		LongAccumulator numSepals = context.sc().longAccumulator();
		Double avgSepalLength = Double.valueOf(totalSepalLength)/irisRDD.count();
		Broadcast<Double> avgLength = context.broadcast(avgSepalLength);
		
		JavaRDD<String> aboveAvgSepals = irisRDD.filter(new Function<String, Boolean>() {

			@Override
			public Boolean call(String v1) throws Exception {
				String[] vals = v1.split(",");
				Double d= Double.valueOf(vals[0]);
				boolean res = d > avgLength.value();//use of broadcast
				if(res) { 
					
					numSepals.add(1);
				}
				return res; 
			}
			
		});
		
		aboveAvgSepals.count(); //IMPORTANT: without this statement the accumulators wont be triggered.
		System.out.println("Num of above average sepals : "+numSepals.value());
		//printRDD(aboveAvgSepals, 100);
		

	}
	
	public static Float getFloatValue(String val1) {
		String[] rec2 = val1.split(",");
		return Float.valueOf(rec2[0]);
	}
	
	public static boolean isFloat(String str){
		return str.matches("^[+-]?\\d+\\.?\\d?+$");
	}
	
	public static <T> void printRDD(JavaRDD<T> rdd, int count){
		for (T s: rdd.take(count)) {
			System.out.println(s);
		}
	}
	
	
}

class CleanseRDDM implements Function<String, String>{

	@Override
	public String call(String v1) throws Exception {
		String[] arr = v1.split(",");
		
		arr[4] = arr[4].equals("2") ? "two" : arr[4].equals("3") ? "three" : "one";
		
		return Arrays.toString(arr);	
	}
	
	
	
}

class GetPair implements PairFunction<String, String, Double>{

	@Override
	public Tuple2<String, Double> call(String t) throws Exception {
		String[] splited = t.split(",");
		Double vals = Double.valueOf(splited[0]);
		return new Tuple2<String, Double>(splited[splited.length-1],vals);
	}
	
}

class GetMinSepalLengths implements Function2<Double, Double, Double>{

	@Override
	public Double call(Double v1, Double v2) throws Exception {
		return v1 > v2 ? v2 : v1;
	}
	
}
