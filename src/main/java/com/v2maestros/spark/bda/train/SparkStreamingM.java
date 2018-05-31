package com.v2maestros.spark.bda.train;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.streaming.Duration;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import com.v2maestros.spark.bda.common.ExerciseUtils;

import scala.Function;
import scala.Tuple2;

public class SparkStreamingM {

	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		
		SparkConf conf = new SparkConf().setAppName("Manoj").setMaster("local[2]");
		
		JavaStreamingContext jStrCon = new JavaStreamingContext(conf, Durations.seconds(3));
		
		JavaReceiverInputDStream<String> lines = jStrCon.socketTextStream("localhost", 9000);
		
		lines.print();
		
		//split each line in words
		
		JavaDStream<String> words = lines.flatMap(new FlatMapFunction<String, String>() {

			@Override
			public Iterator<String> call(String t) throws Exception {
				return Arrays.asList(t.split(" ")).iterator();
			}
			
		});
		
		
		JavaPairDStream<String, Integer> maps = words.mapToPair(new PairFunction<String, String, Integer>() {
			@Override
			public Tuple2<String, Integer> call(String t) throws Exception {
				return new Tuple2<String, Integer>(t,1);
			}
		});
		
		
		JavaPairDStream<String, Integer> reduced = maps.reduceByKey(new Function2<Integer, Integer, Integer>() {
			
			@Override
			public Integer call(Integer v1, Integer v2) throws Exception {
				
				return v1 + v2;
			}
		});
		
		
		reduced.print();
		
		Function2<Integer,Integer,Integer> reduceFunc = new Function2<Integer, Integer, Integer>() {

			@Override
			public Integer call(Integer v1, Integer v2) throws Exception {
				return v1 + v2;
			}
	
		};
		
		
		JavaPairDStream<String, Integer> overlapped = reduced.reduceByKeyAndWindow(reduceFunc, Durations.seconds(15),Durations.seconds(3));
		
		overlapped.print();
		
		jStrCon.start();
		
		
		try {
			jStrCon.awaitTermination();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		ExerciseUtils.hold();
		
		
	}

}
