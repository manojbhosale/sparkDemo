/** File provided by V2 Maestros for its students for learning purposes only
 * Copyright @2016 All rights reserved.
 */
package com.v2maestros.spark.bda.common;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class DataResource {

	public static JavaRDD<Integer> getCollData() {
		JavaSparkContext spContext = SparkConnection.getContext();
		List<Integer> data = Arrays.asList(3,6,3,4,8);
		JavaRDD<Integer> collData = spContext.parallelize(data);
		collData.cache();
		return collData;
	}
}
