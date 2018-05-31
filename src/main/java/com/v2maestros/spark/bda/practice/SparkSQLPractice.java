/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Practice Exercises : Spark SQL
*****************************************************************************/

package com.v2maestros.spark.bda.practice;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.max;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.v2maestros.spark.bda.common.SparkConnection;

public class SparkSQLPractice {
	
	public static void main(String[] args) {
		
		/*----------------------------------------------------------------------
		 * 		Common Setup
		 ---------------------------------------------------------------------*/
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		JavaSparkContext spContext = SparkConnection.getContext();
		SparkSession spSession = SparkConnection.getSession();
		
		/*----------------------------------------------------------------------
		  		# Spark Data Frames
		 ---------------------------------------------------------------------*/
		
		// 1. Your dataset has a file iris.csv. load it into a data frame irisDf
		//	All Column Datatypes other than for Species should be of Double type
		// Print the contents and the schema.
		
		//Create the schema for the data to be loaded into Dataset.
		StructType irisSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("SEPAL_LENGTH", DataTypes.DoubleType, false),
						DataTypes.createStructField("SEPAL_WIDTH", DataTypes.DoubleType, false),
						DataTypes.createStructField("PETAL_LENGTH", DataTypes.DoubleType, false),
						DataTypes.createStructField("PETAL_WIDTH", DataTypes.DoubleType, false),
						DataTypes.createStructField("SPECIES", DataTypes.StringType, false) 
					});

		//Load data file into an RDD
		JavaRDD<String> rdd1 = spContext.textFile("data/iris.csv");
		//Remove the header line
		String header = rdd1.first();
		JavaRDD<String> rdd2 = rdd1.filter(s -> !s.equals(header));
		
		//Function to map to Row
		JavaRDD<Row> rdd3 = rdd2.map( new Function<String, Row>() {

			@Override
			public Row call(String row) throws Exception {
				
				String[] attList = row.split(",");
				
				Row retRow = RowFactory.create( Double.valueOf(attList[0]), 
								Double.valueOf(attList[1]), 
								Double.valueOf(attList[2]), 
								Double.valueOf(attList[3]), 								
								attList[4]);
				
				return retRow;
			}

		});
		
		//Create Data Frame back.
		Dataset<Row> irisDf = spSession.createDataFrame(rdd3, irisSchema);
		
		System.out.println("\n---------- Spark Data Frames -----------------\n");
		System.out.println("1. Transformed Data :");
		irisDf.show(5);
		irisDf.printSchema();
		
		//2. Select Sepal Length and Sepal Width where Petal Length > 2.0
		
		System.out.println("2. Sepal Length and Sepal Width where Petal Length > 2.0 : ");
		
		irisDf.filter(col("PETAL_LENGTH").gt(2.0))
			.select(col("SEPAL_LENGTH"),col("SEPAL_WIDTH")).show(5);
		
		//3. Find max-of Sepal Width and average-of Petal Length by Species.
		
		System.out.println("2. Max of Sepal Width and Average of Petal Length by Species: ");
		
		irisDf.groupBy(col("SPECIES"))
			.agg(max(irisDf.col("SEPAL_WIDTH")), avg(irisDf.col("PETAL_LENGTH"))).show();
		
		/*----------------------------------------------------------------------
		  		# Spark Temp Views
		 ---------------------------------------------------------------------*/
		System.out.println("\n---------- Spark Temp Views -----------------\n");
		
		// 1. Create a Temp View call "IRIS" based on irisDf created in the earlier 
		// exercise
		irisDf.createOrReplaceTempView("IRIS");
		
		//2. Select Sepal Length and Sepal Width where Petal Length > 2.0
		System.out.println("2. Sepal Length and Sepal Width where Petal Length > 2.0 : ");
		
		spSession.sql("SELECT SEPAL_LENGTH, SEPAL_WIDTH " +
						"FROM IRIS WHERE PETAL_LENGTH > 2.0").show();
		
		
		//3. Find max-of Sepal Width and average-of Petal Length by Species.
		
		System.out.println("2. Max of Sepal Width and Average of Petal Length by Species: ");
		
		spSession.sql("SELECT SPECIES, max(SEPAL_WIDTH), avg(PETAL_LENGTH) " +
						"FROM IRIS GROUP BY SPECIES").show();
	}
}
