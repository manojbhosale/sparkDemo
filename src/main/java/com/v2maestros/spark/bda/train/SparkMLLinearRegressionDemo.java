/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Linear Regression

Problem Statement
*****************
The input data set contains data about details of various car 
models. Based on the information provided, the goal is to come up 
with a model to predict Miles-per-gallon of a given model.

Techniques Used:

1. Linear Regression ( multi-variate)
2. Data Imputation - replacing non-numeric data with numeric ones
3. Variable Reduction - picking up only relevant features

*****************************************************************************/
package com.v2maestros.spark.bda.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.v2maestros.spark.bda.common.ExerciseUtils;
import com.v2maestros.spark.bda.common.SparkConnection;
import org.apache.spark.api.java.JavaRDD;

public class SparkMLLinearRegressionDemo { 


	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		JavaSparkContext spContext = SparkConnection.getContext();
		SparkSession spSession = SparkConnection.getSession();
		
		/*--------------------------------------------------------------------------
		Load Data
		--------------------------------------------------------------------------*/
		Dataset<Row> autoDf = spSession.read()
				.option("header","true")
				.csv("data/auto-miles-per-gallon.csv");
		autoDf.show(5);
		autoDf.printSchema();
		
		/*--------------------------------------------------------------------------
		Cleanse Data
		--------------------------------------------------------------------------*/	
		//Convert all data types as double; Change missing values to standard ones.
		
		//Create the schema for the data to be loaded into Dataset.
		StructType autoSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("MPG", DataTypes.DoubleType, false),
						DataTypes.createStructField("CYLINDERS", DataTypes.DoubleType, false),
						DataTypes.createStructField("DISPLACEMENT", DataTypes.DoubleType, false),
						DataTypes.createStructField("HP", DataTypes.DoubleType, false),
						DataTypes.createStructField("WEIGHT", DataTypes.DoubleType, false),
						DataTypes.createStructField("ACCELERATION", DataTypes.DoubleType, false),
						DataTypes.createStructField("MODELYEAR", DataTypes.DoubleType, false),
						DataTypes.createStructField("NAME", DataTypes.StringType, false) 
					});

		//Broadcast the default value for HP
		Broadcast<Double> avgHP = spContext.broadcast(80.0);
		
		//Change data frame back to RDD
		JavaRDD<Row> rdd1 = autoDf.toJavaRDD().repartition(2);
		
		//Function to map.
		JavaRDD<Row> rdd2 = rdd1.map( new Function<Row, Row>() {

			@Override
			public Row call(Row iRow) throws Exception {
				
				double hp = (iRow.getString(3).equals("?") ?
						avgHP.value() : Double.valueOf(iRow.getString(3))); 
				
				Row retRow = RowFactory.create( Double.valueOf(iRow.getString(0)), 
								Double.valueOf(iRow.getString(1)), 
								Double.valueOf(iRow.getString(2)), 
								Double.valueOf(hp),
								Double.valueOf(iRow.getString(4)), 
								Double.valueOf(iRow.getString(5)), 
								Double.valueOf(iRow.getString(6)), 
								iRow.getString(7)
						);
				
				return retRow;
			}

		});
		
		//Create Data Frame back.
		Dataset<Row> autoCleansedDf = spSession.createDataFrame(rdd2, autoSchema);
		System.out.println("Transformed Data :");
		autoCleansedDf.show(5);
		
		/*--------------------------------------------------------------------------
		Analyze Data
		--------------------------------------------------------------------------*/
		
		//Perform correlation analysis
		for ( StructField field : autoSchema.fields() ) {
			if ( ! field.dataType().equals(DataTypes.StringType)) {
				System.out.println( "Correlation between MPG and " + field.name()
				 	+ " = " + autoCleansedDf.stat().corr("MPG", field.name()) );
			}
		}
		
		/*--------------------------------------------------------------------------
		Prepare for Machine Learning. 
		--------------------------------------------------------------------------*/
		
		//Convert data to labeled Point structure
		JavaRDD<Row> rdd3 = autoCleansedDf.toJavaRDD().repartition(2);
		
		JavaRDD<LabeledPoint> rdd4 = rdd3.map( new Function<Row, LabeledPoint>() {

			@Override
			public LabeledPoint call(Row iRow) throws Exception {
				
				LabeledPoint lp = new LabeledPoint(iRow.getDouble(0) , 
									Vectors.dense(iRow.getDouble(2),
											iRow.getDouble(4),
											iRow.getDouble(5)));
				
				return lp;
			}

		});

		Dataset<Row> autoLp = spSession.createDataFrame(rdd4, LabeledPoint.class);
		autoLp.show(5);
		
		// Split the data into training and test sets (10% held out for testing).
		Dataset<Row>[] splits = autoLp.randomSplit(new double[]{0.9, 0.1});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];
		
		/*--------------------------------------------------------------------------
		Perform machine learning. 
		--------------------------------------------------------------------------*/	
		
		//Create the object
		LinearRegression lr = new LinearRegression();
		//Create the model
		LinearRegressionModel lrModel = lr.fit(trainingData);
		
		//Print out coefficients and intercept for LR
		System.out.println("Coefficients: "
				  + lrModel.coefficients() + " Intercept: " + lrModel.intercept());
		
		//Predict on test data
		Dataset<Row> predictions = lrModel.transform(testData);
		
		//View results
		predictions.select("label", "prediction", "features").show(5);
		
		//Compute R2 for the model on test data.
		RegressionEvaluator evaluator = new RegressionEvaluator()
				  .setLabelCol("label")
				  .setPredictionCol("prediction")
				  .setMetricName("r2");
		double r2 = evaluator.evaluate(predictions);
		System.out.println("R2 on test data = " + r2);
		
		// Keep the program running so we can checkout things.
		ExerciseUtils.hold();
	}

    /*Practice : Try Regression with different sets of feature variables
    and see how regression accuracy varies based on correlation */
	
}
