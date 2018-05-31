/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Code Samples : Recommendations

Problem Statement
*****************
The input data contains a file with user, item and ratings. 
The purpose of the exercise is to build a recommendation model
and then predict the affinity for users to various items

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
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
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
import org.apache.spark.util.DoubleAccumulator;
import org.apache.spark.util.LongAccumulator;

import static org.apache.spark.sql.functions.*;

import com.v2maestros.spark.bda.common.ExerciseUtils;
import com.v2maestros.spark.bda.common.SparkConnection;
import org.apache.spark.api.java.JavaRDD;

public class SparkMLRecommendationsDemo { 


	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		JavaSparkContext spContext = SparkConnection.getContext();
		SparkSession spSession = SparkConnection.getSession();
		
		/*--------------------------------------------------------------------------
		Load Data
		--------------------------------------------------------------------------*/
		
		Dataset<Row> rawDf = spSession.read()
							.csv("data/UserItemData.txt");
		rawDf.show(5);
		rawDf.printSchema();
		
		/*--------------------------------------------------------------------------
		Cleanse Data - convert data type
		--------------------------------------------------------------------------*/	
		
		//Create the schema for the data to be loaded into Dataset.
		StructType ratingsSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("user", DataTypes.IntegerType, false),
						DataTypes.createStructField("item", DataTypes.IntegerType, false),
						DataTypes.createStructField("rating", DataTypes.DoubleType, false) 
					});
		
		JavaRDD<Row> rdd1 = rawDf.toJavaRDD().repartition(2);
		
		//Function to map.
		JavaRDD<Row> rdd2 = rdd1.map( new Function<Row, Row>() {

			@Override
			public Row call(Row iRow) throws Exception {
				
				Row retRow = RowFactory.create( 
								Integer.valueOf(iRow.getString(0)),
								Integer.valueOf(iRow.getString(1)), 
								Double.valueOf(iRow.getString(2)) );
				
				return retRow;
			}

		});

		Dataset<Row> ratingsDf = spSession.createDataFrame(rdd2, ratingsSchema);
		System.out.println("Ratings Data: ");
		ratingsDf.show(5);
		
		/*--------------------------------------------------------------------------
		Perform Machine Learning
		--------------------------------------------------------------------------*/

		Dataset<Row>[] splits = ratingsDf.randomSplit(new double[]{0.9, 0.1});
		Dataset<Row> training = splits[0];
		Dataset<Row> test = splits[1];
		
		ALS als = new ALS()
				  .setMaxIter(5)
				  .setRegParam(0.01)
				  .setUserCol("user")
				  .setItemCol("item")
				  .setRatingCol("rating");
		
		ALSModel model = als.fit(training);

		// Evaluate the model by computing the RMSE on the test data
		Dataset<Row> predictions = model.transform(test);
		
		System.out.println("Predictions : ");
		predictions.show();
		
		// Keep the program running so we can checkout things.
		ExerciseUtils.hold();
	}
	
}
