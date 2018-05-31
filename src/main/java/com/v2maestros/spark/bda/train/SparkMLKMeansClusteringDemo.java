/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Code Samples : K-Means Clustering

The input data contains samples of cars and technical / price 
information about them. The goal of this problem is to group 
these cars into 4 clusters based on their attributes

## Techniques Used

1. K-Means Clustering
2. Centering and Scaling

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

public class SparkMLKMeansClusteringDemo { 


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
				.csv("data/auto-data.csv");
		autoDf.show(5);
		autoDf.printSchema();
		
		/*--------------------------------------------------------------------------
		Cleanse Data convert data type
		--------------------------------------------------------------------------*/	
		
		//Create the schema for the data to be loaded into Dataset.
		StructType autoSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("DOORS", DataTypes.DoubleType, false),
						DataTypes.createStructField("BODY", DataTypes.DoubleType, false),
						DataTypes.createStructField("HP", DataTypes.DoubleType, false),
						DataTypes.createStructField("RPM", DataTypes.DoubleType, false),
						DataTypes.createStructField("MPG", DataTypes.DoubleType, false) 
					});
		
		JavaRDD<Row> rdd1 = autoDf.toJavaRDD().repartition(2);
		
		//Function to map.
		JavaRDD<Row> rdd2 = rdd1.map( new Function<Row, Row>() {

			@Override
			public Row call(Row iRow) throws Exception {
				
				double doors = ( iRow.getString(3).equals("two") ? 1.0 : 2.0);
				double body = ( iRow.getString(4).equals("sedan") ? 1.0 : 2.0);
				
				Row retRow = RowFactory.create( doors, body,
								Double.valueOf(iRow.getString(7)),
								Double.valueOf(iRow.getString(8)), 
								Double.valueOf(iRow.getString(9)) );
				
				return retRow;
			}

		});
		
		//Create Data Frame back.
		Dataset<Row> autoCleansedDf = spSession.createDataFrame(rdd2, autoSchema);
		System.out.println("Transformed Data :");
		autoCleansedDf.show(5);
		
		/*--------------------------------------------------------------------------
		Prepare for Machine Learning - Perform Centering and Scaling
		--------------------------------------------------------------------------*/
		Row meanRow = autoCleansedDf.agg(avg(autoCleansedDf.col("DOORS")), 
							avg(autoCleansedDf.col("BODY")),
							avg(autoCleansedDf.col("HP")),
							avg(autoCleansedDf.col("RPM")),
							avg(autoCleansedDf.col("MPG")))
					.toJavaRDD().takeOrdered(1).get(0)  ;
		Row stdRow = autoCleansedDf.agg(stddev(autoCleansedDf.col("DOORS")), 
							stddev(autoCleansedDf.col("BODY")),
							stddev(autoCleansedDf.col("HP")),
							stddev(autoCleansedDf.col("RPM")),
							stddev(autoCleansedDf.col("MPG")))
					.toJavaRDD().takeOrdered(1).get(0)  ;

		System.out.println("Mean Values : " + meanRow);
		System.out.println("Std Dev Values : " + stdRow);
		
		Broadcast<Row> bcMeanRow = spContext.broadcast(meanRow);
		Broadcast<Row> bcStdRow = spContext.broadcast(stdRow);
		DoubleAccumulator rowId = spContext.sc().doubleAccumulator();
		rowId.setValue(1);
		
		//Perform center-and-scale and create a vector
		JavaRDD<Row> rdd3 = autoCleansedDf.toJavaRDD().repartition(2);
		JavaRDD<LabeledPoint> rdd4 = rdd3.map( new Function<Row, LabeledPoint>() {

			@Override
			public LabeledPoint call(Row iRow) throws Exception {
				
				double doors = (bcMeanRow.value().getDouble(0) - iRow.getDouble(0))
								/ bcStdRow.value().getDouble(0);
				double body =  (bcMeanRow.value().getDouble(1) - iRow.getDouble(1))
								/ bcStdRow.value().getDouble(1);
				double hp =  (bcMeanRow.value().getDouble(2) - iRow.getDouble(2))
								/ bcStdRow.value().getDouble(2);
				double rpm =  (bcMeanRow.value().getDouble(3) - iRow.getDouble(3))
								/ bcStdRow.value().getDouble(3);
				double mpg =  (bcMeanRow.value().getDouble(4) - iRow.getDouble(4))
								/ bcStdRow.value().getDouble(4);
				
				double id= rowId.value();
				rowId.setValue(rowId.value()+1);
				
				LabeledPoint lp = new LabeledPoint( id,
						Vectors.dense( doors, body, hp, rpm, mpg));
				
				return lp;
			}

		});

		Dataset<Row> autoVector = spSession.createDataFrame(rdd4, LabeledPoint.class );
		System.out.println("Centered and scaled vector :" + autoVector.count());
		autoVector.show(5);
		
		/*--------------------------------------------------------------------------
		Perform Machine Learning
		--------------------------------------------------------------------------*/
		KMeans kmeans = new KMeans()
							.setK(4)
							.setSeed(1L);
		
		KMeansModel model = kmeans.fit(autoVector);
		Dataset<Row> predictions = model.transform(autoVector);
		
		System.out.println("Groupings : ");
		predictions.show(5);
		
		System.out.println("Groupings Summary : ");
		predictions.groupBy(col("prediction")).count().show();

		// Keep the program running so we can checkout things.
		ExerciseUtils.hold();
	}
	
}
