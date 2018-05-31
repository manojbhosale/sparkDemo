/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Random Forests and PCA

Problem Statement
*****************
The input data contains surveyed information about potential 
customers for a bank. The goal is to build a model that would 
predict if the prospect would become a customer of a bank, 
if contacted by a marketing exercise.

## Techniques Used

1. Random Forests
2. Training and Testing
3. Confusion Matrix
4. Indicator Variables
5. Variable Reduction
6. Principal Component Analysis

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
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
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
import static org.apache.spark.sql.functions.*;

import com.v2maestros.spark.bda.common.ExerciseUtils;
import com.v2maestros.spark.bda.common.SparkConnection;
import org.apache.spark.api.java.JavaRDD;

public class SparkMLRandomForestsDemo { 


	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		JavaSparkContext spContext = SparkConnection.getContext();
		SparkSession spSession = SparkConnection.getSession();
		
		/*--------------------------------------------------------------------------
		Load Data
		--------------------------------------------------------------------------*/
		Dataset<Row> bankDf = spSession.read()
				.option("header","true")
				.option("sep", ";")
				.csv("data/bank.csv");
		bankDf.show(5);
		bankDf.printSchema();
		
		/*--------------------------------------------------------------------------
		Cleanse Data
		--------------------------------------------------------------------------*/	
		//Convert all data types as double; 
		//Use Indicator variables
		
		//Create the schema for the data to be loaded into Dataset.
		StructType bankSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("OUTCOME", DataTypes.DoubleType, false),
						DataTypes.createStructField("AGE", DataTypes.DoubleType, false),
						DataTypes.createStructField("SINGLE", DataTypes.DoubleType, false),
						DataTypes.createStructField("MARRIED", DataTypes.DoubleType, false),
						DataTypes.createStructField("DIVORCED", DataTypes.DoubleType, false),
						DataTypes.createStructField("PRIMARY", DataTypes.DoubleType, false),
						DataTypes.createStructField("SECONDARY", DataTypes.DoubleType, false),
						DataTypes.createStructField("TERTIARY", DataTypes.DoubleType, false),
						DataTypes.createStructField("DEFAULT", DataTypes.DoubleType, false),
						DataTypes.createStructField("BALANCE", DataTypes.DoubleType, false),
						DataTypes.createStructField("LOAN", DataTypes.DoubleType, false) 
					});

		//Change data frame back to RDD
		JavaRDD<Row> rdd1 = bankDf.toJavaRDD().repartition(2);
		
		//Function to map.
		JavaRDD<Row> rdd2 = rdd1.map( new Function<Row, Row>() {

			@Override
			public Row call(Row iRow) throws Exception {
				
				//Convert age to float
				double age = Double.valueOf(iRow.getString(0));
				//Convert outcome to float
				double outcome = (iRow.getString(16).equals("yes") ? 1.0: 0.0 );
				
				//Create indicator variables for marital status
				double single = (iRow.getString(2).equals("single") ? 1.0 : 0.0);
				double married = (iRow.getString(2).equals("married") ? 1.0 : 0.0);
				double divorced = (iRow.getString(2).equals("divorced") ? 1.0 : 0.0);
				
				//Create indicator variables for education
				double primary = (iRow.getString(3).equals("primary") ? 1.0 : 0.0);
				double secondary = (iRow.getString(3).equals("secondary") ? 1.0 : 0.0);
				double tertiary = (iRow.getString(3).equals("tertiary") ? 1.0 : 0.0);
				
				//Convert default to float
				double dflt = (iRow.getString(4).equals("yes") ? 1.0 : 0.0);
				//Convert balance to float
				double balance = Double.valueOf(iRow.getString(5));
				//Convert loan to float
				double loan = (iRow.getString(7).equals("yes") ? 1.0 : 0.0);
				
				Row retRow = RowFactory.create( outcome, age, single, married, divorced,
								primary,secondary, tertiary, dflt, balance, loan);
				
				return retRow;
			}

		});
		
		//Create Data Frame back.
		Dataset<Row> bankCleansedDf = spSession.createDataFrame(rdd2, bankSchema);
		System.out.println("Transformed Data :");
		bankCleansedDf.show(5);
		
		/*--------------------------------------------------------------------------
		Analyze Data
		--------------------------------------------------------------------------*/
		
		//Perform correlation analysis
		for ( StructField field : bankSchema.fields() ) {
			if ( ! field.dataType().equals(DataTypes.StringType)) {
				System.out.println( "Correlation between OUTCOME and " + field.name()
				 	+ " = " + bankCleansedDf.stat().corr("OUTCOME", field.name()) );
			}
		}
		
		/*--------------------------------------------------------------------------
		Prepare for Machine Learning. 
		--------------------------------------------------------------------------*/
		
		//Convert data to labeled Point structure
		JavaRDD<Row> rdd3 = bankCleansedDf.toJavaRDD().repartition(2);
		
		JavaRDD<LabeledPoint> rdd4 = rdd3.map( new Function<Row, LabeledPoint>() {

			@Override
			public LabeledPoint call(Row iRow) throws Exception {
				
				LabeledPoint lp = new LabeledPoint(iRow.getDouble(0) , 
									Vectors.dense(iRow.getDouble(1),
											iRow.getDouble(2),
											iRow.getDouble(3),
											iRow.getDouble(4),
											iRow.getDouble(5),
											iRow.getDouble(6),
											iRow.getDouble(7),
											iRow.getDouble(8),
											iRow.getDouble(9),
											iRow.getDouble(10)));
				
				return lp;
			}

		});

		Dataset<Row> bankLp = spSession.createDataFrame(rdd4, LabeledPoint.class);
		System.out.println("Transformed Label and Features :");
		bankLp.show(5);
		
		//Add an index using string indexer.
		StringIndexer indexer = new StringIndexer()
				  .setInputCol("label")
				  .setOutputCol("indLabel");
		
		StringIndexerModel siModel = indexer.fit(bankLp);
		Dataset<Row> indexedBankLp = siModel.transform(bankLp);
		System.out.println("Indexed Bank LP :");
		indexedBankLp.show(5);
		
		//Perform PCA
		PCA pca = new PCA()
				  .setInputCol("features")
				  .setOutputCol("pcaFeatures")
				  .setK(3);
		PCAModel pcaModel = pca.fit(indexedBankLp);
		Dataset<Row> bankPCA = pcaModel.transform(indexedBankLp);
		System.out.println("PCA'ed Indexed Bank LP :");
		bankPCA.show(5);
		
		// Split the data into training and test sets (30% held out for testing).
		Dataset<Row>[] splits = bankPCA.randomSplit(new double[]{0.7, 0.3});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];
		
		/*--------------------------------------------------------------------------
		Perform machine learning. 
		--------------------------------------------------------------------------*/	
		
		//Create the object
		// Train a DecisionTree model.
		RandomForestClassifier rf = new RandomForestClassifier()
		  .setLabelCol("indLabel")
		  .setFeaturesCol("pcaFeatures");

		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString()
				  .setInputCol("indLabel")
				  .setOutputCol("labelStr")
				  .setLabels(siModel.labels());
		
		IndexToString predConverter = new IndexToString()
				  .setInputCol("prediction")
				  .setOutputCol("predictionStr")
				  .setLabels(siModel.labels());
				
		RandomForestClassificationModel rfModel = rf.fit(trainingData);
		
		//Predict on test data
		Dataset<Row> rawPredictions = rfModel.transform(testData);
		Dataset<Row> predictions = predConverter.transform(
										labelConverter.transform(rawPredictions));
		
		//View results
		System.out.println("Result sample :");
		predictions.select("labelStr", "predictionStr", "features").show(5);

		//View confusion matrix
		System.out.println("Confusion Matrix :");
		predictions.groupBy(col("labelStr"), col("predictionStr")).count().show();
		
		//Accuracy computation
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				  .setLabelCol("indLabel")
				  .setPredictionCol("prediction")
				  .setMetricName("accuracy");
				double accuracy = evaluator.evaluate(predictions);
				System.out.println("Accuracy = " + Math.round( accuracy * 100) + " %" );
				
		// Keep the program running so we can checkout things.
		ExerciseUtils.hold();
	}
	
}

/*Practice : Add other columns in the dataset and do the same prediction
See if you have better accuracy with additional columns */


