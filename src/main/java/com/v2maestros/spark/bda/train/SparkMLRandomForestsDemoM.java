package com.v2maestros.spark.bda.train;

import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class SparkMLRandomForestsDemoM {
	
	
	public static void main(String[] args) {
		//Filter loggers for only error messages
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		// create spark session and context
		SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("ManojRF");
		JavaSparkContext context = new JavaSparkContext(conf);
		
		System.setProperty("hadoop.home.dir", "C:\\Manoj\\Progamming\\MachineLearning\\Udemy\\Spark_Java\\winUtils\\hadoop-common-2.2.0-bin-master");
		
		String tempDir = "file:///C:/Manoj/Progamming/MachineLearning/Udemy/Spark_Java/spark-warehouse";
		SparkSession session = SparkSession.builder().appName("MnaojRF").master("local[2]").config("spark.sql.warehouse.dir",tempDir).getOrCreate();
		
		
		Dataset<Row> file = session.read().option("header", "true").option("sep", ";").csv("data/bank.csv");
		file.show(5);
		file.printSchema();
		
		//create schema
		StructType bankSchema = DataTypes.createStructType(new StructField[] {
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
				DataTypes.createStructField("LOAN", DataTypes.DoubleType, false),
				});
		
		//chnage dataframe back to rdd
		
		JavaRDD<Row> rdd1 = file.toJavaRDD().repartition(2);
		
		//change the datatypes of the columns
		
		JavaRDD<Row> rdd2 = rdd1.map(new Function<Row, Row>() {

			@Override
			public Row call(Row v1) throws Exception {
				
				double outcome = (v1.getString(16).equals("yes") ? 1.0: 0.0);
				double age = Double.valueOf(v1.getString(0));
				double married = v1.getString(2).equals("married") ? 1 : 0;
				double single = v1.getString(2).equals("single") ? 1 : 0;
				double divorced = v1.getString(2).equals("divorced") ? 1 : 0;
				double primary = v1.getString(3).equals("primary") ? 1 : 0;
				double secondary = v1.getString(3).equals("secondary") ? 1 : 0;
				double tertiary = v1.getString(3).equals("tertiary") ? 1 : 0;
				double defaulter = v1.getString(3).equals("no") ? 0 : 1;
				double balance = Double.valueOf(v1.getString(5));
				double loan = v1.getString(3).equals("yes") ? 1 : 0;
				Row r = RowFactory.create(outcome, age, married, single, divorced, primary, secondary, tertiary, defaulter, balance, loan);
				
				return r;
			}			
		});
		
		
		Dataset<Row> schemaData = session.createDataFrame(rdd2, bankSchema);
		
		schemaData.show();
		
		// cooreleation analysis
		
		for(StructField field : bankSchema.fields()) {
			
			if(!field.dataType().equals(DataTypes.StringType)) {
				System.out.println("Correlation of OUTCOME with "+ field.name()+" is "+ schemaData.stat().corr("OUTCOME", field.name()));
			}
			
		}
		
		JavaRDD<Row> rdd3 = schemaData.toJavaRDD().repartition(2);
		JavaRDD<LabeledPoint> rdd4 = rdd3.map(new Function<Row, LabeledPoint>() {

			@Override
			public LabeledPoint call(Row v1) throws Exception {
				
				
				LabeledPoint lp = new LabeledPoint(v1.getDouble(0), Vectors.dense(
						v1.getDouble(1),
						v1.getDouble(2),
						v1.getDouble(3),
						v1.getDouble(4),
						v1.getDouble(5),
						v1.getDouble(6),
						v1.getDouble(7)
						));
				
				
				return lp;
			}
			
		});
		
		
		
		Dataset<Row> banklp = session.createDataFrame(rdd4, LabeledPoint.class);
		
		banklp.show(5);
		
		//create string indexer
		StringIndexer indexer = new StringIndexer().setInputCol("label").setOutputCol("indLabel");
		
		StringIndexerModel siModel = indexer.fit(banklp);
		Dataset<Row> indexedBankLp = siModel.transform(banklp);
		System.out.println("Indexed bankLp:");
		indexedBankLp.show(20);
		
		//fit PCA
		PCA pca1 = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(3);
				
		PCAModel pcam = pca1.fit(indexedBankLp);
		Dataset<Row> bankPca = pcam.transform(indexedBankLp);
		bankPca.show(5);
		
		Dataset<Row>[] splits = bankPca.randomSplit(new double[] {0.7,0.3});
		Dataset<Row> trainData = splits[0];
		Dataset<Row> testData = splits[1];
		
		RandomForestClassifier rfcl = new RandomForestClassifier().setLabelCol("indLabel").setFeaturesCol("pcaFeatures");
		
		
		//convert indexed labels back to original labels
		IndexToString labelConverter = new IndexToString().setInputCol("indLabel").setOutputCol("labelStr").setLabels(siModel.labels());
		
		IndexToString predConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictionStr").setLabels(siModel.labels());
		
		RandomForestClassificationModel rfm = rfcl.fit(trainData);
		
		Dataset<Row> rawPredictions = rfm.transform(testData);
		Dataset<Row> predictions = predConverter.transform(labelConverter.transform(rawPredictions));
		
		predictions.select("labelStr","predictionStr","features").show(5);
		
		predictions.groupBy(col("labelStr"),col("predictionStr")).count().show();
		
		MulticlassClassificationEvaluator mle = new MulticlassClassificationEvaluator().setLabelCol("indLabel").setPredictionCol("prediction").setMetricName("accuracy");
		
		double accuracy = mle.evaluate(predictions);
		System.out.println("Accuracy is "+ Math.round(accuracy * 100) +" %" );
		
		
	}
	
	

}
