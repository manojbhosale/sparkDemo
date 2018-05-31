package com.v2maestros.spark.bda.train;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;

public class SparkMLDecisionTreesDemoM {
	private static String tempDir = "file:///C:/Manoj/Progamming/MachineLearning/Udemy/Spark_Java/spark-warehouse";

	public static void main(String[] args) {

		//logger levels
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);


		//hadoop binary
		System.setProperty("hadoop.home.dir", "C:\\Manoj\\Progamming\\MachineLearning\\Udemy\\Spark_Java\\winUtils\\hadoop-common-2.2.0-bin-master");	


		// get spaark context
		SparkConf conf = new SparkConf().setAppName("IrisDecision").setMaster("local[2]");
		JavaSparkContext context = new JavaSparkContext(conf);

		// get spark sesssion

		SparkSession session = SparkSession.builder().config(conf).config("spark.sql.warehouse.dir", tempDir).getOrCreate();


		//load data in dataframe

		Dataset<Row> iris = session.read().option("header","true")
				.option("sep", ",")
				.csv("data/iris.csv");

		iris.show(5);

		// create schema
		StructType schema = DataTypes.createStructType(new StructField[] {
				DataTypes.createStructField("SepL", DataTypes.DoubleType, true),
				DataTypes.createStructField("SepW", DataTypes.DoubleType, true),
				DataTypes.createStructField("PetL", DataTypes.DoubleType, true),
				DataTypes.createStructField("PetW", DataTypes.DoubleType, true),
				DataTypes.createStructField("Species", DataTypes.StringType, true),
		});

		// create RDD from iris data

		JavaRDD<Row> rdd1 = iris.toJavaRDD().repartition(2);

		JavaRDD<Row> rdd2 = rdd1.map(new Function<Row, Row>() {
			@Override
			public Row call(Row v1) throws Exception {

				double seplL = Double.valueOf(v1.getString(0));
				double seplW = Double.valueOf(v1.getString(1));
				double petlL = Double.valueOf(v1.getString(2));
				double petlW = Double.valueOf(v1.getString(3));
				//double species = v1.getString(4).equals("setosa") ? 0 : v1.getString(4).equals("versicolor") ? 1 : 2;
				String species = v1.getString(4);

				return RowFactory.create(seplL,seplW,petlL,petlW,species);
			}
		});

		/*
		//find coorelation
		Dataset<Row> forCorr = session.createDataFrame(rdd2, schema);

		for(StructField s : schema.fields()) {

			if(!s.name().equals("Species")) {
				System.out.println(" Correlation between SPECIES and "+ s.name() +" is :"+forCorr.stat().corr("Species", s.name()));
			}
		}
		//high correlation between petlL and PetlW
		 */		


		//Add index column for species

		Dataset<Row> forCorr = session.createDataFrame(rdd2, schema);

		StringIndexer indexer = new StringIndexer().setInputCol("Species").setOutputCol("SpeciesInd");

		StringIndexerModel indexModel = indexer.fit(forCorr);
		Dataset<Row> irisIndexed = indexModel.transform(forCorr);		

		irisIndexed.show(5);

		irisIndexed.groupBy(col("Species"),col("SpeciesInd")).count().show();


		JavaRDD<Row> rdd3 = irisIndexed.toJavaRDD().repartition(2);

		JavaRDD<LabeledPoint> rdd4 = rdd3.map(new Function<Row, LabeledPoint>() {

			@Override
			public LabeledPoint call(Row v1) throws Exception {

				LabeledPoint lp = new LabeledPoint(v1.getDouble(5), Vectors.dense(v1.getDouble(0),
						v1.getDouble(1),
						v1.getDouble(2),
						v1.getDouble(3)));

				return lp;
			}

		});			

		Dataset<Row> forMl = session.createDataFrame(rdd4, LabeledPoint.class);

		forMl.show();

//train test split
		
		Dataset<Row>[] splits = forMl.randomSplit(new double[] {0.7,0.3});
		Dataset<Row> train = splits[0];
		Dataset<Row> test = splits[1];

		//create decision tree classifier
		
		DecisionTreeClassifier dtc = new DecisionTreeClassifier().setLabelCol("label").setPredictionCol("predictions").setFeaturesCol("features");
		//convert indexed labesl back to originals
		IndexToString labelStr = new IndexToString().setInputCol("label")
				.setOutputCol("labelStr")
				.setLabels(indexModel.labels());
		
		IndexToString predStr = new IndexToString().setInputCol("predictions")
				.setOutputCol("predStr")
				.setLabels(indexModel.labels());
		
		
		DecisionTreeClassificationModel dtm = dtc.fit(train);
		Dataset<Row> rawPredictions = dtm.transform(test);
		Dataset<Row> labPredictions = predStr.transform(labelStr.transform(rawPredictions));
		System.out.println("Raw predictions ");
		rawPredictions.show(5);
		System.out.println("Results");
		//labPredictions.select("labelStr","predStr","features").show(5);
		labPredictions.show(5);
		
		System.out.println("Confusion Matrix");
		labPredictions.groupBy(col("labelStr"),col("predStr")).count().show();
		
		
		MulticlassClassificationEvaluator mle = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("predictions")
				.setMetricName("accuracy");
		
		double accuracy = mle.evaluate(labPredictions);
		
		System.out.println("Accuracy is :"+accuracy*100+"%");









	}

}
