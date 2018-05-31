package com.v2maestros.spark.bda.train;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.sun.corba.se.impl.protocol.BootstrapServerRequestDispatcher;

public class SparkMLLinearRegressionDemoM {

	public static void main(String[] args) {

		// set logger level
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		//get spark context
		//create spark configuration

		SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("ManojLinear");
		System.setProperty("hadoop.home.dir", "C:\\Manoj\\Progamming\\MachineLearning\\Udemy\\Spark_Java\\winUtils\\hadoop-common-2.2.0-bin-master");
		JavaSparkContext context = new JavaSparkContext(conf);

		//create sql session
		String tempDir = "file:///C:/Manoj/Progamming/MachineLearning/Udemy/Spark_Java/spark-warehouse";
		SparkSession session = SparkSession.builder().appName("ManojLinear").master("local[2]").config("spark.sql.warehouse.dir", tempDir).getOrCreate();

		//open the file and get data as dataframe
		Dataset<Row> autoDf = session.read().option("header", "true").option("infertype", "true")
				.csv("data/auto-miles-per-gallon.csv");
		autoDf.show(5);
		autoDf.printSchema();
		

	// feature engineering
		
		//Create schema for data to be loaded in the frame
		StructType autoSchema = DataTypes.createStructType(new StructField[] {DataTypes.createStructField("MPG", DataTypes.DoubleType, false)
				,DataTypes.createStructField("CYLINDERS", DataTypes.DoubleType, false)
				,DataTypes.createStructField("DISPLACEMENT", DataTypes.DoubleType, false)
				,DataTypes.createStructField("HORSEPOWER", DataTypes.DoubleType, false)
				,DataTypes.createStructField("WEIGHT", DataTypes.DoubleType, false)
				,DataTypes.createStructField("ACCELERATION", DataTypes.DoubleType, false)
				,DataTypes.createStructField("MODELYEAR",DataTypes.DoubleType,false)
				,DataTypes.createStructField("NAME", DataTypes.StringType, false)
				});
		//broadcast avg value of HP
		Broadcast<Double> avgHp = context.broadcast(80.0);
		//change data frame back to RDD
		JavaRDD<Row> rdd1 = autoDf.toJavaRDD().repartition(2);
		
		
		//function to map
		
		JavaRDD<Row> rdd2 = rdd1.map(new Function<Row, Row>() {

			@Override
			public Row call(Row v1) throws Exception {
				Double hp = v1.getString(3).equals("?") ? avgHp.getValue() : Double.valueOf(v1.getString(3));
				
				Row ret = RowFactory.create(Double.valueOf(v1.getString(0)),
						Double.valueOf(v1.getString(1)),
						Double.valueOf(v1.getString(2)),
						hp,
						Double.valueOf(v1.getString(4)),
						Double.valueOf(v1.getString(5)),
						Double.valueOf(v1.getString(6)),
						v1.getString(7)
						);
				
				return ret;
			}
			
			
		});
		
		//create back the dataframe
		Dataset<Row> autoCleanedDf = session.createDataFrame(rdd2, autoSchema);
		System.out.println("Transformed Data:");
		autoCleanedDf.show(5);
		
		//perform coorelation analysis
		for(StructField field : autoSchema.fields()) {
			
			if(!field.dataType().equals(DataTypes.StringType)) {
				System.out.println("Correlation between MPG and "+field.name()+" "+autoCleanedDf.stat().corr("MPG",field.name()));
				
			}
			
		}
	
//prepare data for machine learning
		//convert data to labelled point structure
		JavaRDD<Row> rdd3 = autoCleanedDf.toJavaRDD().repartition(2);

		
		JavaRDD<LabeledPoint> rdd4 = rdd3.map(new Function<Row, LabeledPoint>() {
			public LabeledPoint call(Row r1)throws Exception {
				
				LabeledPoint lp = new LabeledPoint(r1.getDouble(0), Vectors.dense(r1.getDouble(2),r1.getDouble(4),r1.getDouble(5)));
				
				return lp;
			}
			
		});

		Dataset<Row> autoLp = session.createDataFrame(rdd4, LabeledPoint.class);
		autoLp.show(5);
		
		
		
//train test split
		Dataset<Row>[] rows = autoLp.randomSplit(new double[]{0.9, 0.1});
		Dataset<Row> train = rows[0];
		Dataset<Row> test = rows[1];
		
		
//build the model
		LinearRegression lr = new LinearRegression();
		LinearRegressionModel lrm = lr.fit(train);

		System.out.println("Coefficients :"+lrm.coefficients()+" Intercept :"+lrm.intercept());
		
//Predict
		Dataset<Row> predictions = lrm.transform(test);
		predictions.select("label","prediction","features").show(5);
		
//check Residual or r-squared
		RegressionEvaluator eval = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("r2");
		double r2 = eval.evaluate(predictions);
		
		System.out.println("R2 on test data :"+ r2);
		


	}

}
