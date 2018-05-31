package com.v2maestros.spark.bda.practice;

import static org.apache.spark.sql.functions.col;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.avro.ipc.trace.TestFileSpanStorage;
import org.apache.hadoop.yarn.api.protocolrecords.impl.pb.FinishApplicationMasterRequestPBImpl;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.util.DoubleAccumulator;

import static org.apache.spark.sql.functions.*;

public class CreditDefaultSolution {

	SparkConf config = null;

	public static void main(String[] args) throws IOException {

		// SHOW ONLY THE ERROR LOGGERS
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		// CREATE CONTEXT and SESSION
		JavaSparkContext context = SparkSettings.getSparkContext();
		SparkSession session = SparkSettings.getSparkSession();

		// LOAD INPUT FILE as DATAFRAME
		Dataset<Row> data = session.read().option("header", "true")
				// .option("inferSchema","true")
				.csv("data/credit-card-default-1000.csv");

		// CREATE SCHEMA FOR INPUT VALUES

		StructType detailsSchema = DataTypes.createStructType(
				new StructField[] { DataTypes.createStructField("CUSTID", DataTypes.IntegerType, false),
						DataTypes.createStructField("LIMIT_BAL", DataTypes.IntegerType, false),
						DataTypes.createStructField("SEX", DataTypes.IntegerType, false),
						DataTypes.createStructField("EDUCATION", DataTypes.IntegerType, false),
						DataTypes.createStructField("MARRIAGE", DataTypes.IntegerType, false),
						DataTypes.createStructField("AGE", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_1", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_2", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_3", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_4", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_5", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_6", DataTypes.IntegerType, false),
						DataTypes.createStructField("BILL_AMT1", DataTypes.IntegerType, false),
						DataTypes.createStructField("BILL_AMT2", DataTypes.IntegerType, false),
						DataTypes.createStructField("BILL_AMT3", DataTypes.IntegerType, false),
						DataTypes.createStructField("BILL_AMT4", DataTypes.IntegerType, false),
						DataTypes.createStructField("BILL_AMT5", DataTypes.IntegerType, false),
						DataTypes.createStructField("BILL_AMT6", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_AMT1", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_AMT2", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_AMT3", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_AMT4", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_AMT5", DataTypes.IntegerType, false),
						DataTypes.createStructField("PAY_AMT6", DataTypes.IntegerType, false),
						DataTypes.createStructField("DEFAULTED", DataTypes.IntegerType, false),
						DataTypes.createStructField("AVG_DELAY", DataTypes.IntegerType, false),
						DataTypes.createStructField("AVG_BILL_AMT", DataTypes.IntegerType, false),
						DataTypes.createStructField("AVG_REPAY", DataTypes.IntegerType, false),
						DataTypes.createStructField("PERCENT_REPAY", DataTypes.IntegerType, false), });

		// CREATE RDD for cleaning the garbage values in bottom
		JavaRDD<Row> rawRdd = data.toJavaRDD().repartition(2);

		// Clean the garbage values and cange the datatypes of the values

		JavaRDD<Row> cleaned = rawRdd.map(new Function<Row, Row>() {

			@Override
			public Row call(Row v1) throws Exception {

				Integer[] arr = new Integer[v1.size()];
				for (int i = 0; i < v1.size(); i++) {
					String val = v1.getString(i);

					// for filtering the non numeric lines
					if (i == 0) {
						if (!isNummeric(val)) {
							break;
						}
					}

					// for altering the M and F characters for numbers
					if (i == 2) {
						arr[i] = isNummeric(val) ? Integer.valueOf(val) : val.equals("M") ? 1 : 2;
						continue;
					}

					arr[i] = isNummeric(val) ? Integer.valueOf(val) : 0;

				}

				Row ret = RowFactory.create(arr);
				return ret;
			}

		});

		// get rid of the records with null values, created in above step.
		JavaRDD<Row> filtered = cleaned.filter(new Function<Row, Boolean>() {

			@Override
			public Boolean call(Row v1) throws Exception {
				return v1.get(0) != null;
			}

		});

		JavaRDD<Row> newCols = filtered.map(new Function<Row, Row>() {

			@Override
			public Row call(Row v1) throws Exception {
				Integer rowSize = v1.size();
				int extraCols = 4;
				Integer[] newRow = new Integer[rowSize + extraCols]; // for additional 4 columns

				newRow[rowSize] = Lab.getAverage(v1, 6, 11); // average delay
				newRow[rowSize + 1] = Lab.getAverage(v1, 12, 17); // average bill amount
				newRow[rowSize + 2] = Lab.getAverage(v1, 18, 23); // average bill payment

				newRow[rowSize + 3] = newRow[rowSize + 1] == 0 ? 100
						: (newRow[rowSize + 2] / newRow[rowSize + 1]) * 100; // percentage payed
				for (int i = 0; i < rowSize; i++) {
					if (i == 5) {
						newRow[i] = Lab.getAgeRange(v1.getInt(i));
						continue;
					}
					newRow[i] = v1.getInt(i);
				}

				return RowFactory.create(newRow);
			}

		});

		Dataset<Row> testFinal = session.createDataFrame(newCols, detailsSchema);
		// testFinal.show(10);

		// create dataframe with new Column for SEX_NAME

		// create rows
		Row male = RowFactory.create(1, "Male");
		Row female = RowFactory.create(2, "Female");
		// create sex list
		List<Row> sexName = new ArrayList<Row>();
		sexName.add(male);
		sexName.add(female);
		// create RDD
		JavaRDD<Row> sexes = context.parallelize(sexName);
		// create data type struct
		StructType sexTypes = DataTypes.createStructType(
				new StructField[] { DataTypes.createStructField("SEX_ID", DataTypes.IntegerType, false),
						DataTypes.createStructField("SEX_NAME", DataTypes.StringType, false) });
		// create dataframe
		Dataset<Row> sexDataset = session.createDataFrame(sexes, sexTypes);
		// join dataframes
		Dataset<Row> testsexJoin = testFinal.join(sexDataset, col("SEX").equalTo(col("SEX_ID")));

		// create dataframe with new Column for Education Name

		// create rows
		Row graduate = RowFactory.create(1, "graduate");
		Row university = RowFactory.create(2, "university");
		Row highSchool = RowFactory.create(3, "highSchool");
		Row others = RowFactory.create(4, "others");

		// create sex list
		List<Row> eduNames = new ArrayList<Row>();
		eduNames.add(graduate);
		eduNames.add(university);
		eduNames.add(highSchool);
		eduNames.add(others);
		// create RDD
		JavaRDD<Row> edus = context.parallelize(eduNames);
		// create data type struct
		StructType eduTypes = DataTypes.createStructType(
				new StructField[] { DataTypes.createStructField("EDU_ID", DataTypes.IntegerType, false),
						DataTypes.createStructField("EDU_NAME", DataTypes.StringType, false) });
		// create dataframe
		Dataset<Row> eduDataset = session.createDataFrame(edus, eduTypes);
		// join dataframes
		Dataset<Row> eduJoin = testsexJoin.join(eduDataset, col("EDUCATION").equalTo(col("EDU_ID")));

		// create dataframe with new Column for marriage Name

		// create rows
		Row single = RowFactory.create(1, "single");
		Row married = RowFactory.create(2, "married");
		Row othersm = RowFactory.create(3, "others");

		// create sex list
		List<Row> statusNames = new ArrayList<Row>();
		statusNames.add(single);
		statusNames.add(married);
		statusNames.add(othersm);
		// create RDD
		JavaRDD<Row> statuses = context.parallelize(eduNames);
		// create data type struct
		StructType statusTypes = DataTypes.createStructType(
				new StructField[] { DataTypes.createStructField("STATUS_ID", DataTypes.IntegerType, false),
						DataTypes.createStructField("MARR_DESC", DataTypes.StringType, false), });
		// create dataframe
		Dataset<Row> statusDataset = session.createDataFrame(statusNames, statusTypes);
		// join dataframes
		Dataset<Row> statusJoin = eduJoin.join(statusDataset, col("MARRIAGE").equalTo(col("STATUS_ID")));

		// testsexJoin.show(20);
		// statusJoin.filter(col("STATUS_ID").equalTo(2)).show();

		/*
		 * // create dataframe for use Dataset<Row> finalInput =
		 * session.createDataFrame(filtered, detailsSchema);
		 * 
		 * //finalInput.toJavaRDD().saveAsTextFile("data/testFinal.txt");
		 * 
		 * // Use of lambdas for filtering // finalInput.filter(row ->
		 * (Integer)row.get(0) > 500).show();
		 */
		statusJoin.createOrReplaceTempView("raw");
		// session.sql("select * from raw").show();
		session.sql(
				"select SEX as SEX_NAME, count(SEX) as Total , sum(DEFAULTED) as Defaults, ROUND(sum(DEFAULTED)/count(*) * 100) as PER_DEFAULT from raw group by SEX")
				.show();
		session.sql(
				"select MARR_DESC , EDU_NAME, count(*) as Total, sum(DEFAULTED) as Defaults, ROUND(sum(DEFAULTED)/count(*) * 100) as PER_DEFAULT from raw group by MARR_DESC, EDU_NAME ORDER BY 1,2")
				.show();

		session.sql("select AVG_DELAY, count(*) as Total, sum(DEFAULTED) as Defaults, ROUND(sum(DEFAULTED)/count(*)*100) as PER_DEFAULT from raw group by AVG_DELAY").show();
		
		// correlation analysis

		/*
		 * StructField[] columns = detailsSchema.fields();
		 * 
		 * for(StructField field : columns) {
		 * if(field.dataType().equals(DataTypes.StringType)) { continue; }
		 * System.out.println("Correlation between DEFAULTED and "+field.name()+" is : "
		 * +statusJoin.stat().corr(field.name(), "DEFAULTED")); }
		 * 
		 * //MARRIAGE, AGE, AVG_BILL_AMT, AVG_REPAY
		 * 
		 * JavaRDD<LabeledPoint> forModelRdd =
		 * statusJoin.toJavaRDD().repartition(2).map(new Function<Row, LabeledPoint>() {
		 * 
		 * @Override public LabeledPoint call(Row v1) throws Exception {
		 * 
		 * LabeledPoint lp = new LabeledPoint(v1.getInt(24), Vectors.dense(
		 * v1.getInt(4), v1.getInt(5), v1.getInt(26), v1.getInt(27) ));
		 * 
		 * return lp; }
		 * 
		 * });
		 * 
		 * //create train and test data sets
		 * 
		 * Dataset<Row> forModeldf = session.createDataFrame(forModelRdd,
		 * LabeledPoint.class);
		 * 
		 * Dataset<Row>[] all = forModeldf.randomSplit(new double[] {0.9,0.1});
		 * Dataset<Row> train = all[0]; Dataset<Row> test = all[1];
		 * 
		 * train.show(5); //Naive bayes classifier
		 * 
		 * NaiveBayes nbclf = new
		 * NaiveBayes().setLabelCol("label").setFeaturesCol("features");
		 * 
		 * //NaiveBayesModel nbmodel = nbclf.fit(train); //Dataset<Row> predictions =
		 * nbmodel.transform(test);
		 * 
		 * //predictions.groupBy(col("label"),col("prediction")).count().show();
		 * 
		 * MulticlassClassificationEvaluator mle = new
		 * MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol(
		 * "prediction").setMetricName("accuracy");
		 * 
		 * //double accuracy = mle.evaluate(predictions);
		 * 
		 * //System.out.println("Naive bayes accuracy is: "+ accuracy * 100 +" %");
		 * 
		 * //random forest classifier
		 * 
		 * RandomForestClassifier randomFoClf = new
		 * RandomForestClassifier().setLabelCol("label").setFeaturesCol("features");
		 * 
		 * //RandomForestClassificationModel forestModel = randomFoClf.fit(train);
		 * 
		 * //Dataset<Row> forestPredict = forestModel.transform(test);
		 * 
		 * //double forestAccuracy = mle.evaluate(forestPredict);
		 * 
		 * //System.out.println("Random forest accuracy is: "+ forestAccuracy * 100
		 * +" %");
		 * 
		 * //decision tree classifier DecisionTreeClassifier dtreeclf = new
		 * DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features");
		 * 
		 * //DecisionTreeClassificationModel dtreeml = dtreeclf.fit(train);
		 * 
		 * //Dataset<Row> dtreeres = dtreeml.transform(test);
		 * 
		 * //double dtreeAccu = mle.evaluate(dtreeres);
		 * 
		 * //System.out.println("Decision Tree accuracy is: "+ dtreeAccu * 100 +" %");
		 * 
		 * // K means clustering
		 * 
		 * //filter for required attributes
		 * 
		 * Dataset<Row> reqCols =
		 * statusJoin.select(col("SEX"),col("EDUCATION"),col("MARRIAGE"),col("AGE"));
		 * 
		 * Row means =
		 * reqCols.agg(avg(col("SEX")),avg(col("EDUCATION")),avg(col("MARRIAGE")),avg(
		 * col("AGE"))).toJavaRDD().takeOrdered(1).get(0); Row stds =
		 * reqCols.agg(stddev(col("SEX")),stddev(col("EDUCATION")),stddev(col("MARRIAGE"
		 * )),stddev(col("AGE"))).toJavaRDD().takeOrdered(1).get(0);
		 * 
		 * Broadcast<Row> bcmeans = context.broadcast(means); Broadcast<Row> bcstds =
		 * context.broadcast(stds); DoubleAccumulator idAcc =
		 * context.sc().doubleAccumulator();
		 * 
		 * 
		 * //perform scaling and centering
		 * 
		 * JavaRDD<LabeledPoint> reqColsRdd = reqCols.toJavaRDD().repartition(2).map(new
		 * Function<Row, LabeledPoint>() {
		 * 
		 * @Override public LabeledPoint call(Row v1) throws Exception {
		 * 
		 * double c1 = (v1.getInt(0) -
		 * bcmeans.value().getDouble(0))/bcstds.value().getDouble(0); double c2 =
		 * (v1.getInt(1) - bcmeans.value().getDouble(1))/bcstds.value().getDouble(1);
		 * double c3 = (v1.getInt(2) -
		 * bcmeans.value().getDouble(2))/bcstds.value().getDouble(2); double c4 =
		 * (v1.getInt(3) - bcmeans.value().getDouble(3))/bcstds.value().getDouble(3);
		 * double id = idAcc.value(); idAcc.setValue(idAcc.value()+1); return new
		 * LabeledPoint(id, Vectors.dense(c1,c2,c3,c4)); } });
		 * 
		 * 
		 * //Perform Clustering Dataset<Row> forCluster =
		 * session.createDataFrame(reqColsRdd, LabeledPoint.class);
		 * 
		 * forCluster.show(5);
		 * 
		 * KMeans kmns = new KMeans().setSeed(1L).setK(4); KMeansModel mdl =
		 * kmns.fit(forCluster);
		 * 
		 * Dataset<Row> mdl_prd = mdl.transform(forCluster);
		 * 
		 * //groupings System.out.println("Groupings !!"); mdl_prd.show(5);
		 * 
		 * 
		 * //summary System.out.println("Predictions SUmmary");
		 * mdl_prd.groupBy(col("prediction")).count().show();
		 */ }

	public static boolean isNummeric(Object obj) {
		String str = (String) obj;
		return str.matches("[+-]?\\d*\\.?\\d*");
	}

}
