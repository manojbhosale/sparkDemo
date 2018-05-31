package com.v2maestros.spark.bda.train;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.v2maestros.spark.bda.common.SparkConnection;

public class SparkSQLDemoM {
	
	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		SparkSession session = SparkConnection.getSession();
		
		JavaSparkContext context = SparkConnection.getContext();
		
		//Read and show Json file as database table
		Dataset<Row> empdf = session.read().json("data/CustomerData.json");
		empdf.show();
		//empdf.printSchema();
		// SELECT
		empdf.select(col("age"),col("gender"),col("salary")).show();
		//FILTER
		empdf.filter(col("age").gt(40)).show();
		//GROUP BY
		empdf.groupBy(col("deptid")).count().show();
		//complex GROUP BY
		Dataset<Row> result = empdf.groupBy(col("deptid")).agg(avg(col("salary")),max(col("age")));
		//result.show();
		
		//Create dataframe from list of objects
		Department d1 = new Department("100", "Sales");
		Department d2 = new Department("200", "Delivery");
		
		List<Department> dptList = new ArrayList<>();
		
		dptList.add(d1);
		dptList.add(d2);
		
		Dataset<Row> deptFrame = session.createDataFrame(dptList, Department.class);
		deptFrame.show();
		
		Dataset<Row> joinDf =  empdf.join(deptFrame, col("deptid").equalTo(col("id")));
		
		//joinDf.show();
		
		//Cascading operations example. POWERFUL
		Dataset<Row> cascade  =  empdf.filter(col("age").gt(40))
		.join(deptFrame, col("deptid").equalTo(col("id")))
		.groupBy(col("deptid"))
		.agg(avg(col("salary")),max(col("age")));
		
		//cascade.show();
		
		//create dataframes from a CSV
		Dataset<Row> autoDf = session.read()
				.option("header", "true")
				.csv("data/auto-data.csv");
		
		//autoDf.show();
		
		//create dataframe from row objects and RDD
		Row row1 = RowFactory.create(1,"USA");
		Row row12 = RowFactory.create(2,"USA");
		
		List<Row> rows = new ArrayList<>();
		
		rows.add(row1);
		rows.add(row12);
		
		JavaRDD<Row> rowRDD = context.parallelize(rows);
		
		StructType schema = DataTypes.createStructType(new StructField[] {
			DataTypes.createStructField("id", DataTypes.IntegerType, false),
			DataTypes.createStructField("name", DataTypes.StringType, true),
		});
		
		Dataset<Row> newrame = session.createDataFrame(rowRDD, schema);
		//newrame.show();
		
		
		//work with temptables
		autoDf.createOrReplaceTempView("autos");
		session.sql("select * from autos where hp > 200").show();
		session.sql("select make, max(rpm) from autos group by make order by 1").show();

		//convert dataframe to RDD
		JavaRDD<Row> autoRdd = autoDf.rdd().toJavaRDD();
		
		//working with databases
		Map<String, String> jdbcOptions = new HashMap<>();
		
		jdbcOptions.put("url", "jdbc:postgresql://localhost:5432/testdb");
		jdbcOptions.put("driver", "org.postgresql.Driver");
		jdbcOptions.put("dbtable", "contact");
		jdbcOptions.put("user", "postgres");
		jdbcOptions.put("password", "root123");
		
		//create dataframe from dbTable
		
		Dataset<Row> demoDf = session.read().format("jdbc").options(jdbcOptions).load();
		
		demoDf.show(2);

		
		
		
	}

}
