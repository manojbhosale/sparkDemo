
/****************************************************************************

                   Spark with Java

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark SQL
*****************************************************************************/
package com.v2maestros.spark.bda.train;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
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

import com.v2maestros.spark.bda.common.ExerciseUtils;
import com.v2maestros.spark.bda.common.SparkConnection;

public class SparkSQLDemo {

	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		SparkSession spSession = SparkConnection.getSession();
		JavaSparkContext spContext = SparkConnection.getContext();
		
		/*-------------------------------------------------------------------
		 * Working with Data Frames
		 -------------------------------------------------------------------*/
		//Load a JSON file into a data frame.
		Dataset<Row> empDf = spSession.read().json("data/customerData.json");
		empDf.show();
		empDf.printSchema();
		
		//Do data frame queries
		System.out.println("SELECT Demo :");
		empDf.select(col("name"),col("salary")).show();
		
		System.out.println("FILTER for Age == 40 :");
		empDf.filter(col("age").equalTo(40)).show();
		
		System.out.println("GROUP BY gender and count :");
		empDf.groupBy(col("gender")).count().show();
		
		System.out.println("GROUP BY deptId and find average of salary and max of age :");
		Dataset<Row> summaryData = empDf.groupBy(col("deptid"))
			.agg(avg(empDf.col("salary")), max(empDf.col("age")));
		summaryData.show();
		
		//Create Dataframe from a list of objects
		Department dp1 = new Department("100","Sales");
		Department dp2 = new Department("200","Engineering");
		List<Department> deptList = new ArrayList<Department>();
		deptList.add(dp1);
		deptList.add(dp2);
		
		Dataset<Row> deptDf = spSession
            .createDataFrame(deptList, Department.class);
		System.out.println("Contents of Department DF : ");
		deptDf.show();
		
		System.out.println("JOIN example :");
		Dataset<Row> joinDf = empDf.join(deptDf,
                            col("deptid").equalTo(col("id")));
		joinDf.show();
		
		System.out.println("Cascading operations example : ");
		empDf.filter( col("age").gt(30) )
				.join(deptDf,col("deptid").equalTo(col("id")))
				.groupBy(col("deptid"))
				.agg(avg(empDf.col("salary")), max(empDf.col("age"))).show();

		//Create Data Frames from an CSV
		Dataset<Row> autoDf = spSession.read()
						.option("header","true")
						.csv("data/auto-data.csv");
		autoDf.show(5);
		
		//Create dataframe from Row objects and RDD
		Row iRow = RowFactory.create(1,"USA");
		Row iRow2 = RowFactory.create(2,"India");
		
		List<Row> rList = new ArrayList<Row>();
		rList.add(iRow);
		rList.add(iRow2);
		
		JavaRDD<Row> rowRDD = spContext.parallelize(rList);
		
		StructType schema = DataTypes
			.createStructType(new StructField[] {
				DataTypes.createStructField("id", DataTypes.IntegerType, false),
				DataTypes.createStructField("name", DataTypes.StringType, false) });
		
		Dataset<Row> tempDf = spSession.createDataFrame(rowRDD, schema);
		tempDf.show();
		
		/*-------------------------------------------------------------------
		 * Working with Temp tables
		 -------------------------------------------------------------------*/
		autoDf.createOrReplaceTempView("autos");
		System.out.println("Temp tables Demo : ");
		spSession.sql("select * from autos where hp > 200").show();
		spSession.sql("select make, max(rpm) from autos group by make order by 2").show();
		
		//Convert DataFrame to JavaRDD
		JavaRDD<Row> autoRDD = autoDf.rdd().toJavaRDD();
		
		//Working with Databases
		Map<String,String> jdbcOptions = new HashMap<String,String>();


		
		System.out.println("Create Dataframe from a DB Table");
		Dataset<Row> demoDf = spSession.read().format("jdbc")
				.options(jdbcOptions).load();
		demoDf.show();
		
		ExerciseUtils.hold();

	}

}
