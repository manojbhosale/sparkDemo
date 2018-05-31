package com.v2maestros.spark.bda.practice;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

public class SparkSettings {

	private static SparkConf conf;
	private static JavaSparkContext context;
	private static SparkSession session;
	private static String tempDir = "file:///C:/Manoj/Progamming/MachineLearning/Udemy/Spark_Java/spark-warehouse";

	
	private static Properties loadConfig() throws IOException {
		Properties prop = new Properties();
		String filename = "config.properties";
		InputStream input = CreditDefaultSolution.class.getClassLoader().getResourceAsStream(filename);
		
		if(input == null) {
			throw new FileNotFoundException("Unable to find the file !!");
		}
		prop.load(input);
		
		return prop;
	}
	
	
	public static SparkConf getSparkConfig() {
		
		if(conf != null) {
			return conf;
		}
		
		Properties prop = null;
		try {
			prop = loadConfig();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		conf = new SparkConf();
		return conf.setAppName(prop.getProperty("APPLICAION_NAME")).setMaster(prop.getProperty("MASTER_URL"));
		
		

	}


	public static JavaSparkContext getSparkContext() {
		if(conf == null) {
			getSparkConfig();
		}
		System.setProperty("hadoop.home.dir", "C:\\Manoj\\Progamming\\MachineLearning\\Udemy\\Spark_Java\\winUtils\\hadoop-common-2.2.0-bin-master");	
		context = new JavaSparkContext(conf);
		return context;
	}
	
	public static SparkSession getSparkSession() {
		if(context == null) {
			getSparkContext();
		}
		
		session =  SparkSession.
				builder().
				config(conf).
				config("spark.sql.warehouse.dir", tempDir)
				.getOrCreate();
		
		return session;
		
	}

}
