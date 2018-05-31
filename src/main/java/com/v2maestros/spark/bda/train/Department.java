/** File provided by V2 Maestros for its students for learning purposes only
 * Copyright @2016 All rights reserved.
 */
package com.v2maestros.spark.bda.train;

public class Department {
	String id;
	/**
	 * @return the id
	 */
	public String getId() {
		return id;
	}

	/**
	 * @param id the id to set
	 */
	public void setId(String id) {
		this.id = id;
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	String name;
	
	public Department(String id, String name){
		this.id=id;
		this.name=name;
	}
}
