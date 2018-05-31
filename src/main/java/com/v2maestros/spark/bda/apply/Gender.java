/** File provided by V2 Maestros for its students for learning purposes only
 * Copyright @2016 All rights reserved.
 */
package com.v2maestros.spark.bda.apply;

public class Gender {

	Double sexId;
	/**
	 * @return the sexId
	 */
	public Double getSexId() {
		return sexId;
	}

	/**
	 * @param sexId the sexId to set
	 */
	public void setSexId(Double sexId) {
		this.sexId = sexId;
	}

	/**
	 * @return the sexName
	 */
	public String getSexName() {
		return sexName;
	}

	/**
	 * @param sexName the sexName to set
	 */
	public void setSexName(String sexName) {
		this.sexName = sexName;
	}

	String sexName;
	
	public Gender( Double sexId, String sexName) {
		this.sexId=sexId;
		this.sexName=sexName;
	}
}
